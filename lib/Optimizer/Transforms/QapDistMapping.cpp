/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/ADT/GraphCSR.h"
#include "cudaq/Support/CircuitSlicer.h"
#include "cudaq/Support/Device.h"
#include "cudaq/Support/Placement.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define DEBUG_TYPE "quantum-mapper"

using namespace mlir;

// Use specific cudaq elements without bringing in the full namespace
using cudaq::Device;
using cudaq::Placement;
using cudaq::GraphCSR;
using cudaq::QuantumMeasure;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

namespace cudaq::opt {
#define GEN_PASS_DEF_QAPDISTMAPPINGPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

//===----------------------------------------------------------------------===//
// Placement
//===----------------------------------------------------------------------===//

// void identityPlacement(Placement &placement) {
//   for (unsigned i = 0, end = placement.getNumVirtualQ(); i < end; ++i)
//     placement.map(Placement::VirtualQ(i), Placement::DeviceQ(i));
// }

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

/// This class encapsulates an quake operation that uses wires with information
/// about the virtual qubits these wires correspond.
struct VirtualOp {
  mlir::Operation *op;
  SmallVector<Placement::VirtualQ, 2> qubits;

  VirtualOp(mlir::Operation *op, ArrayRef<Placement::VirtualQ> qubits)
      : op(op), qubits(qubits) {}
};

class QAPSolver {
public:
  std::vector<std::vector<int>> fMat; // flow matrix, n by n
  std::vector<std::vector<int>> dMat; // distance matrix, n by n
  unsigned int n;                     // the number of facilities/locations
  QAPSolver(std::vector<std::vector<int>> flow,
            std::vector<std::vector<int>> distance, unsigned int size) {
    fMat = flow;
    dMat = distance;
    n = size;
  }

  ~QAPSolver() {}

  void genInitSol(std::vector<int> *arr, unsigned int iter = 10) {
    randomGen(&*arr, iter);
  }

  // Tabu Search algorithm. Takes in the initial solution, maximum iteration,
  // min tabu tenure and max tabu tenure
  void TS(std::vector<int> current, int maxIter = 10, int tmin = 7,
          int tmax = 10) {
    // generate the sequence of tenure values
    std::vector<int> tenureArr;
    // llvm::raw_ostream &os = llvm::errs();

    generateTenureList(&tenureArr, tmin, tmax);
    unsigned int tenureValInd = 0; // to determine the tenure value assigned to
                                   // each new tabu-active moves
    // initialization step
    bestSol = current;
    std::vector<std::vector<int>>
        candidateList; // list of neighborhood solution that is tabu-inactive
    std::vector<int> moveValue; // objective function value of the moves
    std::vector<std::vector<unsigned int>>
        moveList; // stores the 2-opt swap move
    int iter = 1;
    std::vector<std::vector<unsigned int>>
        tabuList; // stores the best move as a tabu-active move
    std::vector<int> tabuTenureList;

    while (iter <= maxIter) {
      // os<<"new iteration"<<"\n";
      tenureCheck(&tabuList, &tabuTenureList,
                  &iter); // check if the tenure for all tabu-active moves have
                          // expired, remove them if true
      for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
          // os<<"indices "<<i<<" "<<j<<"\n";
          std::vector<unsigned int> mv{i, j};
          // only consider a move if it is not tabu active, else check if
          // aspiration criteria is met
          if (!tabuCheck(tabuList, mv)) {
            // os<<"Ranjani checking current "<<current[i]<<"
            // "<<current[j]<<"\n";
            transpose2(current, i, j);
            candidateList.push_back(neighborSol);
            moveValue.push_back(objectiveFunction(neighborSol));
            moveList.push_back(mv);
          } else {
            // os<<"Ranjani checking current tabu"<<current[i]<<"
            // "<<current[j]<<"\n";
            transpose2(current, i, j);
            int neighborVal = objectiveFunction(neighborSol);
            if (neighborVal < objectiveFunction(bestSol)) {
              // revoke the tabu status of a move that could give better
              // solution than current best solution
              tabuRevoke(&tabuList, &tabuTenureList, mv);
              candidateList.push_back(neighborSol);
              moveValue.push_back(neighborVal);
              moveList.push_back(mv);
            }
          }
        }
      }
      // Select the best candidate, it is the moves that yield the lowest
      // objective function value
      std::vector<int>::iterator bestNeighbor =
          std::min_element(moveValue.begin(), moveValue.end());
      int bestNeighborPos = int(std::distance(moveValue.begin(), bestNeighbor));
      // os<<"best neighbor position "<<bestNeighborPos<<"
      // "<<candidateList.size()<<"\n";
      if (candidateList.size() == 0) {
        break;
      }
      // Compare the best candidate to the current best solution
      if (moveValue[bestNeighborPos] < objectiveFunction(bestSol)) {
        bestSol = candidateList[bestNeighborPos];
      }

      // best candidate is made as the current solution for the next iteration
      current = candidateList[bestNeighborPos];
      // make the move that yield the best candidate a tabu-active
      tabuAdd(&tabuList, &tabuTenureList, moveList[bestNeighborPos], &tenureArr,
              tenureValInd, iter);
      // clear the current iterations candidate list, move list and move values
      candidateList.clear();
      moveValue.clear();
      moveList.clear();
      iter++; // increase the iterations
    }
  }

  std::vector<int> getPerm() { return bestSol; }

  std::vector<int> getPermInitSol() { return initSol; }

  int getSolution() { return objectiveFunction(bestSol); }

  int getSolutionInitSol() { return objectiveFunction(initSol); }

private:
  std::vector<int>
      neighborSol; // the temporary solution from a neighborhood operation
  std::vector<int> bestSol; // the solution that gives the best value to the
                            // optimization problem
  std::vector<int> initSol; // save the initial solution generated

  // generate the initial solution randomly, set iter to non-zero to search
  // iteratively for a better initial solution *arr is the pointer to the array
  // the initial solution is write into. iter is set to 10 by default.
  void randomGen(std::vector<int> *arr, unsigned int iter = 10) {
    std::srand(std::time(nullptr));
    if (!initSol.empty()) {
      initSol.clear();
    }
    // create the array from 1 to n
    for (unsigned int i = 0; i < n; i++) {
      initSol.push_back(int(i + 1));
    }
    std::random_shuffle(initSol.begin(), initSol.end());
    int score = INT_MAX;
    std::vector<int> temp = initSol;
    while (iter != 0) {
      std::random_shuffle(initSol.begin(), initSol.end());
      int cscore = objectiveFunction(initSol);
      if (cscore < score) {
        score = cscore;
        temp = initSol;
      }
      iter--;
    }
    *arr = temp;
    initSol = temp;
  }

  // calculate the Objective Function value of the QAP
  int objectiveFunction(std::vector<int> solution) {
    // llvm::raw_ostream &os = llvm::errs();
    // os<<"Ranjani checking objective 1"<<"\n";
    int total = 0;
    for (unsigned int i = 0; i < n; i++) {
      // os<<"Ranjani checking obj indices "<<solution[i]<<"\n";
      for (unsigned int j = 0; j < n; j++) {
        if (i != j) {
          // os<<"Ranjani checking obj indices "<<i<<" "<<j<<"
          // "<<solution[j]<<"\n"; os<<"Ranjani checking obj "<<fMat[i][j]<<"
          // "<<dMat[static_cast<unsigned int>(solution[i]) -
          // 1][static_cast<unsigned int>(solution[j]) - 1]<<"\n";
          total += fMat[i][j] *
                   dMat[static_cast<unsigned int>(solution[i]) - 1]
                       [static_cast<unsigned int>(solution[j]) - 1];
        }
      }
    }
    return total;
  }

  // transposition of 2 elements
  void transpose2(std::vector<int> solution, unsigned int i, unsigned int j) {
    // llvm::raw_ostream &os = llvm::errs();
    neighborSol = solution;
    // os<<"Ranjani checking transposition 1 "<< i<<" "<<neighborSol[i]<<"\n";
    unsigned int temp = neighborSol[i];
    // os<<"Ranjani checking transposition 2 "<<j<<" "<<neighborSol[j]<<"\n";
    neighborSol[i] = neighborSol[j];
    // os<<"Ranjani checking transposition 3"<<"\n";
    neighborSol[j] = temp;
  }

  // generate the initial tenure list in ascending order between tmin and tmax
  // (inclusive)
  void generateTenureList(std::vector<int> *tenure_arr, int tmin, int tmax) {
    for (int i = tmin; i <= tmax; i++) {
      tenure_arr->push_back(i);
    }
  }

  // check if the move is tabu-active. true if it is tabu active, false
  // otherwise
  bool tabuCheck(std::vector<std::vector<unsigned int>> tList,
                 std::vector<unsigned int> move) {
    for (unsigned int i = 0; i < tList.size(); i++) {
      if (tList[i] == move) {
        return true;
      }
    }
    return false;
  }

  // revoke a tabu status based on a move
  void tabuRevoke(std::vector<std::vector<unsigned int>> *tList,
                  std::vector<int> *tenList, std::vector<unsigned int> move) {
    std::vector<std::vector<unsigned int>> t = *tList;
    std::vector<int> ten = *tenList;
    std::vector<unsigned int> moveMirror{move[1], move[0]};
    std::vector<unsigned int> pos;
    for (unsigned int i = 0; i < tList->size(); i++) {
      if (t[i] == move || t[i] == moveMirror) {
        pos.push_back(i);
      }
      if (pos.size() == 2) {
        break;
      }
    }

    tList->erase(tList->begin() + pos[0]);
    tList->erase(tList->begin() + (pos[1] - 1));
    tenList->erase(tenList->begin() + pos[0]);
    tenList->erase(tenList->begin() + (pos[1] - 1));
  }

  // add a new tabu-active move to the tabu list, assigned a different tenure at
  // each iteration using the systematic dynamic tenure principle
  void tabuAdd(std::vector<std::vector<unsigned int>> *tList,
               std::vector<int> *tenList, std::vector<unsigned int> move,
               std::vector<int> *tenure_arr, unsigned int tenureIndex,
               int iter) {
    std::vector<unsigned int> moveMirror{
        move[1], move[0]}; // create the symmetrical opposite of the swap move
    // make a move tabu by appending it and its symmetric opposite to the tabu
    // list
    tList->push_back(move);
    tList->push_back(moveMirror);
    // add the tabu tenure to the new tabu-active move
    // if the end of the sequence for the range of tabu tenure is reached,
    // reshuffle the sequence randomly and start the index at 0 again.
    if (tenureIndex == tenure_arr->size()) {
      std::random_shuffle(tenure_arr->begin(), tenure_arr->end());
      tenureIndex = 0; // restart the sequence
    }
    std::vector<int> ten_arr = *tenure_arr;
    tenList->push_back(iter + ten_arr[tenureIndex]);
    tenList->push_back(iter +
                       ten_arr[tenureIndex]); // the symmetrical opposite move
                                              // also share the same tenure
    tenureIndex = tenureIndex + 1;
  }

  // check the tenure of each tabu-active moves, revoke them if the tenure is
  // expired
  void tenureCheck(std::vector<std::vector<unsigned int>> *tList,
                   std::vector<int> *tenList, int *iter) {
    std::vector<int> ten = *tenList;
    std::vector<unsigned int> pos;
    for (unsigned int i = 0; i < ten.size(); i++) {
      if (ten[i] < *iter) {
        pos.push_back(
            i); // record all expired tabu-active tenure's index in order
      }
    }
    // revoke tabu status for all tabu-active moves with expired tenure
    for (unsigned int j = 0; j < pos.size(); j++) {
      tList->erase(tList->begin() + (pos[j] - j));
      tenList->erase(tenList->begin() + (pos[j] - j));
    }
  }
};

/// The `SabreRouter` class is modified implementation of the following paper:
/// Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem for
/// NISQ-era quantum devices." In Proceedings of the Twenty-Fourth International
/// Conference on Architectural Support for Programming Languages and Operating
/// Systems, pp. 1001-1014. 2019.
/// https://dl.acm.org/doi/pdf/10.1145/3297858.3304023
///
/// Routing starts with source operations collected during the analysis. These
/// operations form a layer, called the `frontLayer`, which is a set of
/// operations that have no unmapped predecessors. In the case of these source
/// operations, the router only needs to iterate over the front layer while
/// visiting all users of each operation. The processing of this front layer
/// will create a new front layer containing all operations that have being
/// visited the same number of times as their number of wire operands.
///
/// After processing the very first front layer, the algorithm proceeds to
/// process the newly created front layer. Once again, it processes the front
/// layer and map all operations that are compatible with the current placement,
/// i.e., one-wire operations and two-wire operations using wires that
/// correspond to qubits that are adjacently placed in the device. When an
/// operation is successfully mapped, it is removed from the front layer and all
/// its users are visited. Those users that have no unmapped predecessors are
/// added to the front layer. If the mapper cannot successfully map any
/// operation in the front layer, then it adds a swap to the circuit and tries
/// to map the front layer again. The routing process ends when the front layer
/// is empty.
///
/// Modifications from the published paper include the ability to defer
/// measurement mapping until the end, which is required for QIR Base Profile
/// programs (see the `allowMeasurementMapping` member variable).

class SabreRouter {
  using Qubit = GraphCSR::Node;
  using WireMap = DenseMap<Value, Placement::VirtualQ>;
  using Swap = std::pair<Placement::DeviceQ, Placement::DeviceQ>;

public:
  SabreRouter(Device &device, WireMap &wireMap, Placement &placement, std::vector<int> &InverseMapping,
              unsigned extendedLayerSize, float extendedLayerWeight,
              float decayDelta, unsigned roundsDecayReset)
      : device(device), wireToVirtualQ(wireMap), placement(placement), InverseMapping(InverseMapping),
        extendedLayerSize(extendedLayerSize),
        extendedLayerWeight(extendedLayerWeight), decayDelta(decayDelta),
        roundsDecayReset(roundsDecayReset),
        phyDecay(device.getNumQubits(), 1.0), phyToWire(device.getNumQubits()),
        allowMeasurementMapping(false) {}

  /// Main entry point into SabreRouter routing algorithm
  void route(Block &block, ArrayRef<quake::NullWireOp> sources);

  /// After routing, this contains the final values for all the qubits
  ArrayRef<Value> getPhyToWire() { return phyToWire; }
  int totalCost = 0;

private:
  void visitUsers(ResultRange::user_range users,
                  SmallVectorImpl<VirtualOp> &layer,
                  SmallVectorImpl<Operation *> *incremented = nullptr);

  LogicalResult mapOperation(VirtualOp &virtOp);

  LogicalResult mapFrontLayer();

  void selectExtendedLayer();

  double computeLayerCost(ArrayRef<VirtualOp> layer);
  void reMap(std::vector<int> NewMap, OpBuilder builder);

  Swap chooseSwap();

private:
  Device &device;
  WireMap &wireToVirtualQ;
  Placement &placement;
  std::vector<int> InverseMapping;

  // Parameters
  const unsigned extendedLayerSize;
  const float extendedLayerWeight;
  const float decayDelta;
  const unsigned roundsDecayReset;

  // Internal data
  SmallVector<VirtualOp> frontLayer;
  SmallVector<VirtualOp> extendedLayer;
  SmallVector<VirtualOp> measureLayer;
  llvm::SmallPtrSet<mlir::Operation *, 32> measureLayerSet;
  llvm::SmallSet<Placement::DeviceQ, 32> involvedPhy;
  SmallVector<float> phyDecay;

  SmallVector<Value> phyToWire;

  /// Keeps track of how many times an operation was visited.
  DenseMap<Operation *, unsigned> visited;

  /// Keep track of whether or not we're in the phase that allows measurements
  /// to be mapped
  bool allowMeasurementMapping;
  std::vector<int> currentMap;

#ifndef NDEBUG
  /// A logger used to emit diagnostics during the maping process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};

void SabreRouter::visitUsers(ResultRange::user_range users,
                             SmallVectorImpl<VirtualOp> &layer,
                             SmallVectorImpl<Operation *> *incremented) {
  for (auto user : users) {
    auto [entry, created] = visited.try_emplace(user, 1);
    if (!created)
      entry->second += 1;
    if (incremented)
      incremented->push_back(user);

    if (!quake::isSupportedMappingOperation(user)) {
      LLVM_DEBUG({
        auto *tmpOp = dyn_cast<mlir::Operation *>(user);
        logger.getOStream() << "WARNING: unsupported op: " << *tmpOp << '\n';
      });
    } else {
      auto wires = quake::getQuantumOperands(user);
      if (entry->second == wires.size()) {
        SmallVector<Placement::VirtualQ, 2> qubits;
        for (auto wire : wires)
          qubits.push_back(wireToVirtualQ[wire]);
        // Don't process measurements until we're ready
        if (allowMeasurementMapping || !user->hasTrait<QuantumMeasure>()) {
          layer.emplace_back(user, qubits);
        } else {
          // Add to measureLayer. Don't add duplicates.
          if (measureLayerSet.find(user) == measureLayerSet.end()) {
            measureLayer.emplace_back(user, qubits);
            measureLayerSet.insert(user);
          }
        }
      }
    }
  }
}

LogicalResult SabreRouter::mapOperation(VirtualOp &virtOp) {
  // Take the device qubits from this operation.
  SmallVector<Placement::DeviceQ, 2> deviceQubits;
  deviceQubits.clear();
  for (auto vr : virtOp.qubits)
  {
    llvm::errs() << "virtual qubits "<<vr.index<<" ";
    deviceQubits.push_back(placement.getPhy(vr));
    llvm::errs() << "physical qubits "<<placement.getPhy(vr)<<"\n";
  }
  llvm::errs() << "\n";

  // An operation cannot be mapped if it is not a measurement and uses two
  // qubits virtual qubit that are no adjacently placed.
  if (!virtOp.op->hasTrait<QuantumMeasure>() && deviceQubits.size() == 2 &&
      !device.areConnected(deviceQubits[0], deviceQubits[1])) {
    // device.dump();
    LLVM_DEBUG(logger.getOStream()
               << " Map FAILURE\n"
               << deviceQubits[0] << " " << deviceQubits[1]);
    return failure();
  }
  if (!virtOp.op->hasTrait<QuantumMeasure>() && deviceQubits.size() == 2)
    llvm::errs() << "device qubits " <<deviceQubits[0] << " " << deviceQubits[1]<< '\n';

  // Rewire the operation.
  SmallVector<Value, 2> newOpWires;
  for (auto phy : deviceQubits){
    newOpWires.push_back(phyToWire[phy.index]);
    llvm::errs() << "device wires "<<phy.index<<" "<<phyToWire[phy.index]<< '\n';
  }
  if (failed(quake::setQuantumOperands(virtOp.op, newOpWires))) {
    LLVM_DEBUG(logger.getOStream() << "Secondary Map FAILURE\n");
    return failure();
  }

  if (isa<quake::SinkOp>(virtOp.op))
    return success();

  // Update the mapping between device qubits and wires.
  for (auto &&[w, q] :llvm::zip_equal(quake::getQuantumResults(virtOp.op), deviceQubits))
  {
    llvm::errs() << "wires " <<w<<" "<<wireToVirtualQ[w]<< '\n';
    phyToWire[q.index] = w;
  }
     

  return success();
}

LogicalResult SabreRouter::mapFrontLayer() {
  bool mappedAtLeastOne = false;
  SmallVector<VirtualOp> newFrontLayer;

  LLVM_DEBUG({
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
  });
  for (auto virtOp : frontLayer) {
    LLVM_DEBUG({
      logger.startLine() << "* ";
      virtOp.op->print(logger.getOStream(),
                       OpPrintingFlags().printGenericOpForm());
    });
    if (failed(mapOperation(virtOp))) {
      LLVM_DEBUG(logger.getOStream() << " --> FAILURE\n");
      newFrontLayer.push_back(virtOp);
      for (auto vr : virtOp.qubits)
        involvedPhy.insert(placement.getPhy(vr));
      LLVM_DEBUG({
        auto phy0 = placement.getPhy(virtOp.qubits[0]);
        auto phy1 = placement.getPhy(virtOp.qubits[1]);
        logger.indent();
        logger.startLine() << "+ virtual qubits: " << virtOp.qubits[0] << ", "
                           << virtOp.qubits[1] << '\n';
        logger.startLine() << "+ device qubits: " << phy0 << ", " << phy1
                           << '\n';
        logger.unindent();
      });
      continue;
    }
    LLVM_DEBUG(logger.getOStream() << " --> SUCCESS\n");
    mappedAtLeastOne = true;
    visitUsers(virtOp.op->getUsers(), newFrontLayer);
  }
  LLVM_DEBUG(logger.unindent());
  frontLayer = std::move(newFrontLayer);
  return mappedAtLeastOne ? success() : failure();
}


void SabreRouter::selectExtendedLayer() {
  extendedLayer.clear();
  SmallVector<Operation *, 20> incremented;
  SmallVector<VirtualOp> tmpLayer = frontLayer;
  while (!tmpLayer.empty() && extendedLayer.size() < extendedLayerSize) {
    SmallVector<VirtualOp> newTmpLayer;
    for (VirtualOp &virtOp : tmpLayer)
      visitUsers(virtOp.op->getUsers(), newTmpLayer, &incremented);
    for (VirtualOp &virtOp : newTmpLayer)
      // We only add operations that can influence placement to the extended
      // frontlayer, i.e., quantum operators that use two qubits.
      if (!virtOp.op->hasTrait<QuantumMeasure>() &&
          quake::getQuantumOperands(virtOp.op).size() == 2)
        extendedLayer.emplace_back(virtOp);
    tmpLayer = std::move(newTmpLayer);
  }

  for (auto virtOp : incremented)
    visited[virtOp] -= 1;
}

double SabreRouter::computeLayerCost(ArrayRef<VirtualOp> layer) {
  double cost = 0.0;
  for (VirtualOp const &virtOp : layer) {
    auto phy0 = placement.getPhy(virtOp.qubits[0]);
    auto phy1 = placement.getPhy(virtOp.qubits[1]);
    cost += device.getWeightedDistance(phy0, phy1) - 1;
  }
  return cost / layer.size();
}

SabreRouter::Swap SabreRouter::chooseSwap() {
  // Obtain SWAP candidates
  SmallVector<Swap> candidates;
  for (auto phy0 : involvedPhy)
    for (auto phy1 : device.getNeighbours(phy0))
      candidates.emplace_back(phy0, phy1);

  if (extendedLayerSize)
    selectExtendedLayer();

  // Compute cost
  SmallVector<double> cost;
  for (auto [phy0, phy1] : candidates) {
    placement.swap(phy0, phy1);
    double swapCost = computeLayerCost(frontLayer);
    double maxDecay = std::max(phyDecay[phy0.index], phyDecay[phy1.index]);

    if (!extendedLayer.empty()) {
      double extendedLayerCost =
          computeLayerCost(extendedLayer) / extendedLayer.size();
      swapCost /= frontLayer.size();
      swapCost += extendedLayerWeight * extendedLayerCost;
    }

    cost.emplace_back(maxDecay * swapCost);
    placement.swap(phy0, phy1);
  }
  // Find and return the swap with minimal cost
  std::size_t minIdx = 0u;
  for (std::size_t i = 1u, end = cost.size(); i < end; ++i)
    if (cost[i] < cost[minIdx])
      minIdx = i;

  LLVM_DEBUG({
    logger.startLine() << "Choosing a swap:\n";
    logger.indent();
    logger.startLine() << "Involved device qubits:";
    for (auto phy : involvedPhy)
      logger.getOStream() << " " << phy;
    logger.getOStream() << "\n";
    logger.startLine() << "Swap candidates:\n";
    logger.indent();
    for (auto &&[qubits, c] : llvm::zip_equal(candidates, cost))
      logger.startLine() << "* " << qubits.first << ", " << qubits.second
                         << " (cost = " << c << ")\n";
    logger.getOStream() << "\n";
    logger.unindent();
    logger.startLine() << "Selected swap: " << candidates[minIdx].first << ", "
                       << candidates[minIdx].second << '\n';
    logger.unindent();
  });
  return candidates[minIdx];
}

void SabreRouter::reMap(std::vector<int> NewMap, OpBuilder builder) {
  llvm::raw_ostream &os = llvm::errs();
  auto wireType = builder.getType<quake::WireType>();
  auto addSwap = [&](Placement::DeviceQ q0, Placement::DeviceQ q1) {
    placement.swap(q0, q1);
    auto swap = builder.create<quake::SwapOp>(
        builder.getUnknownLoc(), TypeRange{wireType, wireType}, false,
        ValueRange{}, ValueRange{},
        ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
        DenseBoolArrayAttr{});
    phyToWire[q0.index] = swap.getResult(0);
    phyToWire[q1.index] = swap.getResult(1);
    // wireToVirtualQ[swap.getResult(0)]=placement.getVr(q0);
    // wireToVirtualQ[swap.getResult(1)]=placement.getVr(q1);
    //os<<"[updating current map ("<<placement.getVr(q0).index<<":"<<q1.index<<") "<<placement.getVr(q1).index<<": "<<q0.index<<") "<<"] ";
    currentMap[placement.getVr(q0).index] = q0.index + 1;
    currentMap[placement.getVr(q1).index] = q1.index + 1;
  };
  int numQubits = device.getNumQubits();
  device.RecomputeChildren();
  std::vector<int> InverseNewMap(numQubits,0);
  os<<"Remapping \n";
  for (int i = 0; i < numQubits; i++) {
    InverseNewMap[NewMap[i]-1]=i;
  }

  while (!device.IsEmptyMST()){
    int leaf = device.PickRandomLeaf();
    os<<"need to get from  "<<currentMap[InverseNewMap[leaf]]-1<<" to "<<leaf<<" for "<<InverseNewMap[leaf]<<"\n";
    auto path =device.getMSTPath(Qubit(currentMap[InverseNewMap[leaf]] - 1), Qubit(leaf));
    Qubit init = Qubit(currentMap[InverseNewMap[leaf]] - 1);
    for (auto node : path) {
      if (init == node)
        continue;
      os<<"swapping "<<init.index<<" "<<node.index<<" ";
      addSwap(init, node);
      totalCost += device.getWeightedDistance(init, node);
      init = node;
    }
    os<<"\n";

  }
  // for (int i = 0; i < numQubits; i++) {
  //   auto path =
  //       device.getShortestPath(Qubit(currentMap[i] - 1), Qubit(NewMap[i] - 1));
  //   //os<<"need to get from  "<<currentMap[i]-1<<" to "<<NewMap[i]-1<<" for "<<i<<"\n";
  //   // path_rev=device.getShortestPath(Qubit(NewMap[i]),Qubit(currentMap[i]));
  //   Qubit init = Qubit(currentMap[i] - 1);
  //   for (auto node : path) {
  //     if (init == node)
  //       continue;
  //     //os<<"swapping "<<init.index<<" "<<node.index<<" ";
  //     addSwap(init, node);
  //     totalCost += device.getWeightedDistance(init, node);
  //     init = node;
  //   }
  //   //os<<"Remapped \n";
  // }
}

void SabreRouter::route(Block &block, ArrayRef<quake::NullWireOp> sources) {
#ifndef NDEBUG
  constexpr char logLineComment[] =
      "//===-------------------------------------------===//\n";
#endif

  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logLineComment;
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
    for (auto virtOp : sources)
      logger.startLine() << "* " << virtOp << " --> SUCCESS\n";
    logger.unindent();
    logger.startLine() << logLineComment;
  });
  llvm::raw_ostream &os = llvm::errs();
  // The source ops can always be mapped.
  for (quake::NullWireOp nullWire : sources) {
    visitUsers(nullWire->getUsers(), frontLayer);
    Value wire = nullWire.getResult();
    auto phy = placement.getPhy(wireToVirtualQ[wire]);
    phyToWire[phy.index] = wire;
  }

  OpBuilder builder(&block, block.begin());
  auto wireType = builder.getType<quake::WireType>();
  auto addSwap = [&](Placement::DeviceQ q0, Placement::DeviceQ q1) {
    placement.swap(q0, q1);
    auto swap = builder.create<quake::SwapOp>(
        builder.getUnknownLoc(), TypeRange{wireType, wireType}, false,
        ValueRange{}, ValueRange{},
        ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
        DenseBoolArrayAttr{});
    phyToWire[q0.index] = swap.getResult(0);
    phyToWire[q1.index] = swap.getResult(1);
  };

  std::size_t numSwapSearches = 0;
  bool done = false;
  os<<"routing \n";
  int numQubits=device.getNumQubits();
  for (int i=0;i<numQubits;i++){
    currentMap.push_back(i+1);
  }
  reMap(InverseMapping, builder);
  while (!done) {
    // Once frontLayer is empty, grab everything from measureLayer and go again.
    if (frontLayer.empty()) {
      if (allowMeasurementMapping) {
        done = true;
      } else {
        allowMeasurementMapping = true;
        frontLayer = std::move(measureLayer);
      }
      continue;
    }

    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.getOStream() << ""
                          << " baaaaaaaa\n";
      logger.startLine() << logLineComment;
    });

    if (succeeded(mapFrontLayer()))
      continue;

    LLVM_DEBUG(logger.getOStream() << "\n";);

    // Add a swap
    numSwapSearches++;
    LLVM_DEBUG(logger.getOStream() << "I'm choosing a swap \n";);
    auto [phy0, phy1] = chooseSwap();
    addSwap(phy0, phy1);
    totalCost += device.getWeightedDistance(phy0, phy1);
    involvedPhy.clear();

    // Update decay
    if ((numSwapSearches % roundsDecayReset) == 0) {
      std::fill(phyDecay.begin(), phyDecay.end(), 1.0);
    } else {
      phyDecay[phy0.index] += decayDelta;
      phyDecay[phy1.index] += decayDelta;
    }
  }
  LLVM_DEBUG(logger.startLine() << '\n' << logLineComment << '\n';);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct Mapper : public cudaq::opt::impl::QapDistMappingPassBase<Mapper> {
  using QapDistMappingPassBase::QapDistMappingPassBase;

  /// Device dimensions that come from inside the `device` option parenthesis,
  /// like X and Y for star(X,Y)
  std::size_t deviceDim[2];

  enum DeviceTopologyEnum { Unknown, Path, Ring, Star, Grid, File, Bypass };
  DeviceTopologyEnum deviceTopoType;

  /// If the deviceTopoType is File, this is the path to the file.
  StringRef deviceFilename;

  virtual LogicalResult initialize(MLIRContext *context) override {
    // Initialize prior to parsing
    deviceDim[0] = deviceDim[1] = 0;

    // Get device
    StringRef deviceDef = device;
    StringRef deviceTopoStr =
        deviceDef.take_front(deviceDef.find_first_of('('));

    // Trim the dimensions off of `deviceDef` if dimensions were provided in the
    // string
    if (deviceTopoStr.size() < deviceDef.size())
      deviceDef = deviceDef.drop_front(deviceTopoStr.size());

    if (deviceTopoStr.equals_insensitive("file")) {
      if (deviceDef.consume_front("(")) {
        deviceDef = deviceDef.ltrim();
        if (deviceDef.consume_back(")")) {
          deviceFilename = deviceDef;
          // Remove any leading and trailing single quotes that may have been
          // added in order to pass files with spaces into the pass (required
          // for parsePassPipeline).
          if (deviceFilename.size() >= 2 && deviceFilename.front() == '\'' &&
              deviceFilename.back() == '\'')
            deviceFilename = deviceFilename.drop_front(1).drop_back(1);
          // Make sure the file exists before continuing
          if (!llvm::sys::fs::exists(deviceFilename)) {
            llvm::errs() << "Path " << deviceFilename << " does not exist\n";
            return failure();
          }
        } else {
          llvm::errs() << "Missing closing ')' in device option\n";
          return failure();
        }
      } else {
        llvm::errs() << "Filename must be provided in device option like "
                        "file(/full/path/to/device_file.txt): "
                     << device.getValue() << '\n';
        return failure();
      }
    } else {
      if (deviceDef.consume_front("(")) {
        deviceDef = deviceDef.ltrim();
        deviceDef.consumeInteger(/*Radix=*/10, deviceDim[0]);
        deviceDef = deviceDef.ltrim();
        if (deviceDef.consume_front(","))
          deviceDef.consumeInteger(/*Radix=*/10, deviceDim[1]);
        deviceDef = deviceDef.ltrim();
        if (!deviceDef.consume_front(")")) {
          llvm::errs() << "Missing closing ')' in device option\n";
          return failure();
        }
      }
    }

    deviceTopoType = llvm::StringSwitch<DeviceTopologyEnum>(deviceTopoStr)
                         .Case("path", Path)
                         .Case("ring", Ring)
                         .Case("star", Star)
                         .Case("grid", Grid)
                         .Case("file", File)
                         .Case("bypass", Bypass)
                         .Default(Unknown);
    if (deviceTopoType == Unknown) {
      llvm::errs() << "Unknown device option: " << deviceTopoStr << '\n';
      return failure();
    }

    return success();
  }

  /// Add `op` and all of its users into `opsToMoveToEnd`. `op` may not be
  /// nullptr.
  void addOpAndUsersToList(Operation *op,
                           SmallVectorImpl<Operation *> &opsToMoveToEnd) {
    opsToMoveToEnd.push_back(op);
    for (auto user : op->getUsers())
      addOpAndUsersToList(user, opsToMoveToEnd);
  }

  void runOnOperation() override {

    // Allow enabling debug via pass option `debug`
#ifndef NDEBUG
    if (debug) {
      llvm::DebugFlag = true;
      llvm::setCurrentDebugType(DEBUG_TYPE);
    }
#endif

    auto func = getOperation();
    auto &blocks = func.getBlocks();

    // Current limitations:
    //  * Can only map a entry-point kernel
    //  * The kernel can only have one block

    if (deviceTopoType == Bypass)
      return;

    // FIXME: Add the ability to handle multiple blocks.
    if (blocks.size() > 1) {
      func.emitError("The mapper cannot handle multiple blocks");
      signalPassFailure();
      return;
    }

    // Sanity checks and create a wire to virtual qubit mapping.
    Block &block = *blocks.begin();
    SmallVector<quake::NullWireOp> sources;
    SmallVector<Operation *> sinksToRemove;
    DenseMap<Value, Placement::VirtualQ> wireToVirtualQ;
    SmallVector<std::size_t> userQubitsMeasured;
    DenseMap<std::size_t, Value> finalQubitWire;
    for (Operation &op : block.getOperations()) {
      if (auto qop = dyn_cast<quake::NullWireOp>(op)) {
        // Assign a new virtual qubit to the resulting wire.
        wireToVirtualQ[qop.getResult()] = Placement::VirtualQ(sources.size());
        finalQubitWire[sources.size()] = qop.getResult();
        sources.push_back(qop);
      } else if (quake::isSupportedMappingOperation(&op)) {
        // Make sure the operation is using value semantics.
        if (!quake::isLinearValueForm(&op)) {
          llvm::errs() << "This is not SSA form: " << op << '\n';
          llvm::errs() << "isa<quake::NullWireOp>() = "
                       << isa<quake::NullWireOp>(&op) << '\n';
          llvm::errs() << "isAllReferences() = " << quake::isAllReferences(&op)
                       << '\n';
          llvm::errs() << "isWrapped() = " << quake::isWrapped(&op) << '\n';
          func.emitError("The mapper requires value semantics.");
          signalPassFailure();
          return;
        }

        // Since `quake.sink` operations do not generate new wires, we don't
        // need to further analyze.
        if (isa<quake::SinkOp>(op)) {
          sinksToRemove.push_back(&op);
          continue;
        }

        // Get the wire operands and check if the operators uses at most two
        // qubits. N.B: Measurements do not have this restriction.
        auto wireOperands = quake::getQuantumOperands(&op);
        if (!op.hasTrait<QuantumMeasure>() && wireOperands.size() > 2) {
          func.emitError("Cannot map a kernel with operators that use more "
                         "than two qubits.");
          signalPassFailure();
          return;
        }

        // Save which qubits are measured
        if (isa<quake::MeasurementInterface>(op))
          for (const auto &wire : wireOperands)
            userQubitsMeasured.push_back(wireToVirtualQ[wire].index);

        // Map the result wires to the appropriate virtual qubits.
        for (auto &&[wire, newWire] :
             llvm::zip_equal(wireOperands, quake::getQuantumResults(&op))) {
          // Don't use wireToVirtualQ[a] = wireToVirtualQ[b]. It will work
          // *most* of the time but cause memory corruption other times because
          // DenseMap references can be invalidated upon insertion of new pairs.
          wireToVirtualQ.insert(
              {newWire,
               wireToVirtualQ[wire]}); // Ranjani: Is this what Eric was talking
                                       // about- infinite wires?
          finalQubitWire[wireToVirtualQ[wire].index] = newWire;
        }
      }
    }
    // Make a local copy of device dimensions since we may need to modify it.
    // Otherwise multi-threaded operation may cause undefined behavior.
    std::size_t x = deviceDim[0];
    std::size_t y = deviceDim[1];
    std::size_t deviceNumQubits = deviceTopoType == Grid ? x * y : x;

    if (deviceNumQubits && sources.size() > deviceNumQubits) {
      signalPassFailure();
      return;
    }

    if (!deviceNumQubits) {
      x = deviceTopoType == Grid ? std::sqrt(sources.size()) : sources.size();
      y = x;
    }

    // These are captured in the user help (device options in Passes.td), so if
    // you update this, be sure to update that as well.
    Device d;
    if (deviceTopoType == Path)
      // d = Device::path(x);
      d = Device::path2(x, y);
    else if (deviceTopoType == Ring)
      d = Device::ring(x);
    else if (deviceTopoType == Star)
      d = Device::star(/*numQubits=*/x, /*centerQubit=*/y);
    else if (deviceTopoType == Grid)
      d = Device::grid(/*width=*/x, /*height=*/y);
    else if (deviceTopoType == File)
      d = Device::file(deviceFilename);

    if (d.getNumQubits() == 0) {
      func.emitError("Trying to target an empty device.");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({ d.dump(); });

    if (sources.size() > d.getNumQubits()) {
      func.emitError("Your device [" + device + "] has fewer qubits [" +
                     std::to_string(d.getNumQubits()) +
                     "] than your program is " + "attempting to use [" +
                     std::to_string(sources.size()) + "]");
      signalPassFailure();
      return;
    }

    // We've made it past all the initial checks. Remove the sinks now. They
    // will be added back in when the mapping is complete.
    for (auto sink : sinksToRemove)
      sink->erase();
    sinksToRemove.clear();
    std::vector<std::vector<int>> flow(d.getNumQubits(),
                                       std::vector<int>(d.getNumQubits(), 0));
    std::vector<std::vector<int>> distance = d.DistMatrix();
    for (Operation &op : block.getOperations()) {
      if (auto qop = dyn_cast<quake::NullWireOp>(op)) {
        llvm::dbgs() << "checking placement "<<qop<<"\n";
      } else if (quake::isSupportedMappingOperation(&op)) {
        // Make sure the operation is using value semantics.
        if (!quake::isLinearValueForm(&op)) {
          llvm::errs() << "This is not SSA form: " << op << '\n';
          llvm::errs() << "isa<quake::NullWireOp>() = "
                       << isa<quake::NullWireOp>(&op) << '\n';
          llvm::errs() << "isAllReferences() = " << quake::isAllReferences(&op)
                       << '\n';
          llvm::errs() << "isWrapped() = " << quake::isWrapped(&op) << '\n';
          func.emitError("The mapper requires value semantics.");
          signalPassFailure();
          return;
        }

        // Since `quake.sink` operations do not generate new wires, we don't
        // need to further analyze.
        if (isa<quake::SinkOp>(op)) {
          continue;
        }

        // Get the wire operands and check if the operators uses at most two
        // qubits. N.B: Measurements do not have this restriction.
        auto wireOperands = quake::getQuantumOperands(&op);
        if (!op.hasTrait<QuantumMeasure>() && wireOperands.size() > 2) {
          func.emitError("Cannot map a kernel with operators that use more "
                         "than two qubits.");
          signalPassFailure();
          return;
        }
        int i = 0;
        Value prev;
        for (auto wireOp : quake::getQuantumOperands(&op)) {
          // LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: zip" <<wireOp<<"\n");
          if (i == 0) {
            prev = wireOp;
            i++;
            continue;
          }
          flow[wireToVirtualQ[wireOp].index][wireToVirtualQ[prev].index]++;
          flow[wireToVirtualQ[prev].index][wireToVirtualQ[wireOp].index]++;
        }
      }
    }
    for (unsigned int i = 0; i < d.getNumQubits(); i++) {
      for (unsigned int j = 0; j < d.getNumQubits(); j++) {
        if (i != j) {
          llvm::dbgs() << "flow entry " << i << " " << j << " " << flow[i][j]
                       << "\n";
        }
      }
    }
    
    unsigned int numQubits = d.getNumQubits();
    QAPSolver QAPInstance = QAPSolver(flow, distance, numQubits);
    std::vector<int> initial;
    QAPInstance.genInitSol(&initial, 100);
    llvm::dbgs() << "Ranjani checking: Got init random ";
    QAPInstance.TS(initial, 50, 20, 30);
    llvm::dbgs() << "Ranjani checking: Objective val "
                 << QAPInstance.getSolution() << "\n";
    auto solution = QAPInstance.getPerm();
    Placement placement(sources.size(), d.getNumQubits());
    std::vector<int> Inversesolution(numQubits,0);
    
    for ( unsigned int i = 0; i < d.getNumQubits(); i++) {
      llvm::dbgs() << "soln entry " << i << " " << solution[i]-1 << "\n";
      Inversesolution[solution[i]-1]=i;
    }
    for (unsigned int i = 0; i < placement.getNumVirtualQ(); i++) {
      placement.map(Placement::VirtualQ(i),Placement::DeviceQ(i));
    }
    llvm::dbgs() << "placed qubits \n ";
    // Add implicit measurements if necessary
    if (userQubitsMeasured.empty()) {
      OpBuilder builder(&block, block.begin());
      builder.setInsertionPoint(block.getTerminator());
      auto measTy = quake::MeasureType::get(builder.getContext());
      auto wireTy = quake::WireType::get(builder.getContext());
      Type resTy = builder.getI1Type();
      for (unsigned i = 0; i < sources.size(); i++) {
        auto measureOp = builder.create<quake::MzOp>(finalQubitWire[i].getLoc(),
                                                     TypeRange{measTy, wireTy},
                                                     finalQubitWire[i]);
        builder.create<quake::DiscriminateOp>(finalQubitWire[i].getLoc(), resTy,
                                              measureOp.getMeasOut());

        wireToVirtualQ.insert(
            {measureOp.getWires()[0], wireToVirtualQ[finalQubitWire[i]]});

        userQubitsMeasured.push_back(i);
      }
    }

    // Save the order of the measurements. They are not allowed to change.
    SmallVector<mlir::Operation *> measureOrder;
    func.walk([&](quake::MeasurementInterface measure) {
      measureOrder.push_back(measure);
      for (auto user : measure->getUsers())
        measureOrder.push_back(user);
      return WalkResult::advance();
    });

    // Create auxillary qubits if needed. Place them after the last allocated
    // qubit
    unsigned numOrigQubits = sources.size();
    OpBuilder builder(&block, block.begin());
    builder.setInsertionPointAfter(sources[sources.size() - 1]);
    for (unsigned i = sources.size(); i < d.getNumQubits(); i++) {
      auto nullWireOp = builder.create<quake::NullWireOp>(
          builder.getUnknownLoc(), quake::WireType::get(builder.getContext()));
      // wireToVirtualQ[nullWireOp.getResult()] =
      //     Placement::VirtualQ(sources.size());
      sources.push_back(nullWireOp);
    }
    
    // Place
    // Placement placement(sources.size(), d.getNumQubits());
    // identityPlacement(placement);

    // Route
    SabreRouter router(d, wireToVirtualQ, placement, solution, extendedLayerSize,
                       extendedLayerWeight, decayDelta, roundsDecayReset);
    router.route(*blocks.begin(), sources);
    sortTopologically(&block);

    // Ensure that the original measurement ordering is still honored by moving
    // the measurements to the end (in their original order). Note that we must
    // move the users of those measurements to the end as well.
    for (Operation *measure : measureOrder) {
      SmallVector<Operation *> opsToMoveToEnd;
      addOpAndUsersToList(measure, opsToMoveToEnd);
      for (Operation *op : opsToMoveToEnd)
        block.getOperations().splice(std::prev(block.end()),
                                     block.getOperations(), op->getIterator());
    }

    // Remove any auxillary qubits that did not get used. Remove from the end
    // and stop once you hit a used one. If you removed from the middle, you
    // would renumber the qubits, which would invalidate the mapping indices.
    unsigned numRemaining = numOrigQubits;
    for (unsigned i = sources.size() - 1; i >= numOrigQubits; i--) {
      if (sources[i]->use_empty()) {
        sources[i]->erase();
      } else {
        numRemaining = i + 1;
        break;
      }
    }
    // Add sinks where needed
    builder.setInsertionPoint(block.getTerminator());
    auto phyToWire = router.getPhyToWire();
    for (unsigned i = 0; i < numRemaining; i++)
      builder.create<quake::SinkOp>(phyToWire[i].getLoc(), phyToWire[i]);

    // Populate mapping_v2p attribute on this function such that:
    // - mapping_v2p[v] contains the final physical qubit placement for virtual
    //   qubit `v`.
    // To map the backend qubits back to the original user program (i.e. before
    // this pass), run something like this:
    //   for (int v = 0; v < numQubits; v++)
    //     dataForOriginalQubit[v] = dataFromBackendQubit[mapping_v2p[v]];
    llvm::SmallVector<Attribute> attrs(numOrigQubits);
    Placement placement2(sources.size(), d.getNumQubits());
    for (unsigned int i = 0; i < placement2.getNumVirtualQ(); i++) {
      placement2.map(Placement::VirtualQ(i),
                     Placement::DeviceQ(solution[i] - 1));
    }
    for (unsigned int v = 0; v < numOrigQubits; v++)
      attrs[v] =
          IntegerAttr::get(builder.getIntegerType(64),
                           placement2.getPhy(Placement::VirtualQ(v)).index);

    func->setAttr("mapping_v2p", builder.getArrayAttr(attrs));
    llvm::dbgs() << "Total cost " << router.totalCost << "\n";

    // Now populate mapping_reorder_idx attribute. This attribute will be used
    // by downstream processing to reconstruct a global register as if mapping
    // had not occurred. This is important because the global register is
    // required to be sorted by qubit allocation order, and mapping can change
    // that apparent order AND introduce ancilla qubits that we don't want to
    // appear in the final global register.

    // pair is <first=virtual, second=physical>
    using VirtPhyPairType = std::pair<std::size_t, std::size_t>;
    llvm::SmallVector<VirtPhyPairType> measuredQubits;
    measuredQubits.reserve(userQubitsMeasured.size());
    for (auto mq : userQubitsMeasured) {
      measuredQubits.emplace_back(
          mq, placement.getPhy(Placement::VirtualQ(mq)).index);
    }
    // First sort the pairs according to the physical qubits.
    llvm::sort(measuredQubits,
               [&](const VirtPhyPairType &a, const VirtPhyPairType &b) {
                 return a.second < b.second;
               });
    // Now find out how to reorder `measuredQubits` such that the elements are
    // ordered based on the *virtual* qubits (i.e. measuredQubits[].first).
    llvm::SmallVector<std::size_t> reorder_idx(measuredQubits.size());
    for (std::size_t ix = 0; auto &element : reorder_idx)
      element = ix++;
    llvm::sort(reorder_idx, [&](const std::size_t &i1, const std::size_t &i2) {
      return measuredQubits[i1].first < measuredQubits[i2].first;
    });
    // After kernel execution is complete, you can pass reorder_idx[] into
    // sample_result::reorder() in order to undo the ordering change to the
    // global register that the mapping pass induced.
    llvm::SmallVector<Attribute> mapping_reorder_idx(reorder_idx.size());
    for (std::size_t ix = 0; auto &element : mapping_reorder_idx)
      element = IntegerAttr::get(builder.getIntegerType(64), reorder_idx[ix++]);

    func->setAttr("mapping_reorder_idx",
                  builder.getArrayAttr(mapping_reorder_idx));
  }
};

} // namespace