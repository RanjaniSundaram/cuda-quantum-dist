/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/ADT/GraphCSR.h"
#include "cudaq/Support/Graph.h"
#include "llvm/Support/MemoryBuffer.h"

namespace cudaq {

/// The `Device` class represents a device topology with qubits and connections
/// between qubits. It contains various methods to construct the device based on
/// canned geometries, and it contains helper methods to determine paths between
/// qubits.
class CircuitGraph {
public:
  using Qubit = GraphCSR::Node;
  using Path = mlir::SmallVector<Qubit>;
  int numComponents;
  /// Read device connectivity info from a file. The input format is the same
  /// as the Graph dump() format.
  
  /// Create a device with a path topology.
  ///
  ///  0 -- 1 -- ... -- N
  ///
  static CircuitGraph path(unsigned numQubits) {
    assert(numQubits > 0);
    CircuitGraph circuit;
    circuit.topology.createNode();
    for (unsigned i = 1u; i < numQubits; ++i) {
      circuit.topology.createNode();
      circuit.topology.addWeightedEdge(Qubit(i - 1), Qubit(i));
    }
    circuit.computeAllPairShortestPaths();
    circuit.ComputeComponents();
    return circuit;
  }


  /// TODO: Implement a method to load device info from a file.

  /// Returns the number of physical qubits in the device.
  unsigned getNumQubits() const { return topology.getNumNodes(); }

  /// Returns the distance between two qubits.

  unsigned getWeightedDistance(Qubit src, Qubit dst) const {
    unsigned pairID = getPairID(src.index, dst.index);
    return src == dst ? 0 : shortestPathsWeights[pairID];
  }

  

  unsigned getDistance(Qubit src, Qubit dst) const {
    unsigned pairID = getPairID(src.index, dst.index);
    return src == dst ? 0 : shortestPaths[pairID].size() - 1;
  }
  mlir::ArrayRef<Qubit> getNeighbours(Qubit src) const {
    return topology.getNeighbours(src);
  }
  
  int getNumComponents(){
    return numComponents;
  }

  bool areConnected(Qubit q0, Qubit q1) const {
    return getDistance(q0, q1) == 1;
  }
  int getComponent(unsigned q){
    return ComponentMap[q];
  }
  /// Returns a shortest path between two qubits.
  Path getShortestPath(Qubit src, Qubit dst) const {
    unsigned pairID = getPairID(src.index, dst.index);
    if (src.index > dst.index)
      return Path(llvm::reverse(shortestPaths[pairID]));
    return Path(shortestPaths[pairID]);
  }

  void dump(llvm::raw_ostream &os = llvm::errs()) const {
    os << "Graph:\n";
    topology.dump(os);
    os << "dumping weights \n";
    os << "\nShortest Paths:\n";
    for (unsigned src = 0; src < getNumQubits(); ++src)
      for (unsigned dst = 0; dst < getNumQubits(); ++dst) {
        auto path = getShortestPath(Qubit(src), Qubit(dst));
        os << '(' << src << ", " << dst <<", "<<getWeightedDistance(Qubit(src),Qubit(dst))<< ") : {";
        llvm::interleaveComma(path, os);
        os << "}\n";
      }
  }

private:
  using PathRef = mlir::ArrayRef<Qubit>;

  /// Returns a unique id for a pair of values (`u` and `v`). `getPairID(u, v)`
  /// will be equal to `getPairID(v, u)`.
  unsigned getPairID(unsigned u, unsigned v) const {
    if (u > v)
      std::swap(u, v);
    return (u * getNumQubits()) - (((u - 1) * u) / 2) + v - u;
  }

  /// Compute the shortest path between every qubit. This assumes that there
  /// exists at least one path between every source and destination pair. I.e.
  /// the graph cannot be bipartite.
  /*
  void computeAllPairShortestPaths() {
    std::size_t numNodes = topology.getNumNodes();
    shortestPaths.resize(numNodes * (numNodes + 1) / 2);
    mlir::SmallVector<Qubit> path(numNodes);
    for (unsigned n = 0; n < numNodes; ++n) {
      auto parents = getShortestPathsBFS(topology, Qubit(n));
      // Reconstruct the paths
      for (auto m = n + 1; m < numNodes; ++m) {
        path.clear();
        path.push_back(Qubit(m));
        auto p = parents[m];
        while (p != Qubit(n)) {
          path.push_back(p);
          p = parents[p.index];
        }
        path.push_back(Qubit(n));
        pathsData.append(path.rbegin(), path.rend());
        shortestPaths[getPairID(n, m)] =
            PathRef(pathsData.end() - path.size(), pathsData.end());
      }
    }
  }*/
  void ComputeComponents(){
    mlir::SmallVector<unsigned> Visited;
    numComponents=0;
    std::size_t numNodes = topology.getNumNodes();
    ComponentMap.resize(numNodes);
    for (unsigned n = 0; n < numNodes; ++n) {
      if (!llvm::is_contained(Visited, n)){
        Visited.push_back(n);
        numComponents++;
        ComponentMap[n]=numComponents;
        for (auto m = n + 1; m < numNodes; ++m) {
          if (getWeightedDistance(Qubit(n),Qubit(m))<INT_MAX){
            Visited.push_back(m);
            ComponentMap[m]=numComponents;
          }
        }

      }
    }
  }
  void computeAllPairShortestPaths() {
    std::size_t numNodes = topology.getNumNodes();
    shortestPaths.resize(numNodes * (numNodes + 1) / 2);
    shortestPathsWeights.resize(numNodes * (numNodes + 1) / 2);
    mlir::SmallVector<Qubit> path(numNodes);
    int weights=0;
    for (unsigned n = 0; n < numNodes; ++n) {
      auto parents = dijkstra(topology, Qubit(n));
      // Reconstruct the paths
      for (auto m = n + 1; m < numNodes; ++m) {
        weights=0;
        path.clear();
        path.push_back(Qubit(m));
        auto p = parents[m];
        if (p==Qubit(n)){
          int check =0;
          for (auto neighbour : topology.getNeighbours(Qubit(n))){
            if (neighbour==Qubit(m)){
              check=1;
            }
          }
          if (check==0){
            weights+=INT_MAX;
          }
          
        }
        while (p != Qubit(n)) {
          path.push_back(p);
          weights+=1;
          p = parents[p.index];
        }
        path.push_back(Qubit(n));
        pathsData.append(path.rbegin(), path.rend());
        shortestPathsWeights[getPairID(n, m)] = weights;
        shortestPaths[getPairID(n, m)] =
            PathRef(pathsData.end() - path.size(), pathsData.end());
      }
    }
  }

  /// Device nodes (qubits) and edges (connections)
  GraphCSR topology;

  /// List of shortest path from/to every source/destination
  mlir::SmallVector<PathRef> shortestPaths;
  mlir::SmallVector<int> shortestPathsWeights;

  /// Storage for `PathRef`'s in `shortestPaths`
  mlir::SmallVector<Qubit> pathsData;
  mlir::SmallVector<int> ComponentMap;
};

} // namespace cudaq
