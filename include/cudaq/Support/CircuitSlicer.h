/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/ADT/GraphCSR.h"
#include "cudaq/ADT/AbstractCircuit.h"
#include "cudaq/Support/Graph.h"
#include "llvm/Support/MemoryBuffer.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <map>
#include <cstdlib>

namespace cudaq {

class QAPSolver {
	public:
		std::vector<std::vector<int>> fMat; //flow matrix, n by n
		std::vector<std::vector<int>> dMat; // distance matrix, n by n
		unsigned int n; // the number of facilities/locations
		QAPSolver(std::vector<std::vector<int>> flow, std::vector<std::vector<int>> distance, unsigned int size) {
			fMat = flow;
			dMat = distance;
			n = size;
		}

		~QAPSolver() {
		}

		void genInitSol( std::vector<int>* arr, unsigned int iter = 10) {
			randomGen(&*arr, iter);
		}

		// Tabu Search algorithm. Takes in the initial solution, maximum iteration, min tabu tenure and max tabu tenure
		void TS(std::vector<int> current, int maxIter = 10, int tmin = 7, int tmax = 10) {
			// generate the sequence of tenure values
			std::vector<int> tenureArr;
			generateTenureList(&tenureArr, tmin, tmax);
			unsigned int tenureValInd = 0; // to determine the tenure value assigned to each new tabu-active moves
			// initialization step
			bestSol = current;
			std::vector<std::vector<int>> candidateList; // list of neighborhood solution that is tabu-inactive
			std::vector<int> moveValue; // objective function value of the moves
			std::vector<std::vector<unsigned int>> moveList; // stores the 2-opt swap move
			int iter = 1;
			std::vector<std::vector<unsigned int>> tabuList; // stores the best move as a tabu-active move
			std::vector<int> tabuTenureList;

			while (iter <= maxIter) {
				tenureCheck(&tabuList, &tabuTenureList, &iter); // check if the tenure for all tabu-active moves have expired, remove them if true
				for (unsigned int i = 0; i < n; i++) {
					for (unsigned int j = i + 1; j < n; j++) {
						std::vector<unsigned int> mv{ i, j };
						// only consider a move if it is not tabu active, else check if aspiration criteria is met
						if (!tabuCheck(tabuList, mv)) {
							transpose2(current, i, j);
							candidateList.push_back(neighborSol);
							moveValue.push_back(objectiveFunction(neighborSol));
							moveList.push_back(mv);
						}
						else {
							transpose2(current, i, j);
							int neighborVal = objectiveFunction(neighborSol);
							if (neighborVal < objectiveFunction(bestSol)) {
								// revoke the tabu status of a move that could give better solution than current best solution
								tabuRevoke(&tabuList, &tabuTenureList, mv);
								candidateList.push_back(neighborSol);
								moveValue.push_back(neighborVal);
								moveList.push_back(mv);
							}
						}
					}
				}
				// Select the best candidate, it is the moves that yield the lowest objective function value
				std::vector<int>::iterator bestNeighbor = std::min_element(moveValue.begin(), moveValue.end());
				int bestNeighborPos = int(std::distance(moveValue.begin(), bestNeighbor));
				if(candidateList.size()==0){
					break;
				}
				// Compare the best candidate to the current best solution
				if (moveValue[bestNeighborPos] < objectiveFunction(bestSol)) {
					bestSol = candidateList[bestNeighborPos];
				}

				// best candidate is made as the current solution for the next iteration
				current = candidateList[bestNeighborPos];
				// make the move that yield the best candidate a tabu-active
				tabuAdd(&tabuList, &tabuTenureList, moveList[bestNeighborPos], &tenureArr, tenureValInd, iter);
				// clear the current iterations candidate list, move list and move values
				candidateList.clear();
				moveValue.clear();
				moveList.clear();
				iter++; // increase the iterations
			}
		}

		std::vector<int> getPerm() {
			return bestSol;
		}

		std::vector<int> getPermInitSol() {
			return initSol;
		}

		int getSolution() {
			return objectiveFunction(bestSol);
		}

		int getSolutionInitSol() {
			return objectiveFunction(initSol);
		}

	private:
		std::vector<int> neighborSol; // the temporary solution from a neighborhood operation
		std::vector<int> bestSol; // the solution that gives the best value to the optimization problem
		std::vector<int> initSol; // save the initial solution generated

		// generate the initial solution randomly, set iter to non-zero to search iteratively for a better initial solution
		// *arr is the pointer to the array the initial solution is write into. iter is set to 10 by default.
		void randomGen(std::vector<int>* arr, unsigned int iter = 10) {
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
      //llvm::raw_ostream &os = llvm::errs();
      //os<<"Ranjani checking objective 1"<<"\n";
			int total = 0;
			for (unsigned int i = 0; i < n; i++) {
        //os<<"Ranjani checking obj indices "<<solution[i]<<"\n";
				for (unsigned int j = 0; j < n; j++) {
					if (i != j) {
            //os<<"Ranjani checking obj indices "<<i<<" "<<j<<" "<<solution[j]<<"\n";
            //os<<"Ranjani checking obj "<<fMat[i][j]<<" "<<dMat[static_cast<unsigned int>(solution[i]) - 1][static_cast<unsigned int>(solution[j]) - 1]<<"\n";
						total += fMat[i][j] * dMat[static_cast<unsigned int>(solution[i]) - 1][static_cast<unsigned int>(solution[j]) - 1];
					} 
				}
			}
			return total;
		}

		// transposition of 2 elements
		void transpose2(std::vector<int> solution, unsigned int i, unsigned int j) {
      //llvm::raw_ostream &os = llvm::errs();
			neighborSol = solution;
      //os<<"Ranjani checking transposition 1"<<neighborSol[i]<<"\n";
			unsigned int temp = neighborSol[i];
      //os<<"Ranjani checking transposition 2"<<"\n";
      //os<<"Ranjani checking transposition 1"<<neighborSol[i]<<"\n";
			neighborSol[i] = neighborSol[j];
      //os<<"Ranjani checking transposition 3"<<"\n";
			neighborSol[j] = temp;
		}


		// generate the initial tenure list in ascending order between tmin and tmax (inclusive)
		void generateTenureList(std::vector<int>* tenure_arr, int tmin, int tmax) {
			for (int i = tmin; i <= tmax; i++) {
				tenure_arr->push_back(i);
			}
		}

		// check if the move is tabu-active. true if it is tabu active, false otherwise
		bool tabuCheck(std::vector<std::vector<unsigned int>> tList, std::vector<unsigned int> move) {
			for (unsigned int i = 0; i < tList.size(); i++) {
				if (tList[i] == move) {
					return true;
				}
			}
			return false;
		}

		// revoke a tabu status based on a move
		void tabuRevoke(std::vector<std::vector<unsigned int>>* tList, std::vector<int>* tenList, std::vector<unsigned int> move) {
			std::vector<std::vector<unsigned int>> t = *tList;
			std::vector<int> ten = *tenList;
			std::vector<unsigned int> moveMirror{ move[1], move[0] };
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

		// add a new tabu-active move to the tabu list, assigned a different tenure at each iteration using the systematic dynamic tenure principle
		void tabuAdd(std::vector<std::vector<unsigned int>>* tList, std::vector<int>* tenList, std::vector<unsigned int> move, std::vector<int>* tenure_arr, unsigned int tenureIndex, int iter) {
			std::vector<unsigned int> moveMirror{ move[1], move[0] }; // create the symmetrical opposite of the swap move
			// make a move tabu by appending it and its symmetric opposite to the tabu list
			tList->push_back(move);
			tList->push_back(moveMirror);
			// add the tabu tenure to the new tabu-active move
			// if the end of the sequence for the range of tabu tenure is reached, reshuffle the sequence randomly and start the index at 0 again.
			if (tenureIndex == tenure_arr->size()) {
				std::random_shuffle(tenure_arr->begin(), tenure_arr->end());
				tenureIndex = 0;  // restart the sequence
			}
			std::vector<int> ten_arr = *tenure_arr;
			tenList->push_back(iter + ten_arr[tenureIndex]);
			tenList->push_back(iter + ten_arr[tenureIndex]); // the symmetrical opposite move also share the same tenure
			tenureIndex = tenureIndex + 1;
		}

		// check the tenure of each tabu-active moves, revoke them if the tenure is expired
		void tenureCheck(std::vector<std::vector<unsigned int>>* tList, std::vector<int>* tenList, int* iter) {
			std::vector<int> ten = *tenList;
			std::vector<unsigned int> pos;
			for (unsigned int i = 0; i < ten.size(); i++) {
				if (ten[i] < *iter) {
					pos.push_back(i); // record all expired tabu-active tenure's index in order
				}
			}
			// revoke tabu status for all tabu-active moves with expired tenure
			for (unsigned int j = 0; j < pos.size(); j++) {
				tList->erase(tList->begin() + (pos[j] - j));
				tenList->erase(tenList->begin() + (pos[j] - j));
			}
		}

};

class CircuitSlicer {
public:
  using Qubit = AbstractCircuit::Node;
  mlir::SmallVector<int> remappingIndices;
  std::map<int, std::vector<int>> reMaps;

  CircuitSlicer(int numQs, mlir::SmallVector<std::tuple<int,int>> gates ){
    numQubits=numQs;

    for (int i=0; i<numQubits; i++){
      Circuit.createNode();
    }
	//llvm::raw_ostream &os = llvm::errs();
    for (auto& [first,second] : gates){
		//os << "gates " <<first<< " "<<second<<" ";
      	Circuit.addGate(Qubit(first),Qubit(second));
    }
	//os<<"\n";


  }


  void DynamicProgram(std::vector<std::vector<int>> distMat){
    int numGates= Circuit.getNumGates();
	llvm::raw_ostream &os = llvm::errs();
	os << "Dynamic Programming " <<"\n";
    std::vector<std::vector<int>> SolutionCost(numGates, std::vector<int>(numGates, 0));
    //std::vector<std::vector<std::vector< std::tuple<int,int> >>> SolutionIndices;
	std::vector<std::vector<std::vector<std::tuple<int,int>>>> SolutionIndices(numGates, std::vector<std::vector<std::tuple<int,int>>>(numGates, std::vector<std::tuple<int,int>>(numGates)));
    std::vector<std::vector<std::vector<int>>> SubcircuitSolutions(numGates, std::vector<std::vector<int>>(numGates, std::vector<int>(numGates, 0)));
    std::vector<std::vector<int>> SubcircuitCost(numGates, std::vector<int>(numGates, 0));
    int MinVal=INT_MAX;
    int MinIndex=0;
    int CurrVal;
	//os << "Dynamic Programming: initialized " <<"\n";
	for (int i=0; i<numGates-1; i++){
      for (int j=i+1; j<numGates; j++){
        QAPSolver Instance=QAPSolver(FlowMatrix(i,j), distMat, numQubits);
		std::vector<int> initial;
		Instance.genInitSol( &initial, 50);
		Instance.TS(initial, 100, 20, 30);
		SubcircuitCost[i][j]=Instance.getSolution()/2;
		SubcircuitSolutions[i][j]=Instance.getPerm();
      }
    }
	//os << "Dynamic Programming: subcircuit values set " <<"\n";
    for (int i=0; i<numGates-1; i++){
      for (int j=i+1; j<numGates; j++){
		//os << "Dynamic Programming: new indices "<<i<<" "<<j <<"\n";
        MinVal=INT_MAX;
        MinIndex=0;
		SolutionIndices[i][j].clear();
        if (i==0){
          SolutionCost[i][j]=SubcircuitCost[i][j];
		  //os << "Dynamic Programming: min val "<<SubcircuitCost[i][j]<<"\n";
          SolutionIndices[i][j].push_back(std::make_tuple(i,j));
          continue;
        }
		
        for (int k =0; k<i; k++){
          CurrVal=SolutionCost[k][i] +SubcircuitCost[i][j]+
           StitchingCost(SubcircuitSolutions[k][i],SubcircuitSolutions[i][j], distMat);
          if(MinVal>= CurrVal){
            MinVal=CurrVal;
            MinIndex=k;
          }
        }
		//os << "Dynamic Programming: min val and index"<<MinVal<<" "<<MinIndex <<"\n";
        SolutionCost[i][j]=MinVal;
        //SolutionIndices[i][j].insert(SolutionIndices[i][j].end(),SolutionIndices[MinIndex][i].begin(),SolutionIndices[MinIndex][i].end());
		for (auto& [beg,end]: SolutionIndices[MinIndex][i] ){
			SolutionIndices[i][j].push_back(std::make_tuple(beg,end));
			//os << "Dynamic Programming: after "<<beg<<" "<<end <<"\n";
		}
		SolutionIndices[i][j].push_back(std::make_tuple(i,j));
      }
    }
	MinVal=INT_MAX;
    MinIndex=0;
	for (int i=0; i<numGates-1; i++){
		if (MinVal> SolutionCost[i][numGates-1]){
			MinVal=SolutionCost[i][numGates-1];
			MinIndex=i;
		}
	}
	//os << "Dynamic Programming: end value "<<MinVal <<"\n";
	for (auto& [beg,end]: SolutionIndices[MinIndex][numGates-1]){
		//os << "Dynamic Programming: end "<<beg<<" "<<end <<"\n";
		remappingIndices.push_back(beg);
		reMaps[beg]=SubcircuitSolutions[beg][end];
	}
  }



private:
  int numQubits;

  AbstractCircuit Circuit;

  int StitchingCost(std::vector<int> soln1, std::vector<int> soln2, std::vector<std::vector<int>> distMat){
    int cost=0;
	//llvm::raw_ostream &os = llvm::errs();
	//os << "stitching cost  " <<"\n";
	for (int i=0; i<numQubits; i++){
		//os << "stitching cost  " <<i<<" "<< soln1[i]<<" "<<soln2[i]<<"\n";
		cost+=distMat[soln1[i]-1][soln2[i]-1];

	}
	return cost;
  }
  // A function that returns a flow matrix corresponding to the sub-circuit contained
// between the gate indices i and j
  std::vector<std::vector<int>> FlowMatrix(int i, int j){
	//llvm::raw_ostream &os = llvm::errs();
	//os << "Flow matrix for subcircuit " <<"\n";
    mlir::SmallVector<std::tuple<Qubit,Qubit>> SubCircuit(Circuit.gates.begin() + i, Circuit.gates.begin() + j);
    //os << "Flow matrix  "<<numQubits <<"\n";
	std::vector<std::vector<int>> flow(numQubits, std::vector<int>(numQubits, 0));
    for (const auto& [first, second] : SubCircuit) {
		//os << "Flow matrix  "<<first<<" "<<second <<"\n";
		flow[first.index][second.index]++;
		flow[second.index][first.index]++;
		//os << "Flow matrix  "<<first<<" "<<second <<" "<<flow[first.index][second.index]<<"\n";
    }
    return flow;
  }
  
};

} // namespace cudaq
