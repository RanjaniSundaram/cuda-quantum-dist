#pragma once

#include "cudaq/Support/Handle.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

namespace cudaq{

class AbstractCircuit{
    using Offset = unsigned;

public: 
    struct Node : Handle {
        using Handle::Handle;
    };

    AbstractCircuit()=default;

    /// Creates a new node in the graph and returns its unique identifier.
  Node createNode() {
    Node node(getNumNodes());
    vertices.push_back(node);
    return node;
  }

  Node retrieveNode(int index) const{
    return vertices[index];
  }

  void addGate(Node src, Node dst){
    assert(src.isValid() && "Invalid source node");
    assert(dst.isValid() && "Invalid destination node");
    gates.push_back(std::make_tuple(src,dst));

  }


  std::size_t getNumNodes() const { return vertices.size(); }

  std::size_t getNumGates() const { return gates.size(); }

  

  // Stores the destination vertices of each edge.
  mlir::SmallVector<std::tuple<Node,Node>> gates;
  mlir::SmallVector<Node> vertices;
  mlir::SmallVector<int> weights;

};

}