#ifndef TRITON_NPU_CONVERSION_TTCPU_GENERIC_TO_D2M_H
#define TRITON_NPU_CONVERSION_TTCPU_GENERIC_TO_D2M_H

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace npu {

struct GenericPlan {

  GenericPlan() = default;

  struct Operand {
    Operation *boundaryOp; // the tt.load / tt.store consumer
    BlockArgument funcArg;
    SmallVector<int64_t> logicalShape;
    SmallVector<int64_t> tensorTiles;
    AffineMap indexingMap;
    Type elementType;
  };

  // --- operands, ins-then-outs; order must match indexing_maps ---
  SmallVector<Operand, 4> operands;
  unsigned numInputs = 0;

  // --- iteration space ---
  SmallVector<int64_t> gridShape;
  SmallVector<int64_t> blockFactors;
  SmallVector<Attribute> iteratorTypes;

  // --- body ---
  SmallVector<Operation *> dataOps; // topological order

  // --- masks ---
  // ignored for now

  ArrayRef<Operand> getInputs() const {
    return ArrayRef<Operand>(operands).take_front(numInputs);
  }
  ArrayRef<Operand> getOutputs() const {
    return ArrayRef<Operand>(operands).drop_front(numInputs);
  }

  LogicalResult setIterationSpace(ArrayRef<int64_t> workerGrid,
                                  Operation *diagnosticAnchorOp);

  static mlir::FailureOr<GenericPlan> build(cpu::GenericOp);
};

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
