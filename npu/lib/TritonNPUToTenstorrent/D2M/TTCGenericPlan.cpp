#include "TTCGenericPlan.h"

namespace mlir {
namespace triton {
namespace npu {

namespace {}

mlir::FailureOr<GenericPlan> buildPlan(cpu::GenericOp generic) {
  // check generic op metadata criteria

  // TODO: support generics with reductions
  if (generic.getReductionDims().size() != 0) {
    generic.emitError("reduction dims not yet supported in D2M lowering");
    return failure();
  }
  if (generic.getInitVals().size() != 0) {
    generic.emitError("init vals not yet supported in D2M lowering");
    return failure();
  }

  for (auto ins : generic.getIns()) {
    if (isa<RankedTensorType>(ins.getType())) {
      generic.emitError(
          "tensor type inputs to generic op not yet supported in D2M lowering");
      return failure();
    }
  }

  // TODO: we can likely relax this restriction, but for now it's not
  // unreasonable
  for (auto [bVal, t] :
       llvm::zip(generic.getBlockShape(), generic.getTileShape())) {
    // block shape is a value, so check for a constant
    APInt val;
    if (!matchPattern(bVal, m_ConstantInt(&val))) {
      generic.emitError("expected block shape to be constants");
      return failure();
    }
    int64_t b = val.getSExtValue();
    if (b != t) {
      generic.emitError("expected generic block shape and tile shape to be "
                        "equal for D2M lowering");
      return failure();
    }
  }

  GenericPlan plan;

  // TODO: rest

  return plan;
}

} // namespace npu
} // namespace triton
} // namespace mlir
