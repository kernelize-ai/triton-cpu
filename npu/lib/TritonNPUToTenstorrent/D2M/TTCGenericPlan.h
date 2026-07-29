#ifndef TRITON_NPU_CONVERSION_TTCPU_GENERIC_TO_D2M_H
#define TRITON_NPU_CONVERSION_TTCPU_GENERIC_TO_D2M_H

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace npu {

class GenericPlan {
public:
};

mlir::FailureOr<GenericPlan> buildPlan(cpu::GenericOp);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
