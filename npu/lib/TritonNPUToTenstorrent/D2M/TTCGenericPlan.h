#ifndef TRITON_NPU_CONVERSION_TTCPU_GENERIC_TO_D2M_H
#define TRITON_NPU_CONVERSION_TTCPU_GENERIC_TO_D2M_H

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

class GenericPlan {};

mlir::FailureOr<GenericPlan> buildPlan(cpu::GenericOp);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
