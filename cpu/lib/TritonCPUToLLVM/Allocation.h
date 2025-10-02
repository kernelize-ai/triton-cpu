#ifndef TRITON_CONVERSION_TRITONCPU_TO_LLVM_ALLOCATION_H
#define TRITON_CONVERSION_TRITONCPU_TO_LLVM_ALLOCATION_H

#include "mlir/IR/Operation.h"

#include <functional>

namespace mlir {
namespace triton {

namespace cpu {
class TargetInfo;

std::function<unsigned(Operation *)>
getCPUAllocationAnalysisScratchSize(TargetInfo &targetInfo);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
