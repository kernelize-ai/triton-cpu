#ifndef TRITON_CONVERSION_TRITONNPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONNPU_TO_LLVM_UTILITY_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"

namespace mlir {
namespace triton {
namespace npu {

// kernel func calling convention is (kernel_args..., thread_id, block_args...,
// shared_memory_ptr)
constexpr int kSharedMemoryOffset = -1;
constexpr int kProgramIdArgsOffset = -6 + kSharedMemoryOffset;
constexpr int kThreadIdOffset = -1 + kProgramIdArgsOffset;

// Returns a Value for the format string, which you can reuse. Writes the byte
// count for the string to |formatStrByteCount| if not null.
Value llPrintf(StringRef msg, ValueRange args, ArrayRef<bool> isSigned,
               RewriterBase &rewriter, const npu::TargetInfo &targetInfo,
               int *formatStrByteCount = nullptr);

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, unsigned alignment);

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, unsigned alignment);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONNPU_TO_LLVM_UTILITY_H
