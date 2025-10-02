#ifndef TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"

namespace mlir {
namespace triton {
namespace cpu {

// kernel func calling convention is (kernel_args..., thread_id, block_args...,
// shared_memory_ptr)
constexpr int kSharedMemoryOffset = -1;
constexpr int kProgramIdArgsOffset = -6 + kSharedMemoryOffset;
constexpr int kThreadIdOffset = -1 + kProgramIdArgsOffset;

// Returns a Value for the format string, which you can reuse. Writes the byte
// count for the string to |formatStrByteCount| if not null.
Value llPrintf(StringRef msg, ValueRange args, ArrayRef<bool> isSigned,
               RewriterBase &rewriter, const cpu::TargetInfo &targetInfo,
               int *formatStrByteCount = nullptr);

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal,
             std::optional<unsigned> alignment = std::nullopt);

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, std::optional<unsigned> alignment = std::nullopt);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H
