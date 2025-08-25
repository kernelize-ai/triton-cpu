#ifndef TRITON_CONVERSION_TRITONNPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONNPU_TO_LLVM_UTILITY_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"

namespace mlir {
namespace triton {
namespace npu {

/// Create a predicated block, using \p cond as the condition and \p ops for the
/// values supplied by the conditional branch to the exit block. The \p
/// thenOpsFn function is used to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2(%ops)
///   ^br1:
///     %then_ops = `thenOpsFn()`
///     cf.br ^br2(%then_ops)
///   ^br2(%block_ops):
template <typename ThenOpsFn>
Block &createPredicatedBlock(RewriterBase &rewriter, Location loc, Value cond,
                             ArrayRef<Value> ops, ThenOpsFn &&thenOpsFn) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *thenBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *endBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

  rewriter.setInsertionPointToEnd(insertionBlock);
  rewriter.create<cf::CondBranchOp>(loc, cond, thenBlock, endBlock, ops);

  rewriter.setInsertionPointToStart(thenBlock);
  auto thenOps = thenOpsFn();
  assert(thenOps.size() == ops.size() && "Inconsistent size");
  assert(llvm::all_of(llvm::enumerate(ops, thenOps),
                      [](const auto &enumerator) {
                        auto [index, op, thenOp] = enumerator;
                        return op.getType() == thenOp.getType();
                      }) &&
         "type mismatch found");

  if (thenOps.empty())
    rewriter.create<cf::BranchOp>(loc, endBlock);
  else
    rewriter.create<cf::BranchOp>(loc, endBlock, thenOps);

  for (Value op : thenOps)
    endBlock->addArgument(op.getType(), op.getLoc());

  rewriter.setInsertionPointToStart(endBlock);
  return *endBlock;
}

/// Create a predicated block, using \p cond as the condition and \p thenOpsFn
/// to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2
///   ^br1:
///     `thenOpsFn()`
///     cf.br ^br2
///   ^br2:
template <typename ThenOpsFn>
Block &createPredicatedBlock(RewriterBase &rewriter, Location loc, Value cond,
                             ThenOpsFn &&thenOpsFn) {
  return createPredicatedBlock(rewriter, loc, cond, {}, thenOpsFn);
}

// Returns a Value for the format string, which you can reuse. Writes the byte
// count for the string to |formatStrByteCount| if not null.
Value llPrintf(StringRef msg, ValueRange args, ArrayRef<bool> isSigned,
               ConversionPatternRewriter &rewriter,
               const npu::TargetInfo &targetInfo, int *formatStrByteCount);

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, unsigned alignment);

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, unsigned alignment);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONNPU_TO_LLVM_UTILITY_H
