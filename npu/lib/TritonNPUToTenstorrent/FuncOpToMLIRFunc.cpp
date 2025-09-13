#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct TritonFuncOpConversion : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!triton::isKernel(funcOp)) {
      return rewriter.notifyMatchFailure(
          funcOp, "non-kernel functions are not yet supported");
    }

    Location loc = funcOp.getLoc();
    MLIRContext *ctx = funcOp.getContext();

    // create a new funcop void -> void and copy over (or re-use) regions/blocks
    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");
    auto inputTypes = tritonTy.getInputs();
    llvm::errs() << "tritonTy = " << tritonTy << "\n";

    // move function parameters into attributes and change usage of the
    // arguments to custom ops

    // ttKernel lowering expects a function with no arguments and no return in
    // the signature. Arguments will retrieved with get_compile_time_arg_val op.
    mlir::FunctionType newTy = mlir::FunctionType::get(
        ctx, /*inputs=*/TypeRange{}, /*results=*/TypeRange{});

    // TODO: should we copy the name or use `ttkernel_compute`?
    auto newFunc =
        rewriter.create<mlir::func::FuncOp>(loc, funcOp.getName(), newTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFunc->setAttr(SymbolTable::getVisibilityAttrName(), vis);

    // TODO: set tenstorrent specific function attributes

    // Skip copying function attributes from the old function for now.

    rewriter.inlineRegionBefore(funcOp.getBody(), newFunc.getBody(),
                                newFunc.end());
    // zero the block arguments and replace their uses with ttKernel ops
    Block &entry = newFunc.front();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&entry);

    SmallVector<Value> newArgVals(entry.getNumArguments());
    for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
      Type argTy = entry.getArgument(i).getType();

      auto val = rewriter.create<npu::tt::KernelArgOp>(
          loc, argTy, rewriter.getIndexAttr(i));

      // TODO: val or getResult?
      newArgVals[i] = val;
    }

    // Replace all uses of the block args with the newly created ops.
    for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i)
      entry.getArgument(i).replaceAllUsesWith(newArgVals[i]);

    // Now remove the block arguments (backwards to keep indices stable).
    for (int i = static_cast<int>(entry.getNumArguments()) - 1; i >= 0; --i)
      entry.eraseArgument(i);

    rewriter.eraseOp(funcOp); // old symbol is gone; newFunc replaces it
    return success();
  }
};

} // namespace

void mlir::triton::npu::tt::populateFuncOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonFuncOpConversion>(typeConverter, context, benefit);
}
