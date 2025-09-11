#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

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
    FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");
    auto inputTypes = tritonTy.getInputs();
    llvm::errs() << "tritonTy = " << tritonTy << "\n";

    // move function parameters into attributes and change usage of the
    // arguments to custom ops

    // ttKernel lowering expects a function with no arguments and no return in
    // the signature. Arguments will retrieved with get_compile_time_arg_val op.
    FunctionType newTy =
        FunctionType::get(ctx, /*inputs=*/TypeRange{}, /*results=*/TypeRange{});

    // TODO: should we copy the name or use `ttkernel_compute`?
    auto newFunc = rewriter.create<func::FuncOp>(loc, funcOp.getName(), newTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFunc->setAttr(SymbolTable::getVisibilityAttrName(), vis);

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
      llvm::errs() << "need to convert arg : " << argTy << "\n";
      // TODO: try and leverage upstream tenstorrent type converter

      //   auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(i));
      //   auto val = rewriter.create<mydialect::KernelArgOp>(loc, argTy,
      //   idxAttr) /*.getResult()*/; newArgVals[i] = val.getResult(0);
    }

    return failure();
  }
};

} // namespace

void mlir::triton::npu::tt::populateFuncOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonFuncOpConversion>(typeConverter, context, benefit);
}
