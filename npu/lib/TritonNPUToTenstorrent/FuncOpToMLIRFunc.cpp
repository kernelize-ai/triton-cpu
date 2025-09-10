#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct TritonFuncOpConversion : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
                    return failure();
                  }


};

}

void mlir::triton::npu::tt::populateFuncOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonFuncOpConversion>(typeConverter, context, benefit);
}
