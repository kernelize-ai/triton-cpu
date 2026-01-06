#include "TargetInfo.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter,
                  const cpu::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    LDBG("Converting dot op: " << *op);
    LDBG("Converted op A: " << adaptor.getA());

    LDBG("Converted A input type: " << adaptor.getA().getType());

    SmallVector<Value> aElems =
        unpackLLElements(op.getLoc(), adaptor.getA(), rewriter);
    LDBG("Unpacked A elements size: " << aElems.size());
    LDBG("A tensor element 0 type: " << aElems[0].getType());

    // decompose the tensor matmul into vector dot products and accumulate the
    // results into the accumulator input ("C")

    SmallVector<Value> accumulator =
        unpackLLElements(op.getLoc(), adaptor.getC(), rewriter);
    LDBG("Unpacked C elements size: " << accumulator.size());
    // fill accumulator with loop carried values from getC()
    // loop over vectors in tensor A and tensor B
    // dot the a and b vectors
    // write into accumulator[X]

    // b.fma(scalar_a, scalar_b, accumulator[idx]);

    targetInfo.printf(rewriter, "DotOpConversion: unpacked A size: %d\n",
                      {LLVM::createConstantI32(loc, rewriter, aElems.size())});
    Value accumulatorStruct =
        packLLElements(loc, getTypeConverter(), accumulator, rewriter,
                       getTypeConverter()->convertType(op.getType()));
    rewriter.replaceOp(op, {accumulatorStruct});
    return success();
  }

private:
  const cpu::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const cpu::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, targetInfo, benefit);
}
