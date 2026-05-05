#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct LocalAllocOpConversion : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
      LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

       LogicalResult matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        if (!op.isSharedMemoryAlloc())
      return failure();

       Location loc = op->getLoc();
       assert(false && "TODO");
       return success();
       } 

private:
  const TargetInfoBase &targetInfo;
};

}

void mlir::triton::cpu::populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
          patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
    }
