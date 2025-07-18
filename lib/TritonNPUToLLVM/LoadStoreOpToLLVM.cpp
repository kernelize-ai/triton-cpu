#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const NPU::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const { return 1; }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const NPU::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const NPU::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    llvm::errs() << "rewrite load op: " << op << "\n";
    return failure();
  }
};

} // namespace

void mlir::triton::NPU::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion>(typeConverter, targetInfo, axisInfoAnalysis,
                                 benefit);
}
