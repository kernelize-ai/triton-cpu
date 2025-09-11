#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/TritonNPUToTenstorrent/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "../TritonNPUToLLVM/TargetInfo.h"
#include "PatternTritonNPUOpToTenstorrent.h"
#include "TypeConverter.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"

namespace mlir {
namespace triton {
namespace npu {
#define GEN_PASS_DEF_CONVERTTRITONNPUTOTENSTORRENT
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"
} // namespace npu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class TritonTenstorrentFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonTenstorrentFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<func::FuncDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct TritonTenstorrentConversionTarget : public ConversionTarget {
public:
  explicit TritonTenstorrentConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addIllegalDialect<triton::TritonDialect>();
  }
};

struct ConvertTritonNPUToTenstorrent
    : public triton::npu::impl::ConvertTritonNPUToTenstorrentBase<
          ConvertTritonNPUToTenstorrent> {
  using ConvertTritonNPUToTenstorrentBase::ConvertTritonNPUToTenstorrentBase;

  ConvertTritonNPUToTenstorrent() : ConvertTritonNPUToTenstorrentBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Set up the type converter and patterns
    mlir::triton::npu::TargetInfo
        targetInfo; // TODO: tenstorrent specific target info
    TritonNPUToTenstorrentTypeConverter typeConverter(context);

    // Lower functions
    TritonTenstorrentFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);

    // TODO: clean up namespacing
    mlir::triton::npu::tt::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo,
        npu::tt::patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    RewritePatternSet patterns(context);

    // TODO: rest :)

    TritonTenstorrentConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace npu {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonNPUToTenstorrentPass() {
  return std::make_unique<ConvertTritonNPUToTenstorrent>();
}

} // namespace npu
} // namespace triton
} // namespace mlir
