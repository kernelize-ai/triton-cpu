#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRITONCPUTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::cf::ControlFlowDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<mlir::triton::cpu::MaskedLoadOp,
               mlir::triton::cpu::MaskedStoreOp>();
  }
};

struct ConvertTritonCPUToLLVM
    : public triton::cpu::impl::ConvertTritonCPUToLLVMBase<
          ConvertTritonCPUToLLVM> {
  using ConvertTritonCPUToLLVMBase::ConvertTritonCPUToLLVMBase;

  ConvertTritonCPUToLLVM() : ConvertTritonCPUToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Set up the type converter and patterns
    mlir::triton::cpu::TargetInfo targetInfo;

    // Allocate shared memory (uses default allocation scratch size function)
    // TODO: consider overloading defaultAllocationAnalysisScratchSizeFn if the
    // barrier is present
    ModuleAllocation allocation(
        mod,
        mlir::triton::cpu::getCPUAllocationAnalysisScratchSize(targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);

    cpu::populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                         targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    cpu::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    cpu::populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);

    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::cpu::populateGPUtoLLVMConversionPatterns(
        typeConverter, targetInfo, patterns, benefit);

    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);

    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Lower CF ops separately to avoid breaking analysis.
    TritonLLVMFunctionConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    unsigned cpuAddrSpace = 0u;
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/64, cpuAddrSpace);
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonCPUToLLVMPass() {
  return std::make_unique<ConvertTritonCPUToLLVM>();
}
} // namespace cpu
} // namespace triton
} // namespace mlir
