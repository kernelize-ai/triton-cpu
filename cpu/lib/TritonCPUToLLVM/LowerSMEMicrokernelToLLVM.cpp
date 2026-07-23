#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_LOWERSMEMICROKERNELTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

namespace {

struct LowerSMEMicrokernelToLLVMPass
    : public impl::LowerSMEMicrokernelToLLVMBase<
          LowerSMEMicrokernelToLLVMPass> {
  using LowerSMEMicrokernelToLLVMBase::LowerSMEMicrokernelToLLVMBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *context = &getContext();
    // use the base LLVM Type Converter since there should be no triton-specific
    // ops (other than return) in this function
    LLVMTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    FrozenRewritePatternSet frozen(std::move(patterns));

    ConversionTarget target(*context);
    target.addIllegalDialect<arith::ArithDialect, cf::ControlFlowDialect>();
    target.addLegalDialect<LLVM::LLVMDialect, arm_sme::ArmSMEDialect>();
    target.addLegalOp<UnrealizedConversionCastOp, triton::FuncOp,
                      triton::ReturnOp>();

    mod.walk([&](triton::FuncOp func) {
      if (!func->hasAttr("arm_locally_streaming")) // only the SME leaf
        return;
      if (failed(applyPartialConversion(func, target, frozen)))
        signalPassFailure();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerSMEMicrokernelToLLVMPass() {
  return std::make_unique<LowerSMEMicrokernelToLLVMPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
