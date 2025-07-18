#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class GPUThreadIdOpToLLVM
    : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {

public:
  GPUThreadIdOpToLLVM(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp threadIdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement the conversion logic here
    llvm::errs() << "threadIdOp: " << threadIdOp << "\n";
    // TODO: do this properly
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        threadIdOp, threadIdOp.getType(), rewriter.getI32IntegerAttr(0));
    return success();
  }
};

} // namespace

void mlir::triton::NPU::populateGPUtoLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GPUThreadIdOpToLLVM>(typeConverter, benefit);
}
