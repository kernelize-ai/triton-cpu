#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class ThreadIdOpToLLVM : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {

public:
  ThreadIdOpToLLVM(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp threadIdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement the conversion logic here
    llvm::errs() << "threadIdOp: " << threadIdOp << "\n";
    // TODO: do this properly
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        threadIdOp, i32_ty, rewriter.getI32IntegerAttr(0));
    return success();
  }
};

class BlockIdOpToLLVM
    : public ConvertOpToLLVMPattern<mlir::triton::cpu::BlockIdOp> {

public:
  BlockIdOpToLLVM(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::triton::cpu::BlockIdOp>(typeConverter,
                                                             benefit) {}
  LogicalResult
  matchAndRewrite(mlir::triton::cpu::BlockIdOp blockIdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = blockIdOp->getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected LLVM::FuncOp as a parent of GetProgramIdOp");
    auto args = funcOp.getArguments();

    auto programIdDim = blockIdOp.getAxisAsInt();
    llvm::errs() << "programIdDim: " << programIdDim << "\n";
    assert(programIdDim >= 0 && programIdDim < 3);

    auto funcArgIdx = args.size() - 6 + programIdDim;
    assert(funcArgIdx < args.size() && "invalid SPMD program argument index");
    assert(args[funcArgIdx].getType().isInteger(32) &&
           "SPMD program argument must be i32");

    rewriter.replaceOp(blockIdOp, args[funcArgIdx]);
    return success();
  }
};

} // namespace

void mlir::triton::NPU::populateGPUtoLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ThreadIdOpToLLVM>(typeConverter, benefit);
  patterns.add<BlockIdOpToLLVM>(typeConverter, benefit);
}
