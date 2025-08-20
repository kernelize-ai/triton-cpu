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
    assert(programIdDim >= 0 && programIdDim < 3);

    auto funcArgIdx = args.size() - 6 + programIdDim;
    assert(funcArgIdx < args.size() && "invalid SPMD program argument index");
    assert(args[funcArgIdx].getType().isInteger(32) &&
           "SPMD program argument must be i32");

    rewriter.replaceOp(blockIdOp, args[funcArgIdx]);
    return success();
  }
};

Value getNumPrograms(mlir::FunctionOpInterface funcOp, int axis) {
  auto args = funcOp.getArguments();
  assert(funcOp && args.size() >= 6);
  assert(axis >= 0 && axis < 3);

  // The last three of the args are gridX, gridY, gridZ (bounds) of grid.
  auto argIdx = args.size() - 3 + axis;
  assert(argIdx < args.size() && "out-of-bounds arg index");
  assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
  return args[argIdx];
}

class GetNumProgramsOpToLLVM
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {

public:
  GetNumProgramsOpToLLVM(LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::GetNumProgramsOp>(typeConverter,
                                                         benefit) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected LLVM::FuncOp as a parent of GetNumProgramsOp");
    rewriter.replaceOp(op, getNumPrograms(funcOp, op.getAxisAsInt()));
    return success();
  }
};

} // namespace

void mlir::triton::npu::populateGPUtoLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ThreadIdOpToLLVM>(typeConverter, benefit);
  patterns.add<BlockIdOpToLLVM>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpToLLVM>(typeConverter, benefit);
}
