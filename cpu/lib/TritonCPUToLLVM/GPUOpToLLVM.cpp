#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// launch_id = [blockId, threadId, 0, 0]
namespace LaunchIDOffsets {
constexpr int kBlockId = 0;
constexpr int kThreadId = 1;

} // namespace LaunchIDOffsets

class ThreadIdOpToLLVM : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {

public:
  ThreadIdOpToLLVM(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp threadIdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = threadIdOp->getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected LLVM::FuncOp as a parent of ThreadIdOp");
    auto args = funcOp.getArguments();

    auto threadIdDim = threadIdOp.getDimension();
    if (threadIdDim != mlir::gpu::Dimension::x) {
      threadIdOp.emitError("unsupported thread id dimension");
    }

    auto funcArgIdx = args.size() + cpu::kLaunchIdOffset;
    assert(funcArgIdx >= 0 && "Launch id argument must be a pointer");
    auto b = TritonLLVMOpBuilder(threadIdOp.getLoc(), rewriter);
    auto idxTy = typeConverter->convertType(threadIdOp.getType());
    auto axisVal =
        b.i32_val(static_cast<int>(threadIdDim) + LaunchIDOffsets::kThreadId);
    auto gep =
        b.gep(ptr_ty(rewriter.getContext()), idxTy, args[funcArgIdx], axisVal);
    auto threadId = b.load(idxTy, gep);
    rewriter.replaceOp(threadIdOp, threadId);

    return success();
  }
};

Value getNumPrograms(Location loc, ConversionPatternRewriter &rewriter,
                     mlir::FunctionOpInterface funcOp, int axis) {
  assert(funcOp);
  assert(axis >= 0 && axis < 3);

  auto args = funcOp.getArguments();
  auto funcArgIdx = args.size() + cpu::kLaunchSzOffset;
  assert(funcArgIdx >= 0 && "Launch sz argument out of range");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto axisVal = b.i32_val(axis);
  auto gep =
      b.gep(ptr_ty(rewriter.getContext()), i32_ty, args[funcArgIdx], axisVal);
  return b.load(i32_ty, gep);
}

// x = blockIdx % gridX
Value convertBlockIndexToDimX(ConversionPatternRewriter &rewriter,
                              Value blockIdx, FunctionOpInterface funcOp) {
  auto loc = blockIdx.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return LLVM::SRemOp::create(rewriter, loc, blockIdx,
                              getNumPrograms(loc, rewriter, funcOp, 0));
}

// y = (idx % (gridX * gridY)) / gridX
Value convertBlockIndexToDimY(ConversionPatternRewriter &rewriter,
                              Value blockIdx, FunctionOpInterface funcOp) {
  auto loc = blockIdx.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value gridX = getNumPrograms(loc, rewriter, funcOp, 0);
  Value gridXY = b.mul(gridX, getNumPrograms(loc, rewriter, funcOp, 1));
  Value idxModXY = LLVM::SRemOp::create(rewriter, loc, blockIdx, gridXY);
  return b.sdiv(idxModXY, gridX);
}

// z = idx / (gridX * gridY)
Value convertBlockIndexToDimZ(ConversionPatternRewriter &rewriter,
                              Value blockIdx, FunctionOpInterface funcOp) {
  auto loc = blockIdx.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value gridXY = b.mul(getNumPrograms(loc, rewriter, funcOp, 0),
                       getNumPrograms(loc, rewriter, funcOp, 1));
  return b.sdiv(blockIdx, gridXY);
}

class BlockIdOpToLLVM
    : public ConvertOpToLLVMPattern<mlir::triton::cpu::BlockIdOp> {

public:
  BlockIdOpToLLVM(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::triton::cpu::BlockIdOp>(typeConverter,
                                                             benefit) {}
  LogicalResult
  matchAndRewrite(mlir::triton::cpu::BlockIdOp blockIdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto b = TritonLLVMOpBuilder(blockIdOp.getLoc(), rewriter);
    auto funcOp = blockIdOp->getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected LLVM::FuncOp as a parent of GetProgramIdOp");
    auto args = funcOp.getArguments();

    auto funcArgIdx = args.size() + cpu::kLaunchIdOffset;
    assert(funcArgIdx >= 0 && "Launch id argument must be a pointer");
    auto idxTy = typeConverter->convertType(blockIdOp.getType());
    // the linear grid id is the first element in the launch params pointer
    auto gep = b.gep(ptr_ty(rewriter.getContext()), idxTy, args[funcArgIdx],
                     b.i32_val(LaunchIDOffsets::kBlockId));
    auto blockId = b.load(idxTy, gep);

    auto programIdDim = blockIdOp.getAxis();
    switch (programIdDim) {
    case ProgramIDDim::X: {
      rewriter.replaceOp(blockIdOp,
                         convertBlockIndexToDimX(rewriter, blockId, funcOp));
      break;
    }
    case ProgramIDDim::Y: {
      rewriter.replaceOp(blockIdOp,
                         convertBlockIndexToDimY(rewriter, blockId, funcOp));
      break;
    }
    case ProgramIDDim::Z: {
      rewriter.replaceOp(blockIdOp,
                         convertBlockIndexToDimZ(rewriter, blockId, funcOp));
      break;
    }
    default:
      assert(false && "invalid program id dimension");
    }
    return success();
  }
};

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

    auto numPrograms =
        getNumPrograms(op.getLoc(), rewriter, funcOp, op.getAxisAsInt());
    rewriter.replaceOp(op, numPrograms);
    return success();
  }
};

class GpuBarrierOpToLLVM : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
public:
  GpuBarrierOpToLLVM(LLVMTypeConverter &typeConverter,
                     const cpu::TargetInfo &targetInfo, PatternBenefit benefit)
      : targetInfo(targetInfo),
        ConvertOpToLLVMPattern<mlir::gpu::BarrierOp>(typeConverter, benefit) {}

  // Implements a simple, reusable software barrier for a fixed-size set of
  // workers. Assumes input ptrs are initialized to zero.
  LLVM::LLVMFuncOp
  getOrCreateCpuBarrier(Location loc,
                        ConversionPatternRewriter &rewriter) const {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    constexpr StringLiteral kName = "_cpu_barrier";
    if (auto f = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(kName))
      return f;

    auto *context = rewriter.getContext();

    // void barrier(int* count, int* phase, int32_t num_workers)
    auto funcTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                              {ptr_ty(context)},
                                              /*vararg=*/false);

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto func = LLVM::LLVMFuncOp::create(rewriter, moduleOp.getLoc(), kName,
                                         funcTy, LLVM::Linkage::External);
    return func;
  }

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: delete the barrier if num warps is 1
    auto barrierFunc = getOrCreateCpuBarrier(op.getLoc(), rewriter);

    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();

    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    auto args = funcOp.getArguments();
    auto funcArgIdx = args.size() + cpu::kCpuBarrierOffset;
    assert(funcArgIdx >= 0 && "invalid SPMD program argument index");

    SmallVector<Value> call_args{args[funcArgIdx]};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, barrierFunc, call_args);
    return success();
  }

protected:
  const cpu::TargetInfo &targetInfo;
};

class GpuLocalBarrierOpToLLVM
    : public ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp> {
public:
  GpuLocalBarrierOpToLLVM(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp>(typeConverter,
                                                            benefit) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::gpu::BarrierOp>(op);
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateGPUtoLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ThreadIdOpToLLVM>(typeConverter, benefit);
  patterns.add<BlockIdOpToLLVM>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpToLLVM>(typeConverter, benefit);
  patterns.add<GpuBarrierOpToLLVM>(typeConverter, targetInfo, benefit);
  patterns.add<GpuLocalBarrierOpToLLVM>(typeConverter, benefit);
}
