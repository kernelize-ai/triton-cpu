#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class ThreadIdOpToLLVM : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {

public:
  ThreadIdOpToLLVM(LLVMTypeConverter &typeConverter,
                   const npu::TargetInfo &targetInfo, PatternBenefit benefit)
      : targetInfo(targetInfo),
        ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>(typeConverter, benefit) {}

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

    assert(args.size() > 7 &&
           "incorrect npu kernel function signature"); // could be 6, but we
                                                       // always expect at least
                                                       // one argument
    auto funcArgIdx = args.size() - 7;
    assert(args[funcArgIdx].getType().isInteger(32) &&
           "Thread ID argument must be i32");
    // npu::llPrintf("threadid: %d", {args[funcArgIdx]}, {true}, rewriter,
    // targetInfo);
    rewriter.replaceOp(threadIdOp, args[funcArgIdx]);
    return success();
  }

protected:
  const npu::TargetInfo &targetInfo;
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

class GpuBarrierOpToLLVM : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
public:
  GpuBarrierOpToLLVM(LLVMTypeConverter &typeConverter,
                     const npu::TargetInfo &targetInfo, PatternBenefit benefit)
      : targetInfo(targetInfo),
        ConvertOpToLLVMPattern<mlir::gpu::BarrierOp>(typeConverter, benefit) {}

  // Implements a simple, reusable software barrier for a fixed-size set of
  // workers. Assumes input ptrs are initialized to zero.
  LLVM::LLVMFuncOp
  getOrCreateCpuBarrier(Location loc,
                        ConversionPatternRewriter &rewriter) const {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    constexpr StringLiteral kName = "barrier";
    if (auto f = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(kName))
      return f;

    auto *context = rewriter.getContext();

    // void barrier(int* count, int* phase, int32_t num_workers, int32_t
    // TMP_threadId)
    auto funcTy = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context),
        {ptr_ty(context), ptr_ty(context), i32_ty, i32_ty /*REMOVE*/},
        /*vararg=*/false);

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto func =
        rewriter.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), kName, funcTy);

    Block *entryBlock = func.addEntryBlock(rewriter);
    Block *lastBlock = new Block(), *waitBlock = new Block(),
          *afterSpinBlock = new Block(), *exitBlock = new Block();
    func.getBody().push_back(lastBlock);
    func.getBody().push_back(waitBlock);
    func.getBody().push_back(afterSpinBlock);
    func.getBody().push_back(exitBlock);

    // TODO: use TritonLLVMIRRewriter?
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    rewriter.setInsertionPointToEnd(entryBlock);

    // get the current phase
    Value phasePtr = func.getArgument(1);
    Value crtPhase = b.load(i32_ty, phasePtr);
    // llPrintf("crtPhase[%d] = %d\n", {func.getArgument(3), crtPhase}, {true,
    // true}, rewriter, targetInfo);

    // atomically increment the count
    Value countPtr = func.getArgument(0);
    // auto ordering = LLVM::AtomicOrderingAttr::get(context,
    // LLVM::AtomicOrdering::acq_rel);
    auto ordering = LLVM::AtomicOrdering::acq_rel;

    Value old = rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, countPtr, b.i32_val(1), ordering);
    // llPrintf("count[%d] = %d\n", {func.getArgument(3), old}, {true, true},
    // rewriter, targetInfo);

    // check to see if we are the last thread to hit the barrier
    Value numWorkers = func.getArgument(2);
    Value arrived = b.add(old, b.i32_val(1));
    Value amLast = b.icmp_eq(arrived, numWorkers);

    // rewriter.create<cf::BranchOp>(loc, exitBlock); // TODO!
    rewriter.create<cf::CondBranchOp>(loc, amLast, lastBlock, waitBlock);

    // last block
    {
      rewriter.setInsertionPointToEnd(lastBlock);
      // reset count
      b.store(b.i32_val(0), countPtr);
      // increment phase + release
      Value next = b.add(crtPhase, b.i32_val(1));
      // llPrintf("next[%d] = %d\n", {func.getArgument(3), next}, {true, true},
      // rewriter, targetInfo);
      auto release =
          LLVM::AtomicOrderingAttr::get(context, LLVM::AtomicOrdering::release);
      rewriter.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::xchg, phasePtr,
                                         next, LLVM::AtomicOrdering::release);
      rewriter.create<cf::BranchOp>(loc, exitBlock);
    }

    // spin block
    {
      rewriter.setInsertionPointToEnd(waitBlock);
      // check to see if the phase changed
      LLVM::LoadOp latest = b.load(i32_ty, phasePtr);
      latest->setAttr("ordering", LLVM::AtomicOrderingAttr::get(
                                      context, LLVM::AtomicOrdering::acquire));
      latest->setAttr("alignment", rewriter.getI64IntegerAttr(4));
      Value same = b.icmp_eq(latest, crtPhase);
      // llPrintf("latest[%d] = %d (%d)\n", {func.getArgument(3), latest, same},
      // {true, true, true}, rewriter, targetInfo);
      rewriter.create<cf::CondBranchOp>(loc, same, waitBlock, afterSpinBlock);
    }

    // after spin block
    {
      rewriter.setInsertionPointToEnd(afterSpinBlock);
      auto acquire =
          LLVM::AtomicOrderingAttr::get(context, LLVM::AtomicOrdering::acquire);
      rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acquire);
      rewriter.create<cf::BranchOp>(loc, exitBlock);
    }

    // exit block
    {
      rewriter.setInsertionPointToEnd(exitBlock);
      rewriter.create<LLVM::ReturnOp>(loc, ValueRange());
    }

    return func;
  }

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: delete the barrier if num warps is 1
    auto barrierFunc = getOrCreateCpuBarrier(op.getLoc(), rewriter);

    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);

    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    unsigned int numWarps =
        mlir::cast<mlir::IntegerAttr>(moduleOp->getAttr("ttg.num-warps"))
            .getInt();
    Value numThreads = b.i32_val(numWarps);

    // TMP
    Value threadId = rewriter.create<mlir::gpu::ThreadIdOp>(
        op.getLoc(), mlir::gpu::Dimension::x);
    threadId =
        rewriter.create<arith::IndexCastOp>(op.getLoc(), i32_ty, threadId);

    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    unsigned int sharedMemSizeInBytes =
        mlir::cast<mlir::IntegerAttr>(moduleOp->getAttr("ttg.shared")).getInt();

    // barrier shared memory allocation is implicit, so the ptrs we want are
    // offVal and offVal + 4
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                            targetInfo.getSharedAddressSpace());
    auto smemPtr = LLVM::getStackPointer(rewriter, funcOp);
    Value countPtr =
        b.gep(ptrTy, i8_ty, smemPtr, b.i32_val(sharedMemSizeInBytes));
    Value phasePtr =
        b.gep(ptrTy, i8_ty, smemPtr, b.i32_val(sharedMemSizeInBytes + 4));

    SmallVector<Value> args{countPtr, phasePtr, numThreads,
                            threadId}; // TODO: remove thread id
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, barrierFunc, args);
    return success();
  }

protected:
  const npu::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::npu::populateGPUtoLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ThreadIdOpToLLVM>(typeConverter, targetInfo, benefit);
  patterns.add<BlockIdOpToLLVM>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpToLLVM>(typeConverter, benefit);
  patterns.add<GpuBarrierOpToLLVM>(typeConverter, targetInfo, benefit);
}
