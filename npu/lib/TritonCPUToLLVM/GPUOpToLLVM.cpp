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

    assert(args.size() > 7 && "incorrect npu kernel function signature");
    auto funcArgIdx = args.size() + npu::kThreadIdOffset;
    assert(args[funcArgIdx].getType().isInteger(32) &&
           "Thread ID argument must be i32");
    rewriter.replaceOp(threadIdOp, args[funcArgIdx]);
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

    auto funcArgIdx = args.size() + npu::kProgramIdArgsOffset + programIdDim;
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
  auto argIdx = args.size() + npu::kProgramIdArgsOffset + 3 + axis;
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

    // void barrier(int* count, int* phase, int32_t num_workers)
    auto funcTy =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                    {ptr_ty(context), ptr_ty(context), i32_ty},
                                    /*vararg=*/false);

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto func = rewriter.create<LLVM::LLVMFuncOp>(
        moduleOp.getLoc(), kName, funcTy, LLVM::Linkage::Internal);
    auto setBarrierPtrAttrs = [&](unsigned idx) {
      func.setArgAttr(idx, "llvm.align",
                      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 64));
      func.setArgAttr(idx, "llvm.nonnull", rewriter.getUnitAttr());
      func.setArgAttr(idx, "llvm.nocapture", rewriter.getUnitAttr());
      func.setArgAttr(idx, "llvm.noalias", rewriter.getUnitAttr());
    };
    setBarrierPtrAttrs(0);
    setBarrierPtrAttrs(1);

    func->setAttr("llvm.nounwind", rewriter.getUnitAttr());

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
    Value crtPhase = b.load(i32_ty, phasePtr, /*align=*/64);

    // atomically increment the count
    Value countPtr = func.getArgument(0);
    auto ordering = LLVM::AtomicOrdering::acq_rel;

    Value old = rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, countPtr, b.i32_val(1), ordering);

    // check to see if we are the last thread to hit the barrier
    Value numWorkers = func.getArgument(2);
    Value arrived = b.add(old, b.i32_val(1));
    Value amLast = b.icmp_eq(arrived, numWorkers);
    rewriter.create<cf::CondBranchOp>(loc, amLast, lastBlock, waitBlock);

    // last block
    {
      rewriter.setInsertionPointToEnd(lastBlock);
      // reset count
      b.store(b.i32_val(0), countPtr, /*align=*/64);
      // increment phase + release
      Value next = b.add(crtPhase, b.i32_val(1));
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
      LLVM::LoadOp latest = b.load(i32_ty, phasePtr, /*align=*/64);
      latest->setAttr("ordering", LLVM::AtomicOrderingAttr::get(
                                      context, LLVM::AtomicOrdering::acquire));
      Value same = b.icmp_eq(latest, crtPhase);
      rewriter.create<cf::CondBranchOp>(loc, same, waitBlock, afterSpinBlock);
    }

    // after spin block
    {
      rewriter.setInsertionPointToEnd(afterSpinBlock);
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

    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    unsigned int sharedMemSizeInBytes =
        mlir::cast<mlir::IntegerAttr>(moduleOp->getAttr("ttg.shared")).getInt();

    // barrier shared memory allocation is implicit, so the ptrs we want are
    // offVal and offVal + 64
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                            targetInfo.getSharedAddressSpace());
    auto smemPtr = LLVM::getStackPointer(rewriter, funcOp);
    Value countPtr =
        b.gep(ptrTy, i8_ty, smemPtr, b.i32_val(sharedMemSizeInBytes));
    Value phasePtr =
        b.gep(ptrTy, i8_ty, smemPtr, b.i32_val(sharedMemSizeInBytes + 64));

    SmallVector<Value> args{countPtr, phasePtr, numThreads};
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
  patterns.add<ThreadIdOpToLLVM>(typeConverter, benefit);
  patterns.add<BlockIdOpToLLVM>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpToLLVM>(typeConverter, benefit);
  patterns.add<GpuBarrierOpToLLVM>(typeConverter, targetInfo, benefit);
}
