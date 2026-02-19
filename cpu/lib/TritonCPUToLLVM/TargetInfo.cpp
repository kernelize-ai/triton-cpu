#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "Utility.h"

using namespace mlir;

namespace {

LLVM::LLVMFuncOp getPrintfDeclaration(RewriterBase &rewriter) {
  auto *context = rewriter.getContext();
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("printf");

  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, ptr_ty(context),
                                              /*isVarArg=*/true);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context), funcName,
                                  funcType);
}

} // namespace

namespace mlir::triton::cpu {

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return mlir::LLVM::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm::report_fatal_error("ballot not supported on CPU");
  return Value();
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(targets);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  barrier(loc, rewriter, triton::gpu::AddrSpace::All);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value())
    llvm::report_fatal_error(
        "CPU does not support cross-CTA shared memory transfers");

  Type elemTy = val.getType();
  if (isa<VectorType>(elemTy) && !isa<VectorType>(pred.getType())) {
    // TODO: we should handle this case in the llLoad lowering
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const auto numElements = cast<VectorType>(elemTy).getNumElements();
    VectorType predTy = VectorType::get(numElements, i1_ty);
    Value vecPred = b.undef(predTy);
    for (unsigned i = 0; i < numElements; i++) {
      vecPred = b.insert_element(predTy, vecPred, pred, b.i32_val(i));
    }
    pred = vecPred;
  }
  mlir::triton::cpu::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value())
    llvm::report_fatal_error(
        "CPU does not support cross-CTA shared memory transfers");
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getZeroAttr(elemTy));
  if (isa<VectorType>(elemTy) && !isa<VectorType>(pred.getType())) {
    auto vecTy = cast<VectorType>(elemTy);
    SmallVector<Value> predVec(vecTy.getNumElements(), pred);
    pred = packLLVector(loc, predVec, rewriter);
  }
  auto load =
      mlir::triton::cpu::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
  return load;
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  int shared = 0;
  if (auto sharedAttr = mod->getAttr("ttg.shared")) {
    shared = cast<IntegerAttr>(sharedAttr).getInt();
  }
  assert(shared > 0 &&
         "shared memory allocation is required for shuffle XOR operation");

  // Warps have their own shared memory buffer for synchronization after shared
  // memory allocations for the kernel. The warp shared memory buffer alocates
  // 64 bytes per warp, making it 64-byte aligned and large enough for all
  // scalar reductions. The barrier shared memory buffer consists of two 64-byte
  // allocations after the warp synchronization shared memory buffer.
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          getSharedAddressSpace());
  auto funcOp = val.getParentRegion()->getParentOfType<FunctionOpInterface>();
  // warp synchronization buffer is after per-op shared memory allocations
  Value smemBase = b.gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, funcOp),
                         b.i32_val(shared));

  Value threadId = getThreadId(rewriter, loc);
  // TODO: If we allow numWarps > 1 we should compute laneId here

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  unsigned int numWarps =
      mlir::cast<mlir::IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
  assert(numWarps == 1 && "only 1 warp supported for xor reductions on CPU");

  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();

  // store our value to smem
  Value slot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, threadId);
  storeDShared(rewriter, loc, slot, std::nullopt, val, b.true_val());

  b.barrier(triton::gpu::AddrSpace::None);

  // compute target lane id
  Value targetThreadId = b.xor_(threadId, b.i32_val(i));
  Value targetPtr =
      b.gep(ptrTy, int_ty(elemSizeBits), smemBase, targetThreadId);
  // load from target lane
  Value loaded = mlir::triton::cpu::llLoad(
      rewriter, loc, targetPtr, val.getType(),
      b.icmp_slt(targetThreadId, b.i32_val(iWarpSize)), val);
  b.barrier(triton::gpu::AddrSpace::None);
  return loaded;
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm::report_fatal_error("shuffleUp not supported on CPU");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm::report_fatal_error("shuffleIdx not supported on CPU");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm::report_fatal_error("shuffleIdx not supported on CPU");
  return Value();
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm::report_fatal_error("permute not supported on CPU");
  return Value();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return mlir::triton::cpu::BlockIdOp::create(rewriter, loc, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask) const {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  unsigned int warpSize =
      mlir::cast<mlir::IntegerAttr>(mod->getAttr("ttg.threads-per-warp"))
          .getInt();

  // We only support full warp reduction (i.e. all lanes) on CPU for now.
  // Partial reductions (sub-groups) fallback to shuffleXor.
  if (reduceLaneIdMask != (warpSize - 1))
    return false;

  Operation *combinerOp = op.getSingleCombiner();
  if (!combinerOp)
    return false;

  assert(acc.size() == 1 && "only single value reduction supported on CPU");
  auto val = acc[0];

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          getSharedAddressSpace());
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, *this, op);
  Value threadId = getThreadId(rewriter, loc);
  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();

  // 1. All threads store their initial values to shared memory slots
  // corresponding to their thread ID.
  Value slot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, threadId);
  storeDShared(rewriter, loc, slot, std::nullopt, val, b.true_val());

  // Wait for all threads to finish writing.
  b.barrier(triton::gpu::AddrSpace::None);

  // 2. Perform the reduction sequentially on Thread 0 (Leader Thread).
  // Thread 0 iterates through all other threads' values and accumulates them.
  Value isLeaderThread = b.icmp_eq(threadId, b.i32_val(0));
  auto [prevBlock, reductionBlock, continuationBlock] =
      createIfBlock(rewriter, loc, isLeaderThread);
  rewriter.setInsertionPointToStart(reductionBlock);

  Value accumulatedVal = val;
  for (unsigned otherIdx = 1; otherIdx < warpSize; ++otherIdx) {
    Value otherThreadId = b.i32_val(otherIdx);
    Value otherSlot =
        b.gep(ptrTy, int_ty(elemSizeBits), smemBase, otherThreadId);
    Value otherVal = loadDShared(rewriter, loc, otherSlot, std::nullopt,
                                 val.getType(), b.true_val());

    IRMapping mapping;
    mapping.map(combinerOp->getOperand(0), accumulatedVal);
    mapping.map(combinerOp->getOperand(1), otherVal);
    accumulatedVal = rewriter.clone(*combinerOp, mapping)->getResult(0);
  }
  // Store the final reduced value back into Thread 0's slot so others can read
  // it.
  storeDShared(rewriter, loc, slot, std::nullopt, accumulatedVal, b.true_val());

  rewriter.setInsertionPointToStart(continuationBlock);

  // Wait for Thread 0 to finish reduction.
  b.barrier(triton::gpu::AddrSpace::None);

  // 3. All threads load the reduced value from Thread 0's slot
  Value leaderSlot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, b.i32_val(0));
  acc[0] = loadDShared(rewriter, loc, leaderSlot, std::nullopt, val.getType(),
                       b.true_val());

  return true;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  llvm::report_fatal_error("getMulhiFuncName not supported on CPU");
  return "";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getPrintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value, 16> newArgs;
  newArgs.push_back(formatStrStart);
  newArgs.append(args.begin(), args.end());
  LLVM::CallOp::create(rewriter, loc, funcOp, newArgs);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  llvm::report_fatal_error("assertFail not supported on CPU");
}

int TargetInfo::getSharedAddressSpace() const { return 0; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const { return 0; }

bool TargetInfo::supportVectorizedAtomics() const {
  return false; // CPU does not support vectorized atomics
}

} // namespace mlir::triton::cpu
