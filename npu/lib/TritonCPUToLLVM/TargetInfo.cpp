#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"

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

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}

} // namespace

namespace mlir::triton::npu {

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm::report_fatal_error("ballot not supported on NPU");
  return Value();
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  if (isWarpSync) {
    llvm::report_fatal_error("warp sync barrier not supported on NPU");
  }
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier();
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value())
    llvm::report_fatal_error(
        "NPU does not support cross-CTA shared memory transfers");
  mlir::triton::npu::llStore(rewriter, loc, ptr, val, pred, /*alignment=*/4);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value())
    llvm::report_fatal_error(
        "NPU does not support cross-CTA shared memory transfers");
  Value falseVal = rewriter.create<LLVM::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  auto load =
      mlir::triton::npu::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal,
                                /*alignment=*/4);
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

  // Unfortunately we do not have access to the original reduction op here.
  // However, because we are inside a reduction, the total shared memory should
  // be enough for this op due to shared memory being used to store other values
  // during the reductions. And the barrier prevents any other users of the
  // shared memory from interfering with this particular shuffle. If triton were
  // to buffer data in shared memory this could be a problem, though - something
  // to keep an eye on.
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          getSharedAddressSpace());
  auto funcOp = val.getParentRegion()->getParentOfType<FunctionOpInterface>();
  Value smemBase = LLVM::getStackPointer(rewriter, funcOp);

  Value threadId = getThreadId(rewriter, loc);

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  assert(iWarpSize == 1 && "only size 1 warps supported for reductions on NPU");

  Value warpSize = b.i32_val(iWarpSize);
  Value laneId = b.urem(threadId, warpSize);

  // write to our slot
  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();
  Value slot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, threadId);
  storeDShared(rewriter, loc, slot, std::nullopt, val, b.true_val());

  barrier(loc, rewriter);

  // read from our neighbor
  Value neighbor = b.xor_(threadId, b.i32_val(i));
  Value neighborSlot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, neighbor);
  Value loaded = loadDShared(rewriter, loc, neighborSlot, std::nullopt,
                             val.getType(), b.true_val());

  barrier(loc, rewriter);
  return loaded;
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm::report_fatal_error("shuffleUp not supported on NPU");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm::report_fatal_error("shuffleIdx not supported on NPU");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm::report_fatal_error("shuffleIdx not supported on NPU");
  return Value();
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm::report_fatal_error("permute not supported on NPU");
  return Value();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return rewriter.create<mlir::triton::cpu::BlockIdOp>(loc, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // no need to reduce if only one lane is involved
  if (numLaneToReduce == 1)
    return true;

  Operation *reduceOp = op.getSingleCombiner();
  if (!reduceOp)
    return false;

  assert(acc.size() == 1 && "only single value reduction supported on NPU");
  auto val = acc[0];

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          getSharedAddressSpace());
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, *this, op);
  Value threadId = getThreadId(rewriter, loc);

  // only thread (warp) 0 reduces
  Value zero = b.i32_val(0);
  Value isWarp0 = b.icmp_eq(threadId, zero);

  // Split the current block
  Block *currentBlock = rewriter.getBlock();
  Block *thenBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *continueBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());
  continueBlock->addArgument(val.getType(), loc);

  // Create conditional branch
  rewriter.setInsertionPointToEnd(currentBlock);
  Value undef = b.undef(val.getType());
  rewriter.create<cf::CondBranchOp>(loc, isWarp0, thenBlock, ArrayRef<Value>{},
                                    continueBlock, ArrayRef<Value>{undef});

  // Set insertion point to then block for reduction logic
  rewriter.setInsertionPointToStart(thenBlock);

  // Thread 0 reduces
  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();
  Value crtVal = val;
  for (unsigned other = 1; other < numLaneToReduce; ++other) {
    Value otherThreadId = b.i32_val(other);
    Value otherSlot =
        b.gep(ptrTy, int_ty(elemSizeBits), smemBase, otherThreadId);
    Value otherVal = loadDShared(rewriter, loc, otherSlot, std::nullopt,
                                 val.getType(), b.true_val());

    IRMapping mapping;
    mapping.map(reduceOp->getOperand(0), crtVal);
    mapping.map(reduceOp->getOperand(1), otherVal);
    crtVal = rewriter.clone(*reduceOp, mapping)->getResult(0);
  }

  // write back is handled by the caller
  rewriter.create<cf::BranchOp>(loc, continueBlock, ValueRange{crtVal});
  rewriter.setInsertionPointToStart(continueBlock);
  acc[0] = continueBlock->getArgument(0);
  return true;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  llvm::report_fatal_error("getMulhiFuncName not supported on NPU");
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
  rewriter.create<LLVM::CallOp>(loc, funcOp, newArgs);
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
  llvm::report_fatal_error("assertFail not supported on NPU");
}

int TargetInfo::getSharedAddressSpace() const { return 0; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const { return 0; }

bool TargetInfo::supportVectorizedAtomics() const {
  return false; // NPU does not support vectorized atomics
}

} // namespace mlir::triton::npu
