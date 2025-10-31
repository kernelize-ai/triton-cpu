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
                         bool isWarpSync) const {
  if (isWarpSync) {
    llvm::report_fatal_error("warp sync barrier not supported on CPU");
  }
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier();
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

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  assert(iWarpSize == 1 && "only size 1 warps supported for reductions on CPU");
  unsigned int numWarps =
      mlir::cast<mlir::IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();

  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();

  // store our value to smem
  Value slot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, threadId);
  storeDShared(rewriter, loc, slot, std::nullopt, val, b.true_val());

  barrier(loc, rewriter);

  // compute target lane id
  Value targetThreadId = b.xor_(threadId, b.i32_val(i));
  Value targetPtr =
      b.gep(ptrTy, int_ty(elemSizeBits), smemBase, targetThreadId);
  // load from target lane (note we could use 64B alignments here since we're in
  // the sync region)
  Value loaded = mlir::triton::cpu::llLoad(
      rewriter, loc, targetPtr, val.getType(),
      b.icmp_slt(targetThreadId, b.i32_val(numWarps)), val);
  barrier(loc, rewriter);
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
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // no need to reduce if only one lane is involved
  if (numLaneToReduce == 1)
    return true;

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  unsigned int numWarps =
      mlir::cast<mlir::IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
  // Fallback to shuffleXOR (TODO: implement masked warpReduce if performance is
  // better)
  if (numLaneToReduce < numWarps)
    return false;

  Operation *reduceOp = op.getSingleCombiner();
  if (!reduceOp)
    return false;

  assert(acc.size() == 1 && "only single value reduction supported on CPU");
  auto val = acc[0];

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          getSharedAddressSpace());
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, *this, op);
  Value threadId = getThreadId(rewriter, loc);

  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();
  Value crtVal = val;
  for (unsigned other = 1; other < numLaneToReduce; ++other) {
    Value otherThreadId =
        b.urem(b.add(threadId, b.i32_val(other)), b.i32_val(numLaneToReduce));
    Value otherSlot =
        b.gep(ptrTy, int_ty(elemSizeBits), smemBase, otherThreadId);
    Value otherVal = loadDShared(rewriter, loc, otherSlot, std::nullopt,
                                 val.getType(), b.true_val());

    IRMapping mapping;
    mapping.map(reduceOp->getOperand(0), crtVal);
    mapping.map(reduceOp->getOperand(1), otherVal);
    crtVal = rewriter.clone(*reduceOp, mapping)->getResult(0);
  }
  acc[0] = crtVal;
  barrier(loc, rewriter); // barrier before writing back the reduced values to
                          // shared memory
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
