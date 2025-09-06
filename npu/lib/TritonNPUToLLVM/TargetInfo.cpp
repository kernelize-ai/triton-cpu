#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Analysis/Utility.h"

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
  npu::llPrintf("storing to smem %p : %f", {ptr, val}, {false, false}, rewriter, *this);
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
  auto load = mlir::triton::npu::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal,
                                   /*alignment=*/4);
  npu::llPrintf("loading from smem %p : %f (%if)", {ptr, load, pred}, {false, false, false}, rewriter, *this);
  return load;
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
#if 1
  llvm::errs() << "shuffle val: " << val << "\n";
  llvm::errs() << "index: " << i << "\n";
  llvm::errs() << "val = " << val << "\n";
  llvm::errs() << "val type = " << val.getType() << "\n";

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // first make sure we have enough memory for this operation
  int shared = 0;
  if (auto sharedAttr = mod->getAttr("ttg.shared")) {
    shared = cast<IntegerAttr>(sharedAttr).getInt();
  }
  unsigned int elemSizeBits = val.getType().getIntOrFloatBitWidth();
#if 1
  // becasuse we are inside a reduction, the total shared memory should be enough for this op due to shared memory being used to store other values during the reductions
#else
  int totalMemorySizeBytes = i * elemSizeBits / 8; // TODO this isn't right, it should be num_warps not i * elemSizeByes 
  llvm::errs() << "total memory size for shuffle: " << totalMemorySizeBytes << "\n";
  if (shared < totalMemorySizeBytes) {
    mod->setAttr("ttg.shared", rewriter.getIntegerAttr(i32_ty, totalMemorySizeBytes)); 
  }
#endif 
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          getSharedAddressSpace());
  auto funcOp =
        val.getParentRegion()->getParentOfType<FunctionOpInterface>();
  Value smemBase = LLVM::getStackPointer(rewriter, funcOp);
  
  Value threadId = getThreadId(rewriter, loc);

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  assert(iWarpSize == 1 && "only size 1 warps supported for reductions on NPU");
  
  Value warpSize = b.i32_val(iWarpSize);
  Value laneId = b.urem(threadId, warpSize);

  // write to our slot
  llvm::errs() << "int_ty(elemSizeBits) = " << int_ty(elemSizeBits) << "\n";
  Value slot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, threadId);
  storeDShared(rewriter, loc, slot, std::nullopt, val, b.true_val());

  barrier(loc, rewriter);


  // read from our neighbor 
  Value neighbor = b.xor_(threadId, b.i32_val(i));
  llPrintf("thread %d warpsize %d lane %d reducing val %f from %d", {threadId, warpSize, laneId, val, neighbor}, {false, false, false, false, false}, rewriter, *this);

  Value neighborSlot = b.gep(ptrTy, int_ty(elemSizeBits), smemBase, neighbor);
  Value loaded = loadDShared(rewriter, loc, neighborSlot, std::nullopt, val.getType(), b.true_val());

  barrier(loc, rewriter);
  return loaded;
#endif 
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
#if 1
  // warp size on NPU is always 1, so we only need to reduce if multiple lanes in the block are participating. If so, fall back to shuffleXOR or other shuffle instruction + accumulator. 
  llvm::errs() << "numLaneToReduce = " << numLaneToReduce << "\n";
  return numLaneToReduce == 1;
#else
  llvm::errs() << "acc size: " << acc.size() << "\n";
  llvm::errs() << "numLaneToReduce: " << numLaneToReduce << "\n";
  llvm::errs() << "interleave: " << interleave << "\n";



  if (numLaneToReduce == 1) {
    return true;
  }
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
  if (numLaneToReduce != numLanes) {
    // is this a problem?
    op->emitWarning("NPU backend is only optimized for warp reductions across all warps in a block");
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  ReduceOpHelper helper(op);
  auto offset = helper.getThreadOffsetOnReductionAxis();
  llvm::errs() << "reduction offset = " << offset << "\n";
  auto smemShape = helper.getScratchRepShape();
  for (auto shape : smemShape) {
    llvm::errs() << "smem shape: " << shape << "\n";
  }

  Operation *reduxOp = op.getSingleCombiner();
  if (!reduxOp)
    return false;
  llvm::errs() << "reduxOp = " << *reduxOp << "\n";

  auto elemTy = acc[0].getType();
  llvm::errs() << "elemTy = " << elemTy << "\n";

  auto basePtr =
        LLVM::getSharedMemoryBase(loc, rewriter, *this, op.getOperation());
  for (unsigned i = 0; i < numLaneToReduce; i++) {
      auto ptrOffset = b.gep(basePtr.getType(), elemTy, basePtr, b.i32_val(i));
      Value ptrLoad = loadDShared(rewriter, loc, ptrOffset, std::nullopt, elemTy, b.true_val());
      llPrintf("Load for warp reduce index %d = %f\n", {LLVM::createConstantI32(loc, rewriter, i), ptrLoad}, {true, false}, rewriter, *this);
  }



  // reduce into lane 0


  assert(acc.size() == 1);
  llPrintf("warpReduce acc (num lanes = %d): %f", {LLVM::createConstantI32(loc, rewriter, numLaneToReduce), acc[0]}, {true, false}, rewriter, *this);
  // TODO: seeing 2 lanes to reduce seems like a sign... 
  // all warps on CPU are single-threaded, so warp reductions are not necessary
  return true;
#endif 
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
