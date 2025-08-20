#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"

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

namespace mlir::triton::NPU {

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
  llvm::report_fatal_error("barrier not supported on NPU");
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  llvm::report_fatal_error(
      "NPU does not support cross-CTA shared memory transfers");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  llvm::report_fatal_error(
      "NPU does not support cross-CTA shared memory transfers");
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm::report_fatal_error("shuffleXor not supported on NPU");
  return Value();
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

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return rewriter.create<mlir::triton::cpu::BlockIdOp>(loc, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // not supported on CPU
  return false;
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

} // namespace mlir::triton::NPU
