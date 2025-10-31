#include "Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::cpu {

Value llPrintf(StringRef msg, ValueRange args, ArrayRef<bool> isSigned,
               RewriterBase &rewriter, const cpu::TargetInfo &targetInfo,
               int *formatStrByteCount) {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  targetInfo.printf(rewriter, msgValue, msgNewline.size_in_bytes(), args,
                    isSigned);
  if (formatStrByteCount)
    *formatStrByteCount = msgNewline.size_in_bytes();
  return msgValue;
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, std::optional<unsigned> alignment) {
  return cpu::MaskedLoadOp::create(
             rewriter, loc, elemTy, ptr, pred, falseVal,
             alignment ? IntegerAttr::get(rewriter.getI32Type(), *alignment)
                       : IntegerAttr())
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, std::optional<unsigned> alignment) {
  cpu::MaskedStoreOp::create(
      rewriter, loc, ptr, val, pred,
      alignment ? IntegerAttr::get(rewriter.getI32Type(), *alignment)
                : IntegerAttr());
}

} // namespace mlir::triton::cpu
