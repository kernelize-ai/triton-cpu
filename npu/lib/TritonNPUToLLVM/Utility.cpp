#include "Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::npu {

Value llPrintf(StringRef msg, ValueRange args, ArrayRef<bool> isSigned,
               ConversionPatternRewriter &rewriter,
               const npu::TargetInfo &targetInfo, int *formatStrByteCount) {
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
             Value pred, Value falseVal, unsigned alignment) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Block &predicatedLoad = npu::createPredicatedBlock(
      rewriter, loc, pred, falseVal, [&]() -> SmallVector<Value, 1> {
        return {b.load(elemTy, ptr, alignment)};
      });
  Value loadVal = *predicatedLoad.args_begin();

  return loadVal;
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, unsigned alignment) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();

  npu::createPredicatedBlock(rewriter, loc, pred, [&]() -> ArrayRef<Value> {
    b.store(val, ptr, alignment);
    return {};
  });
}

} // namespace mlir::triton::npu
