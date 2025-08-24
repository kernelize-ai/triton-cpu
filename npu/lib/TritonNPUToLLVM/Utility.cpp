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
             Value pred, Value falseVal, const LLVMTypeConverter *typeConverter,
             unsigned vecSize, unsigned alignment) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type ptrTy = typeConverter->convertType(ptr.getType());

  Block &predicatedLoad = npu::createPredicatedBlock(
      rewriter, loc, pred, falseVal, [&]() -> SmallVector<Value, 1> {
        auto loadVecTy = LLVM::getVectorType(elemTy, vecSize);
        Value loadVec = b.undef(loadVecTy);
        for (size_t ii = 0; ii < vecSize; ii++) {
          Value vecIdx =
              mlir::LLVM::createIndexConstant(rewriter, loc, typeConverter, ii);
          Value loadAddr = b.gep(ptrTy, elemTy, ptr, vecIdx);
          Value loadedValue = b.load(elemTy, loadAddr, alignment);
          loadVec = b.insert_element(loadVecTy, loadVec, loadedValue, vecIdx);
        }
        return {loadVec};
      });

  Value loadVal = *predicatedLoad.args_begin();

  return loadVal;
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value storeVal, Value pred, const LLVMTypeConverter *typeConverter,
             unsigned vecSize, unsigned alignment) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();

  for (size_t ii = 0; ii < vecSize; ++ii) {
    // create a predicated store block for each scalar element
    Value vecIdx =
        mlir::LLVM::createIndexConstant(rewriter, loc, typeConverter, ii);
    npu::createPredicatedBlock(rewriter, loc, pred, [&]() -> ArrayRef<Value> {
      Value storeAddr = b.bitcast(ptr, ptr_ty(ctx, 1));
      b.store(b.extract_element(elemTy, storeVal, vecIdx), storeAddr,
              alignment);
      return {};
    });
  }
}

} // namespace mlir::triton::npu
