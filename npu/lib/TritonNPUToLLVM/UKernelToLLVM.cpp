#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "npu/include/TritonNPUToLLVM/TypeConverter.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class SleefNameGenerator {
public:
  SleefNameGenerator(StringRef baseName, unsigned ulp = 10)
      : baseName(baseName), ulpSuffix(4, '\0') {
    if (ulp == 0) {
      ulpSuffix = "";
    } else {
      char buf[5]; // 4 char suffix + '\0' added by snprintf
      snprintf(buf, 5, "_u%02u", ulp);
      ulpSuffix = buf;
    }
  }

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange /*operands*/) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    if (vecSize < 128)
      return "";
    return "Sleef_" + baseName + (bitwidth == 32 ? "f" : "d") +
           std::to_string(numel) + ulpSuffix;
  }

private:
  std::string baseName;
  std::string ulpSuffix;
};

LLVM::LLVMFuncOp getFuncDecl(PatternRewriter &rewriter, StringRef funcName,
                             SmallVector<Type> argsType, Type resultType) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *ctx = rewriter.getContext();

  auto funcType =
      LLVM::LLVMFunctionType::get(resultType, argsType, /*isVarArg*/ false);

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                           funcType);
}

struct ExpOpConversion : public ConvertOpToLLVMPattern<math::ExpOp> {
  ExpOpConversion(TritonNPUToLLVMTypeConverter &converter,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}
  using Base = ConvertOpToLLVMPattern<math::ExpOp>;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  size_t vec_size_in_bits = 128;

  LogicalResult
  matchAndRewrite(math::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    RankedTensorType tensorTy = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorTy || tensorTy.getRank() > 1)
      return failure();

    unsigned vecSize = vec_size_in_bits / tensorTy.getElementTypeBitWidth();
    VectorType vecType = VectorType::get(vecSize, tensorTy.getElementType());
    unsigned elementsPerThread = triton::gpu::getTotalElemsPerThread(tensorTy);

    auto fnName = SleefNameGenerator("exp")(tensorTy.getElementTypeBitWidth(),
                                            vecSize, op->getOperands());
    if (fnName.empty())
      return failure();

    auto operands = adaptor.getOperands();
    if (operands.size() != 1)
      return failure();

    SmallVector<Value> opElements;
    for (int i = 0; i < elementsPerThread; i++) {
      opElements.push_back(b.extract_val(operands[0], i));
    }

    auto funcOp = getFuncDecl(rewriter, fnName, {vecType}, vecType);

    SmallVector<Value> resultVals;
    for (unsigned vecStart = 0; vecStart < elementsPerThread;
         vecStart += vecSize) {
      unsigned crtVecSize = std::min(vecSize, elementsPerThread - vecStart);

      SmallVector<Value> slice = llvm::to_vector(
          ArrayRef<Value>(opElements).slice(vecStart, crtVecSize));
      if (slice.size() < vecSize) {
        // pad the slice with undefs
        slice.insert(slice.end(), vecSize - slice.size(),
                     b.undef(vecType.getElementType()));
      }

      Value vec = packLLVector(loc, slice, rewriter);
      auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, {vec});
      auto results = unpackLLVector(loc, callOp.getResult(), rewriter);
      for (unsigned i = 0; i < crtVecSize; i++) {
        resultVals.push_back(results[i]);
      }
    }

    auto rets =
        packLLElements(loc, typeConverter, resultVals, rewriter, tensorTy);
    rewriter.replaceOp(op, rets);
    return success();
  }
};

} // namespace

void mlir::triton::npu::populateUKernelToLLVMConversionPatterns(
    TritonNPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ExpOpConversion>(typeConverter, benefit);
}
