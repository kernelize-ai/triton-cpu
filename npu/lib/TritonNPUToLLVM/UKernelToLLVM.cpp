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

    auto vecSize = triton::gpu::getTotalElemsPerThread(tensorTy);

    auto fnName = SleefNameGenerator("exp")(tensorTy.getElementTypeBitWidth(),
                                            vecSize, op->getOperands());
    if (fnName.empty())
      return failure();

    llvm::errs() << "Processing op: " << op << "\n";
    auto operands = adaptor.getOperands();
    if (operands.size() != 1)
      return failure();

    llvm::errs() << "operand: " << operands[0] << "\n";
    SmallVector<Value> vecOperands;
    for (int i = 0; i < vecSize; i++) {
      vecOperands.push_back(b.extract_val(operands[0], i));
    }

    // convert from triton to LLVM-compatible vector type
    VectorType vecType = VectorType::get(vecSize, tensorTy.getElementType());
    llvm::errs() << "vecType: " << vecType << "\n";

    // get the decl
    auto funcOp = getFuncDecl(rewriter, fnName, {vecType}, vecType);
    llvm::errs() << "funcOp: " << funcOp << "\n";

    // pack the operands ?
    // Value opVec = packLLVector(loc, )
#if 1
    Value opVal = packLLVector(loc, vecOperands, rewriter);
#else
    Value opVal =
        packLLElements(loc, typeConverter, operands[0], rewriter, vecType);
#endif
    llvm::errs() << "opVal: " << opVal << "\n###\n\n";

    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, {opVal});
    llvm::errs() << "callOp: " << callOp << "\n";
    llvm::errs() << "callOp result: " << callOp.getResult() << "\n";
#if 1
    auto rets = unpackLLVector(loc, callOp.getResult(), rewriter);
    llvm::errs() << "rets.size(): " << rets.size() << "\n";
    assert(rets.size() == vecSize);
    auto ret = packLLElements(loc, typeConverter, rets, rewriter, tensorTy);
#else
    auto ret = packLLElements(loc, typeConverter, callOp.getResult(), rewriter,
                              tensorTy);
#endif
    llvm::errs() << "ret: " << ret << "\n";
    rewriter.replaceOp(op, ret);
    // rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, {opVal});
    return success();
#if 0
        // Create the LLVM operation for exp
        Value input = adaptor.getOperand();
        Value result = b.exp(input, llvmType);

        rewriter.replaceOp(op, result);
        return success();
#endif
  }
};

} // namespace

void mlir::triton::npu::populateUKernelToLLVMConversionPatterns(
    TritonNPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ExpOpConversion>(typeConverter, benefit);
}
