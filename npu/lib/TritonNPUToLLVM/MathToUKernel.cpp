#include "npu/include/TritonNPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
namespace npu {
#define GEN_PASS_DEF_MATHTOUKERNEL
#include "npu/include/TritonNPUToLLVM/Passes.h.inc"
} // namespace npu
} // namespace triton
} // namespace mlir

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

using GetVecFnNameFn = std::function<std::string(
    unsigned /*bitwidth*/, unsigned /*numel*/, ValueRange /*operands*/)>;

template <typename OpT>
struct OpToVecLibConversion : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  virtual std::string getVecFnName(OpT op, unsigned bitwidth,
                                   unsigned numel) const = 0;

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    llvm::errs() << "Getting vector function name for op: " << op << "\n";
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getRank() > 1)
      return failure();

    auto fnName = getVecFnName(op, vecTy.getElementTypeBitWidth(),
                               vecTy.getNumElements());
    if (fnName.empty())
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, fnName));
    // Generate function declaration if it doesn't exists yet.
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      auto fnTy = FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc =
          rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), fnName, fnTy);
      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fnName, op.getType(),
                                              op->getOperands());
    return success();
  }
};

template <typename OpT>
struct VecOpToVecLibConversion : public OpToVecLibConversion<OpT> {
public:
  VecOpToVecLibConversion(MLIRContext *context, GetVecFnNameFn getVecFnName)
      : OpToVecLibConversion<OpT>(context), getVecFnNameImpl(getVecFnName) {}

  std::string getVecFnName(OpT op, unsigned bitwidth,
                           unsigned numel) const override {
    return getVecFnNameImpl(bitwidth, numel, op->getOperands());
  }

private:
  GetVecFnNameFn getVecFnNameImpl;
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns,
                           GetVecFnNameFn getVecFnName,
                           size_t vec_size_in_bits) {
  //   patterns.add<VecOpToFp32<OpTy>>(patterns.getContext());
  //   patterns.add<DecomposeToNativeVecs<OpTy>>(patterns.getContext(),
  // vec_size_in_bits);
  patterns.add<VecOpToVecLibConversion<OpTy>>(patterns.getContext(),
                                              getVecFnName);
}

class MathToUKernelPass
    : public mlir::triton::npu::impl::MathToUKernelBase<MathToUKernelPass> {
protected:
  size_t vec_size_in_bits = 128;

public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);

    populatePatternsForOp<math::ExpOp>(patterns, SleefNameGenerator("exp"),
                                       vec_size_in_bits);

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace npu {

std::unique_ptr<OperationPass<ModuleOp>> createMathToUKernelPass() {
  return std::make_unique<MathToUKernelPass>();
}

} // namespace npu
} // namespace triton
} // namespace mlir
