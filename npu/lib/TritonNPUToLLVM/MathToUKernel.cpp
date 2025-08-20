#include "npu/include/TritonNPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#include "triton/Dialect/TritonGPU/IR/Dialect.h"
// #include "triton/Conversion/TritonGPUToLLVM/Utility.h" // vec_ty

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

LLVM::LLVMFuncOp getFuncDecl(PatternRewriter &rewriter,
                             StringRef funcName, SmallVector<Type> argsType,
                             Type resultType) {
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
    llvm::errs() << "op type: " << op.getType() << "\n";
    RankedTensorType tensorTy = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorTy || tensorTy.getRank() > 1)
      return failure();

    auto vecSize = triton::gpu::getTotalElemsPerThread(tensorTy);

    auto fnName = getVecFnName(op, tensorTy.getElementTypeBitWidth(),
                               vecSize);
    if (fnName.empty())
      return failure();

    if (op->getNumOperands() > 1) {
        op.emitWarning("Multiple operands for uKernel operation not yet supported.");
        return failure();
    }

    if (op->getOperand(0).getType() != tensorTy) {
      op.emitWarning("Mismatched operand and result type for uKernel operation.");
      return failure();
    }

    // convert from triton to LLVM-compatible vector type 
    VectorType vecType = VectorType::get(vecSize, tensorTy.getElementType());
    llvm::errs() << "vecType: " << vecType << "\n";

    // get the decl 
    auto funcOp = getFuncDecl(rewriter, fnName, 
                              {vecType},
                              vecType);
    llvm::errs() << "funcOp: " << funcOp << "\n";

    // TODO: for now let's jsut wrap this all up in the generic LLVM lowering, but later let's work on a design where we have a generic uKernel dialect or op (or set of ops) that we use to wrap candidate ttgpu ops for lowering in a ttgpu pass, then lower those in the llvm lowering pass. 
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, op->getOperands());
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
