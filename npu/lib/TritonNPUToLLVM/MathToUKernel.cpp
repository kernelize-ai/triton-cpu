

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

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns,
                           GetVecFnNameFn getVecFnName,
                           size_t vec_size_in_bits) {
  patterns.add<VecOpToFp32<OpTy>>(patterns.getContext());
  patterns.add<DecomposeToNativeVecs<OpTy>>(patterns.getContext(),
                                            vec_size_in_bits);
  patterns.add<VecOpToVecLibConversion<OpTy>>(patterns.getContext(),
                                              getVecFnName);
}

class MathToUKernelPass
    : public mlir::triton::npu::impl::MathToUKernelPassBase<MathToUKernelPass> {
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
