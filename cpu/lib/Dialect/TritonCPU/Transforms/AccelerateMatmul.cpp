#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TRITONCPUACCELERATEMATMUL
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritoncpu-accelerate-matmul"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

class OptimizeBlockedLayout : public mlir::OpRewritePattern<DotOp> {
public:
  OptimizeBlockedLayout(mlir::MLIRContext *context, int benefit)
      : OpRewritePattern<DotOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    LDBG("`tt.dot` to optimize: " << *dotOp);
    MLIRContext *context = dotOp->getContext();
    auto tensorTy = cast<RankedTensorType>(dotOp.getType());
    auto blockedEncoding =
        dyn_cast<gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
    if (!blockedEncoding)
      return failure();

    // Figure out a new per-thread shape that is more optimal for matmul on CPU.
    // We do not yet know what shape is optimal (TODO), but we need to
    // experiment with something. This currently limits the shape to the
    // smallest result dimension to avoid issues emitting too much code.
    // Apparently the coalesce pass has some utilities in
    // `getNumElementsPerThread` which may be useful.
    auto shape = tensorTy.getShape();
    unsigned min = *std::min_element(shape.begin(), shape.end());
    SmallVector<unsigned> newSizePerThread{min, min};
    auto oldSizePerThread = blockedEncoding.getSizePerThread();
    if (llvm::equal(oldSizePerThread, newSizePerThread))
      return failure();

    // Apply the new layout to the result type.
    auto newBlockedEncoding = gpu::BlockedEncodingAttr::get(
        context, newSizePerThread, blockedEncoding.getThreadsPerWarp(),
        blockedEncoding.getWarpsPerCTA(), blockedEncoding.getOrder(),
        blockedEncoding.getCGALayout());
    RankedTensorType newType = tensorTy.cloneWithEncoding(newBlockedEncoding);

    // Convert the layout of the incoming types. We likely should not be using
    // the same layout for A, B, and C (TODO).
    auto updateOperandEncoding = [](Value v, int opIdx,
                                    RankedTensorType newRetType,
                                    PatternRewriter &rewriter) {
      auto vType = cast<RankedTensorType>(v.getType());
      auto newVEncoding = gpu::DotOperandEncodingAttr::get(
          v.getContext(), opIdx, newRetType.getEncoding(), 0);
      auto newVType = vType.cloneWithEncoding(newVEncoding);
      return gpu::ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v);
    };
    auto a = updateOperandEncoding(dotOp.getA(), 0, newType, rewriter);
    auto b = updateOperandEncoding(dotOp.getB(), 1, newType, rewriter);
    auto oldC = dotOp.getC();
    auto c = gpu::ConvertLayoutOp::create(
        rewriter, oldC.getLoc(),
        oldC.getType().cloneWithEncoding(newBlockedEncoding), oldC);

    // Create the new dot op with updated types and swap it in for the old one.
    auto newDot =
        DotOp::create(rewriter, dotOp.getLoc(), newType, a, b, c,
                      dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
    rewriter.replaceOpWithNewOp<gpu::ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                      newDot.getResult());

    return success();
  }
};

} // namespace

class TritonCPUAccelerateMatmulPass
    : public impl::TritonCPUAccelerateMatmulBase<
          TritonCPUAccelerateMatmulPass> {
public:
  using impl::TritonCPUAccelerateMatmulBase<
      TritonCPUAccelerateMatmulPass>::TritonCPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    patterns.add<OptimizeBlockedLayout>(context, benefitDefault);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
