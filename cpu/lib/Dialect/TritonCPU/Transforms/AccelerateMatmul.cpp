#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

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

class MaterializeAccumulator : public mlir::OpRewritePattern<DotOp> {
public:
  using OpRewritePattern<DotOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();

    auto accumulator = dotOp.getC();

    Operation *accumulatorSrcOp = nullptr;
    Operation *accumulatorParent = accumulator.getDefiningOp();
    if (!accumulatorParent) {
      auto blockArg = dyn_cast<BlockArgument>(accumulator);
      if (blockArg) {
        auto parentOp = blockArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          auto initVal = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
          if (initVal.getDefiningOp()) {
            accumulatorSrcOp =
                dyn_cast<arith::ConstantOp>(initVal.getDefiningOp());
          }
        }
      }
    } else {
      accumulatorSrcOp = dyn_cast<arith::ConstantOp>(accumulatorParent);
    }
    if (!accumulatorSrcOp)
      return failure();

    Location loc = accumulatorSrcOp->getLoc();

    // 1. find the appropriate location
    if (!accumulatorParent) {
      auto blockArg = cast<BlockArgument>(accumulator);
      auto parentOp = blockArg.getOwner()->getParentOp();
      rewriter.setInsertionPoint(parentOp);
    } else {
      rewriter.setInsertionPoint(accumulatorSrcOp);
    }

    // 2. create the accumulator source in shared memory
    auto accumulatorTensorType = cast<RankedTensorType>(accumulator.getType());
    auto accumulatorEncoding =
        dyn_cast<gpu::BlockedEncodingAttr>(accumulatorTensorType.getEncoding());
    if (!accumulatorEncoding)
      return failure();

    // create a shared linear encoding that describes the layout of the
    // accumulator.
    SmallVector<unsigned> shape(accumulatorTensorType.getShape().begin(),
                                accumulatorTensorType.getShape().end());
    LinearLayout layout =
        identityStandardND(StringAttr::get(context, "offset"), shape,
                           accumulatorEncoding.getOrder());
    layout *= LinearLayout::identity1D(1, StringAttr::get(context, "block"),
                                       StringAttr::get(context, "dim0"));
    layout = layout.transposeOuts(standardOutDimNames(context, shape.size()));

    Attribute SharedMemorySpace = gpu::SharedMemorySpaceAttr::get(context);
    auto sharedEncoding =
        gpu::SharedLinearEncodingAttr::get(context, layout, /*alignment=*/4);
    auto accMemDesc = gpu::MemDescType::get(
        accumulatorTensorType.getShape(),
        accumulatorTensorType.getElementType(), sharedEncoding,
        SharedMemorySpace, /*mutable=*/true);

    auto alloc = gpu::LocalAllocOp::create(rewriter, loc, accMemDesc,
                                           accumulatorSrcOp->getResult(0));

    // 3. replace the accumulator usage in the dot op with a load from shared
    // memory
    rewriter.setInsertionPoint(dotOp);
    auto loadedAcc = gpu::LocalLoadOp::create(
        rewriter, loc, accumulatorTensorType, alloc.getResult());

    auto newDot = rewriter.replaceOpWithNewOp<triton::DotOp>(
        dotOp, dotOp.getType(), dotOp.getA(), dotOp.getB(), loadedAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    // 4. store the return value of the dot op back to shared memory, dropping
    // the loop carried accumulator if needed
    gpu::LocalStoreOp::create(rewriter, loc, newDot.getD(), alloc.getResult());

    // 5. clean up the loop, if it exists
    if (auto blockArg = dyn_cast<BlockArgument>(accumulator)) {
      auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      unsigned iterArgIdx = blockArg.getArgNumber() - 1;

      // Replace the for result's downstream uses with a local_load from alloc.
      rewriter.setInsertionPointAfter(forOp);
      auto finalLoad = gpu::LocalLoadOp::create(
          rewriter, loc, accumulatorTensorType, alloc.getResult());
      forOp.getResult(iterArgIdx).replaceAllUsesWith(finalLoad);

      // Turn the yield operand into a pass-through of the iter arg itself.
      // Canonicalize will recognize this and remove the dead iter arg.
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      rewriter.modifyOpInPlace(
          yieldOp, [&]() { yieldOp.setOperand(iterArgIdx, blockArg); });
    }

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
    // patterns.add<OptimizeBlockedLayout>(context, benefitDefault);
    patterns.add<MaterializeAccumulator>(context, benefitDefault);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
