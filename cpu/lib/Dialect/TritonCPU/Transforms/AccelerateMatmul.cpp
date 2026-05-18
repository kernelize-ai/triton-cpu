#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
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

// Walk backward from dotOp.getA() through optional convert_layout + load +
// optional convert_layout to find the for-loop iter arg that feeds the A
// pointer. Validates that the iter arg:
//   (a) is a tensor-of-pointers block arg belonging to forOp
//   (b) has exactly one tt.addptr user (iter arg as ptr, loop-invariant offset,
//       result feeds scf.yield)
//   (c) has no unexpected users beyond convert_layout / tt.load ops
// Returns {iterArg, addptrOp} on success, nullopt on failure.
static std::optional<std::pair<BlockArgument, triton::AddPtrOp>>
matchAPtrIterArg(Value val, scf::ForOp forOp) {
  if (auto cvt = val.getDefiningOp<gpu::ConvertLayoutOp>())
    val = cvt.getOperand();
  auto load = val.getDefiningOp<triton::LoadOp>();
  if (!load)
    return std::nullopt;

  Value ptr = load.getPtr();
  if (auto cvt = ptr.getDefiningOp<gpu::ConvertLayoutOp>())
    ptr = cvt.getOperand();
  auto iterArg = dyn_cast<BlockArgument>(ptr);
  if (!iterArg || iterArg.getOwner() != forOp.getBody())
    return std::nullopt;
  auto tensorTy = dyn_cast<RankedTensorType>(iterArg.getType());
  if (!tensorTy || !isa<triton::PointerType>(tensorTy.getElementType()))
    return std::nullopt;

  // Validate forward uses of the iter arg.
  triton::AddPtrOp addptr;
  for (Operation *user : iterArg.getUsers()) {
    if (auto ap = dyn_cast<triton::AddPtrOp>(user)) {
      if (addptr || ap.getPtr() != iterArg)
        return std::nullopt;
      if (!forOp.isDefinedOutsideOfLoop(ap.getOffset()))
        return std::nullopt;
      addptr = ap;
    } else if (!isa<gpu::ConvertLayoutOp, triton::LoadOp>(user)) {
      return std::nullopt;
    }
  }
  if (!addptr)
    return std::nullopt;
  if (!addptr.getResult().hasOneUse())
    return std::nullopt;

  // addptr result must feed into scf.yield.
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!llvm::is_contained(yield.getOperands(), addptr.getResult()))
    return std::nullopt;

  return std::make_pair(iterArg, addptr);
}

class CanonicalizeKLoop : public mlir::OpRewritePattern<DotOp> {
public:
  CanonicalizeKLoop(mlir::MLIRContext *context, int benefit)
      : OpRewritePattern<DotOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    // look for parent for op with appropriate conditions for canonicalization.
    // Namely, we want a loop carried accumulator and additional iter args that
    // only correspond to ptr arithmetic that is easily re-mapped

    auto forOp = dotOp->getParentOfType<scf::ForOp>();
    if (!forOp)
      return failure();
    if (!matchPattern(forOp.getStep(), m_One()))
      return failure();

    // is the accumulator loop carried?
    auto c = dyn_cast<BlockArgument>(dotOp.getC());
    if (!c)
      return failure();
    if (!c.hasOneUse())
      return failure();
    auto accParent = c.getOwner()->getParentOp();
    if (accParent != forOp)
      return failure();

    auto d = dotOp.getD();
    // d must feed scf.yield at the same position that c comes from, ensuring
    // the accumulator round-trips through the iter arg at a consistent index.
    unsigned accIdx = c.getArgNumber() - forOp.getNumInductionVars();
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yield.getOperand(accIdx) != d)
      return failure();

    auto aMatchResult = matchAPtrIterArg(dotOp.getA(), forOp);
    if (!aMatchResult)
      return failure();
    auto bMatchResult = matchAPtrIterArg(dotOp.getB(), forOp);
    if (!bMatchResult)
      return failure();

    // loop carried ptr args must not be used by for op users
    if (!forOp
             .getResult(aMatchResult->first.getArgNumber() -
                        forOp.getNumInductionVars())
             .use_empty())
      return failure();
    if (!forOp
             .getResult(bMatchResult->first.getArgNumber() -
                        forOp.getNumInductionVars())
             .use_empty())
      return failure();

    // matching complete, we can rewrite the loop (assuming no iter arg spills,
    // checked below)

    // create a new scf.for with only the accumulator iter arg
    auto newFor = scf::ForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(),
        ValueRange{forOp.getInitArgs()[c.getArgNumber() -
                                       forOp.getNumInductionVars()]});

    SmallVector<Value> newForRegionArgValues;
    newForRegionArgValues.push_back(newFor.getInductionVar());
    for (auto arg : forOp.getRegionIterArgs()) {
      if (arg == c) {
        newForRegionArgValues.push_back(newFor.getRegionIterArg(0));
      } else {
        // if not a ptr arg fail
        if (!llvm::is_contained({aMatchResult->first, bMatchResult->first},
                                arg))
          return failure();
        // arg replacement will not be used (uses already cleared by
        // replaceAllUsesWith)
        newForRegionArgValues.push_back(newFor.getInductionVar());
      }
    }

    // rewrite existing loop body before merging into the new loop
    rewriter.setInsertionPointToStart(newFor.getBody());

    // rewrite each dot operand to feed from the for loop induction variable by
    // computing the add ptr value directly
    Value kIV = newFor.getInductionVar();

    IRMapping mapping;
    mapping.map(kIV, newFor.getInductionVar());
    auto rewriteAddPtrForOperand = [&](Value operand, BlockArgument iterArg,
                                       triton::AddPtrOp addPtr) {
      Value initValue = forOp.getInitArgs()[iterArg.getArgNumber() -
                                            forOp.getNumInductionVars()];

      // rewrite addptr to compute the pointer for the current loop iteration:
      // new_addptr = iterArg + splat(kIV) * offset
      auto offset = addPtr.getOffset();
      Value kSplat = triton::SplatOp::create(rewriter, addPtr.getLoc(),
                                             offset.getType(), kIV);
      Value newOffset =
          arith::MulIOp::create(rewriter, offset.getLoc(), kSplat, offset);
      auto newAddPtr = triton::AddPtrOp::create(
          rewriter, addPtr.getLoc(), initValue.getType(), initValue, newOffset);
      // rewriter.replaceAllUsesWith(iterArg, newAddPtr.getResult());
      mapping.map(iterArg, newAddPtr.getResult());
    };

    rewriteAddPtrForOperand(dotOp.getA(), aMatchResult->first,
                            aMatchResult->second);
    rewriteAddPtrForOperand(dotOp.getB(), bMatchResult->first,
                            bMatchResult->second);

    for (auto &op : forOp.getBody()->without_terminator()) {
      if (&op == aMatchResult->second.getOperation() ||
          &op == bMatchResult->second.getOperation())
        continue; // already rewritten
      rewriter.clone(op, mapping);
    }
  
    // auto oldYield = cast<scf::YieldOp>(newFor.getBody()->getTerminator());
    scf::YieldOp::create(rewriter, newFor.getLoc(), ValueRange{d});

    for (auto [i, result] : llvm::enumerate(forOp.getResults())) {
      if (i == c.getArgNumber() - forOp.getNumInductionVars()) {
        rewriter.replaceAllUsesWith(result, newFor.getResult(0));
      } else {
        if (!result.use_empty()) {
          llvm_unreachable(
              "unexpected use of non-accumulator loop carried value");
        }
      }
    }
    // rewriter.eraseOp(forOp);
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
    if (optimizeBlockLayout) {
      patterns.add<OptimizeBlockedLayout>(context, benefitDefault);
    }
    if (canonicalizeKLoop) {
      patterns.add<CanonicalizeKLoop>(context, benefitDefault);
    }
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
