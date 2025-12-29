#include "LaneMapAnalysis.h"

#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TRITONCPUSPLITBLOCKS
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

class SplitBlocksPattern : public OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  SplitBlocksPattern(MLIRContext *context, LaneMapAnalysis *laneMapAnalysis)
      : OpRewritePattern(context), laneMapAnalysis(laneMapAnalysis) {}

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {

    // TODO
    if (isPointwiseStore(cast<triton::StoreOp>(*op), *laneMapAnalysis)) {
      llvm::outs() << "Pointwise store found: " << op << "\n";
      return failure();
    }

    return failure();
  }

private:
  LaneMapAnalysis *laneMapAnalysis;
};

} // namespace

class TritonCPUSplitBlocksPass
    : public impl::TritonCPUSplitBlocksBase<TritonCPUSplitBlocksPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    LaneMapAnalysis *laneMapAnalysis = solver->load<LaneMapAnalysis>();
    if (failed(solver->initializeAndRun(m))) {
      llvm::report_fatal_error("failed to run LaneMapAnalysis");
    }

    RewritePatternSet patterns(context);
    patterns.add<SplitBlocksPattern>(context, laneMapAnalysis);

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
