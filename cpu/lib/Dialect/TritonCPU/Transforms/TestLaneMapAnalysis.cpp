#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "cpu/include/Analysis/LaneMapAnalysis.h"

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TESTLANEMAPANALYSISPASS
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

struct TestLaneMapAnalysisPass
    : public impl::TestLaneMapAnalysisPassBase<TestLaneMapAnalysisPass> {
  using impl::TestLaneMapAnalysisPassBase<
      TestLaneMapAnalysisPass>::TestLaneMapAnalysisPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto solver = mlir::createDataFlowSolver();
    LaneMapAnalysis *analysis = solver->load<LaneMapAnalysis>();

    if (failed(solver->initializeAndRun(mod))) {
      signalPassFailure();
      return;
    }

    mod.walk([&](triton::StoreOp st) {
      llvm::errs() << "STORE ";
      if (isPointwiseStore(st, *analysis))
        llvm::errs() << "POINTWISE\n";
      else
        llvm::errs() << "NOT_POINTWISE\n";
    });
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
