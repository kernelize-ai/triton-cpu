#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"

namespace mlir {
namespace triton {
namespace cpu {

#define DEBUG_TYPE "tritoncpu-loop-peeling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_LOOPPEELINGPASS
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

inline bool programIdOpIsInductionVar(Operation *op) {
  auto parentLoop = dyn_cast<scf::ForOp>(op->getParentOp());
  if (parentLoop && parentLoop->hasAttr("program_id_loop")) {
    return true;
  }
  return false;
}

} // namespace

struct LoopPeelingPass : public impl::LoopPeelingPassBase<LoopPeelingPass> {
  using impl::LoopPeelingPassBase<LoopPeelingPass>::LoopPeelingPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    DenseSet<Operation *> candidateMasks;

    moduleOp->walk([&](triton::LoadOp loadOp) {
      if (auto mask = loadOp.getMask()) {
        LDBG("Evaluating candidate mask for loop peeling: " << mask);

        SetVector<Operation *> slice;
        (void)getBackwardSlice(mask, &slice);

        if (isa<triton::GetProgramIdOp>(slice.front()) &&
            programIdOpIsInductionVar(slice.front())) {
          LDBG("Adding mask to candidate set");
          candidateMasks.insert(mask.getDefiningOp());
        }
      }
    });

    if (!candidateMasks.empty() && candidateMasks.size() == 1) {
      // peel loop
      auto forOp = cast<scf::ForOp>((*candidateMasks.begin())->getParentOp());
      mlir::triton::peelLoopEpilogue(forOp);
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
