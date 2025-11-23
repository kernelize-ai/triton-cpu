#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
namespace cpu {

#define DEBUG_TYPE "tritoncpu-loop-peeling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_LOOPPEELINGPASS
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

void pruneMasks(DenseSet<Operation*>& candidateMasks) {
  for (auto it = candidateMasks.begin(); it != candidateMasks.end(); it++) {
    Operation* maskOp = *it;
    llvm::errs() << "maskOp = " << *maskOp << "\n";

    // take a backward slice and see if this op depends on the loop induction variable 
    // TODO: would be really nice if the induction variable was passed to get program id... 
  }
}

}

struct LoopPeelingPass : public impl::LoopPeelingPassBase<LoopPeelingPass> {
  using impl::LoopPeelingPassBase<LoopPeelingPass>::LoopPeelingPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    DenseSet<Operation*> candidateMasks;
    moduleOp->walk([&](triton::LoadOp loadOp) {
        if (auto mask = loadOp.getMask()) {
            LDBG("Evaluating candidate mask for loop peeling: " << mask);
            candidateMasks.insert(mask.getDefiningOp());
        }

    pruneMasks(candidateMasks);
    });
  }
};

}
}
}
