#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "tritoncpu-tile-and-fuse"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TRITONCPUTILEANDFUSE
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

static bool isClosed(const llvm::SetVector<Operation *> &ops) {
  llvm::DenseSet<Operation *> set(ops.begin(), ops.end());
  for (Operation *op : ops) {
    for (Value r : op->getResults()) {
      for (Operation *user : r.getUsers()) {
        if (!set.contains(user))
          return false;
      }
    }
  }
  return true;
}

struct WrapElementwiseChain : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto genericParent = storeOp->getParentOfType<cpu::GenericOp>();
    if (genericParent) {
      return failure();
    }

    llvm::SetVector<Operation *> opsToClone;
    opsToClone.insert(storeOp);

    llvm::SmallVector<Value> queue;
    queue.push_back(storeOp.getValue());

    bool failed = false;
    while (!queue.empty()) {
      auto v = queue.pop_back_val();
      auto defOp = v.getDefiningOp();

      if (!defOp)
        continue;

      // allow load operations and elementwise operations in ttc.generic
      if (auto loadOp = dyn_cast<triton::LoadOp>(defOp)) {
        // load ops terminate the chain
        opsToClone.insert(defOp);
        continue;
      }
      if (isa<arith::ArithDialect>(defOp->getDialect()) &&
          defOp->hasTrait<OpTrait::Elementwise>()) {
        opsToClone.insert(defOp);
      } else {
        failed = true;
        break;
      }

      for (auto operand : defOp->getOperands()) {
        queue.push_back(operand);
      }
    }
    if (failed) {
      return failure();
    }

    if (!isClosed(opsToClone)) {
      return failure();
    }

    // create generic op using opsToClone as the body. Rewrite load op
    // parameters to be generic op block args
    // TODO
    llvm::errs() << "TODO\n";

    return failure();
  }
};

} // namespace

struct TritonCPUTileAndFusePass
    : public impl::TritonCPUTileAndFuseBase<TritonCPUTileAndFusePass> {
  using TritonCPUTileAndFuseBase::TritonCPUTileAndFuseBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;

    patterns.add<WrapElementwiseChain>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
