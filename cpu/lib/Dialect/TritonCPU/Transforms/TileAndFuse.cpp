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

static std::optional<RankedTensorType>
getCommonType(const llvm::SetVector<Operation *> &ops) {
  // assumes the op result encodings match the inputs
  auto storeOp = cast<triton::StoreOp>(*ops.begin());
  RankedTensorType tensorTy =
      cast<RankedTensorType>(storeOp.getValue().getType());
  for (Operation *op : ops) {
    for (auto result : op->getResults()) {
      auto resultType = cast<RankedTensorType>(result.getType());
      if (resultType != tensorTy) {
        return std::nullopt;
      }
    }
  }
  return tensorTy;
}

struct WrapElementwiseChain : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

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
    if (failed)
      return failure();

    for (auto op : opsToClone) {
      llvm::errs() << "op to clone: " << *op << "\n";
    }

    if (!isClosed(opsToClone))
      return failure();

    auto tensorTy = getCommonType(opsToClone);
    if (!tensorTy)
      return failure();
    auto encoding = dyn_cast<gpu::BlockedEncodingAttr>(tensorTy->getEncoding());
    if (!encoding)
      return failure();

    llvm::errs() << "common type = " << tensorTy << "\n";

    // create generic op using opsToClone as the body. Rewrite load op
    // parameters to be generic op block args

    // the load op arguments will be forwarded through the generic in order
    SmallVector<Value> genericOpInputs;
    for (auto op : opsToClone) {
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        genericOpInputs.push_back(loadOp.getPtr());
        if (loadOp.getMask())
          genericOpInputs.push_back(loadOp.getMask());
        if (loadOp.getOther())
          genericOpInputs.push_back(loadOp.getOther());
      }
    }

    SmallVector<Value> genericOpOutputs; // none for now
    SmallVector<Value> genericOpParams;  // none for now

    auto shape = tensorTy->getShape();
    SmallVector<int32_t> shapeVec(shape.begin(), shape.end());
    auto sizePerThread = encoding.getSizePerThread();
    SmallVector<int32_t> sizePerThreadVec(sizePerThread.begin(),
                                          sizePerThread.end());

    auto generic =
        cpu::GenericOp::create(rewriter, loc, genericOpInputs, genericOpOutputs,
                               genericOpParams, shapeVec, sizePerThreadVec);
    llvm::errs() << "created generic: " << generic << "\n";
    // TODO: populate generic body

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
