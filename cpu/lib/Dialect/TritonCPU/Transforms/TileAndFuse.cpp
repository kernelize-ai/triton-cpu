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

    LLVM_DEBUG({
      for (auto op : opsToClone) {
        DBGS() << "op to clone: " << *op << "\n";
      }
    });

    if (!isClosed(opsToClone))
      return failure();

    auto tensorTy = getCommonType(opsToClone);
    if (!tensorTy)
      return failure();
    auto encoding = dyn_cast<gpu::BlockedEncodingAttr>(tensorTy->getEncoding());
    if (!encoding)
      return failure();

    LDBG("Creating generic op with common type " << tensorTy);

    // create generic op using opsToClone as the body. Rewrite load op
    // parameters to be generic op block args

    // the load op arguments will be forwarded through the generic in order of
    // load op appearance
    // TODO: we should do this as a set vector to avoid duplication
    SmallVector<Value> genericOpInputs;
    for (auto op : opsToClone) {
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        genericOpInputs.push_back(loadOp.getPtr());
        if (loadOp.getMask())
          genericOpInputs.push_back(loadOp.getMask());
        if (loadOp.getOther())
          genericOpInputs.push_back(loadOp.getOther());
      }
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
        genericOpInputs.push_back(storeOp.getPtr());
        // value must be part of the generic region or the generic is not closed
        if (storeOp.getMask())
          genericOpInputs.push_back(storeOp.getMask());
      }
    }

    SmallVector<Value> genericOpParams; // none for now

    auto shape = tensorTy->getShape();
    SmallVector<int32_t> shapeVec(shape.begin(), shape.end());
    auto sizePerThread = encoding.getSizePerThread();
    SmallVector<int32_t> sizePerThreadVec(sizePerThread.begin(),
                                          sizePerThread.end());

    // Use sizePerThread to set the tile size for now. In the future, we could
    // determine this dynamically (say if the number of contiguous entries in
    // the tensor was 1, but we wanted to scatter load and vectorize)
    auto updateTensorType = [&](RankedTensorType oldType) -> RankedTensorType {
      return RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              sizePerThread, [](int32_t s) { return int64_t(s); })),
          oldType.getElementType(), encoding);
    };

    auto generic =
        cpu::GenericOp::create(rewriter, loc, genericOpInputs, genericOpParams,
                               shapeVec, sizePerThreadVec);
    rewriter.createBlock(&generic->getRegion(0));
    Block *entry = &generic->getRegion(0).front();
    SmallVector<BlockArgument> args;
    args.reserve(genericOpInputs.size());
    for (Value v : genericOpInputs) {
      // keep the encoding but replace the block shape with the generic tile
      // shape
      auto existingType = cast<RankedTensorType>(v.getType());
      args.push_back(
          entry->addArgument(updateTensorType(existingType), v.getLoc()));
    }
    IRMapping mapping;
    for (auto [v, a] : llvm::zip(genericOpInputs, args)) {
      mapping.map(v, a);
    }

    rewriter.setInsertionPointToStart(entry);
    for (auto op : llvm::reverse(opsToClone)) {
      auto newOp = rewriter.clone(*op, mapping);
      for (auto result : newOp->getResults()) {
        result.setType(
            updateTensorType(cast<RankedTensorType>(result.getType())));
      }
      mapping.map(op->getResults(), newOp->getResults());
    }

    cpu::YieldOp::create(rewriter, loc);

    for (auto op : opsToClone) {
      rewriter.eraseOp(op);
    }
    return success();
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
