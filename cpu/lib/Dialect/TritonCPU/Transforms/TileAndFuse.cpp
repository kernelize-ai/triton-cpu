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
  for (Operation *op : ops) {
    for (Value r : op->getResults()) {
      for (Operation *user : r.getUsers()) {
        if (!ops.contains(user))
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

std::optional<llvm::SetVector<Operation *>>
getElementwiseChain(Operation *initialOp, Value start) {
  SetVector<Operation *> ops;
  ops.insert(initialOp);

  llvm::SmallVector<Value> queue;
  queue.push_back(start);

  while (!queue.empty()) {
    auto v = queue.pop_back_val();
    auto defOp = v.getDefiningOp();

    if (!defOp)
      continue;

    // allow load operations and elementwise operations in ttc.generic
    if (auto loadOp = dyn_cast<triton::LoadOp>(defOp)) {
      // load ops terminate the chain
      ops.insert(defOp);
      continue;
    }
    // allow elementwise ops and push their operands to the queue
    if (isa<arith::ArithDialect, math::MathDialect>(defOp->getDialect()) &&
        defOp->hasTrait<OpTrait::Elementwise>()) {
      ops.insert(defOp);
    } else {
      continue;
    }

    for (auto operand : defOp->getOperands()) {
      queue.push_back(operand);
    }
  }
  LLVM_DEBUG({
    DBGS() << "getElementwiseChain for " << start << " found ops:\n";
    for (Operation *op : llvm::reverse(ops)) {
      DBGS() << "  " << *op << "\n";
    }
  });

  return ops;
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

    auto opsToClone = getElementwiseChain(storeOp, storeOp.getValue());
    if (!opsToClone)
      return failure();

    if (!isClosed(*opsToClone))
      return failure();

    auto tensorTy = getCommonType(*opsToClone);
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
    for (auto op : *opsToClone) {
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
              sizePerThreadVec, [](int32_t s) { return int64_t(s); })),
          oldType.getElementType(), encoding);
    };

    auto generic = cpu::GenericOp::create(
        rewriter, loc, /*results=*/TypeRange{}, genericOpInputs,
        genericOpParams, shapeVec, sizePerThreadVec);
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
    for (auto op : llvm::reverse(*opsToClone)) {
      auto newOp = rewriter.clone(*op, mapping);
      for (auto result : newOp->getResults()) {
        result.setType(
            updateTensorType(cast<RankedTensorType>(result.getType())));
      }
      mapping.map(op->getResults(), newOp->getResults());
    }

    cpu::YieldOp::create(rewriter, loc, /*values=*/ValueRange{});

    // Empty combiners region — no scalar results.
    // rewriter.createBlock(&generic->getRegion(1));

    for (auto op : *opsToClone) {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct WrapReduceOp : public mlir::OpRewritePattern<triton::ReduceOp> {
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReduceOp reduceOp,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    // Don't re-wrap reductions already inside a ttc.generic.
    if (reduceOp->getParentOfType<cpu::GenericOp>())
      return failure();

#if 1
    // Wrap scalar reductions in ttc.generic. Take the elementwise chain as
    // input.
    auto reduceResult = reduceOp.getResult();
    if (reduceResult.size() != 1)
      return failure();
    if (isa<RankedTensorType>(reduceResult[0].getType()))
      return failure();

    auto srcs = reduceOp.getSrcs();
    if (srcs.size() != 1)
      return failure();

    auto tensorTy = dyn_cast<RankedTensorType>(srcs[0].getType());
    if (!tensorTy)
      return failure();
    auto encoding = dyn_cast<gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
    if (!encoding)
      return failure();

    auto shape = tensorTy.getShape();
    SmallVector<int32_t> blockShape(shape.begin(), shape.end());
    auto sizePerThread = encoding.getSizePerThread();
    SmallVector<int32_t> vectorShape(sizePerThread.begin(),
                                     sizePerThread.end());

    auto updateTensorType = [&](RankedTensorType t) -> RankedTensorType {
      return RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              vectorShape, [](int32_t s) { return int64_t(s); })),
          t.getElementType(), encoding);
    };

    SmallVector<Value> ins;
    ins.push_back(srcs[0]);

    SmallVector<Type> resultTypes(reduceOp.getResultTypes().begin(),
                                  reduceOp.getResultTypes().end());

    LDBG("Creating reduction generic op, result types: " << resultTypes.size());

    auto generic = cpu::GenericOp::create(rewriter, loc, resultTypes, ins,
                                          /*params=*/ValueRange{}, blockShape,
                                          vectorShape);

    Block *body = rewriter.createBlock(&generic.getBody());
    IRMapping bodyMapping;
    for (Value v : ins) {
      Type argTy = v.getType();
      if (auto tt = dyn_cast<RankedTensorType>(argTy))
        argTy = updateTensorType(tt);
      bodyMapping.map(v, body->addArgument(argTy, v.getLoc()));
    }

    rewriter.setInsertionPointToStart(body);

    // Clone the reduce — it now operates on the tile-sized tensor.
    auto *newReduce = rewriter.clone(*reduceOp, bodyMapping);

    SmallVector<Value> partials(newReduce->getResults().begin(),
                                newReduce->getResults().end());
    cpu::YieldOp::create(rewriter, loc, partials);

    // Each block takes (acc, partial) and applies the same combining op as the
    // tt.reduce combiner, replacing tt.reduce.return with ttc.yield.
    Region &combiners = generic.getCombiners();
    Region &reduceCombiner = reduceOp.getCombineOp();
    Block &srcBlock = reduceCombiner.front();

    for (Type resultTy : resultTypes) {
      Block *combBlock = rewriter.createBlock(&combiners);
      Value acc = combBlock->addArgument(resultTy, loc);
      Value partial = combBlock->addArgument(resultTy, loc);

      IRMapping combMapping;
      // The reduce combiner block has args [lhs..., rhs...] interleaved per
      // src. With a single src the layout is simply [lhs, rhs].
      combMapping.map(srcBlock.getArgument(0), acc);
      combMapping.map(srcBlock.getArgument(1), partial);

      rewriter.setInsertionPointToStart(combBlock);
      for (Operation &op : srcBlock.without_terminator())
        rewriter.clone(op, combMapping);

      // Replace tt.reduce.return with ttc.yield.
      SmallVector<Value> yieldVals;
      for (Value rv : srcBlock.getTerminator()->getOperands())
        yieldVals.push_back(combMapping.lookupOrDefault(rv));
      cpu::YieldOp::create(rewriter, loc, yieldVals);
    }

    rewriter.replaceOp(reduceOp, generic.getResults());

#else

    // Only handle the simple case: single source tensor from a tt.load.
    // Loads are duplicated into the generic region; the original load is erased
    // only if it has no remaining users
    auto srcs = reduceOp.getSrcs();
    if (srcs.size() != 1)
      return failure();

    auto opsToClone = getElementwiseChain(reduceOp, srcs[0]);
    if (!opsToClone || opsToClone->empty())
      return failure();

    // we want to build the closed set of tensors but allow passing scalars (and
    // arguments to loads).how do we differentiate?
    //
#if 1
    if (!isClosed(*opsToClone))
      return failure();
#endif

    llvm::errs() << "srcs[0] defining op: " << *srcs[0].getDefiningOp() << "\n";
    auto loadOp = dyn_cast_or_null<triton::LoadOp>(srcs[0].getDefiningOp());
    if (!loadOp)
      return failure();

    auto tensorTy = cast<RankedTensorType>(loadOp.getResult().getType());
    // only support blocked encodings
    auto encoding = dyn_cast<gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
    if (!encoding)
      return failure();

    auto shape = tensorTy.getShape();
    SmallVector<int32_t> blockShape(shape.begin(), shape.end());
    auto sizePerThread = encoding.getSizePerThread();
    SmallVector<int32_t> vectorShape(sizePerThread.begin(),
                                     sizePerThread.end());

    auto updateTensorType = [&](RankedTensorType t) -> RankedTensorType {
      return RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              vectorShape, [](int32_t s) { return int64_t(s); })),
          t.getElementType(), encoding);
    };

    // Collect ins operands from the load.
    SmallVector<Value> ins;
    ins.push_back(loadOp.getPtr());
    if (loadOp.getMask())
      ins.push_back(loadOp.getMask());
    if (loadOp.getOther())
      ins.push_back(loadOp.getOther());

    SmallVector<Type> resultTypes(reduceOp.getResultTypes().begin(),
                                  reduceOp.getResultTypes().end());

    LDBG("Creating reduction generic op, result types: " << resultTypes.size());

    auto generic = cpu::GenericOp::create(rewriter, loc, resultTypes, ins,
                                          /*params=*/ValueRange{}, blockShape,
                                          vectorShape);

    // --- Body region: load one tile, reduce it, yield the partial result. ---
    Block *body = rewriter.createBlock(&generic.getBody());
    IRMapping bodyMapping;
    for (Value v : ins) {
      Type argTy = v.getType();
      if (auto tt = dyn_cast<RankedTensorType>(argTy))
        argTy = updateTensorType(tt);
      bodyMapping.map(v, body->addArgument(argTy, v.getLoc()));
    }

    rewriter.setInsertionPointToStart(body);

    // Clone the load with the tiled type.
    auto *newLoad = rewriter.clone(*loadOp, bodyMapping);
    for (Value r : newLoad->getResults())
      if (auto tt = dyn_cast<RankedTensorType>(r.getType()))
        r.setType(updateTensorType(tt));
    bodyMapping.map(loadOp->getResults(), newLoad->getResults());

    // Clone the reduce — it now operates on the tile-sized tensor.
    auto *newReduce = rewriter.clone(*reduceOp, bodyMapping);

    SmallVector<Value> partials(newReduce->getResults().begin(),
                                newReduce->getResults().end());
    cpu::YieldOp::create(rewriter, loc, partials);

    // --- Combiners region: one block per scalar result. ---
    // Each block takes (acc, partial) and applies the same combining op as the
    // tt.reduce combiner, replacing tt.reduce.return with ttc.yield.
    Region &combiners = generic.getCombiners();
    Region &reduceCombiner = reduceOp.getCombineOp();
    Block &srcBlock = reduceCombiner.front();

    for (Type resultTy : resultTypes) {
      Block *combBlock = rewriter.createBlock(&combiners);
      Value acc = combBlock->addArgument(resultTy, loc);
      Value partial = combBlock->addArgument(resultTy, loc);

      IRMapping combMapping;
      // The reduce combiner block has args [lhs..., rhs...] interleaved per
      // src. With a single src the layout is simply [lhs, rhs].
      combMapping.map(srcBlock.getArgument(0), acc);
      combMapping.map(srcBlock.getArgument(1), partial);

      rewriter.setInsertionPointToStart(combBlock);
      for (Operation &op : srcBlock.without_terminator())
        rewriter.clone(op, combMapping);

      // Replace tt.reduce.return with ttc.yield.
      SmallVector<Value> yieldVals;
      for (Value rv : srcBlock.getTerminator()->getOperands())
        yieldVals.push_back(combMapping.lookupOrDefault(rv));
      cpu::YieldOp::create(rewriter, loc, yieldVals);
    }

    rewriter.replaceOp(reduceOp, generic.getResults());

    if (loadOp->use_empty())
      rewriter.eraseOp(loadOp);
#endif
    return success();
  }
};

struct Fuse : public mlir::OpRewritePattern<triton::cpu::GenericOp> {
  using OpRewritePattern<triton::cpu::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::errs() << "fusing " << genericOp << "\n";
    // get the elementwise chain starting from this op and working backwards

    // TODO: this isn't compatible with the conversion of store op chains to
    // generic (though it might work by accident since ins[0] would be the load
    // value currently)
    auto opsToClone = getElementwiseChain(genericOp.getIns()[0].getDefiningOp(),
                                          genericOp.getIns()[0]);
    if (!opsToClone || opsToClone->empty())
      return failure();

    if (!isa<triton::LoadOp>(opsToClone->back()))
      return failure();

    auto sizePerThreadVec = genericOp.getVectorShape();
    auto updateTensorType = [&](Type oldType) -> Type {
      auto oldTensorType = dyn_cast<RankedTensorType>(oldType);
      if (!oldTensorType)
        return oldType;
      return RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              sizePerThreadVec, [](int32_t s) { return int64_t(s); })),
          oldTensorType.getElementType(), oldTensorType.getEncoding());
    };

    // For each op in the elementwise chain, clone it into the generic body.
    // - Operands defined outside the generic become new ins (new block args).
    // - If an op's result was already an ins to the generic, replace the
    //   corresponding block arg with the cloned result and drop it from ins.

    // Build a lookup: existing ins value → index in the ins list / block arg.
    DenseMap<Value, unsigned> insToArgIdx;
    for (auto [idx, v] : llvm::enumerate(genericOp.getIns()))
      insToArgIdx[v] = idx;

    SmallVector<Value> genericOpInputs = genericOp.getIns();
    SmallVector<unsigned> argIdxToRemove;
    Block *body = &genericOp.getBody().front();
    rewriter.setInsertionPointToStart(body);
    IRMapping mapping;
    for (Operation *op : llvm::reverse(*opsToClone)) {
      for (Value operand : op->getOperands()) {
        if (!mapping.contains(operand)) {
          // Operand is defined outside the generic — add as a new ins.
          genericOpInputs.push_back(operand);
          auto arg = body->addArgument(updateTensorType(operand.getType()),
                                       operand.getLoc());
          mapping.map(operand, arg);
        }
      }
      auto newOp = rewriter.clone(*op, mapping);

      for (auto [origResult, newResult] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        newResult.setType(updateTensorType(newResult.getType()));
        // If this result was previously an ins, the body already uses the
        // corresponding block arg. Replace those uses with the cloned result
        // and mark the block arg for removal.
        if (auto it = insToArgIdx.find(origResult); it != insToArgIdx.end()) {
          body->getArgument(it->second).replaceAllUsesWith(newResult);
          argIdxToRemove.push_back(it->second);
        }
      }
      mapping.map(op->getResults(), newOp->getResults());
    }

    // Remove block args and ins entries that are now produced inside the body.
    // Process in reverse index order so earlier indices stay stable.
    llvm::sort(argIdxToRemove);
    for (unsigned idx : llvm::reverse(argIdxToRemove)) {
      body->eraseArgument(idx);
      genericOpInputs.erase(genericOpInputs.begin() + idx);
    }

    // Update the generic op's ins operand list.
    rewriter.modifyOpInPlace(genericOp, [&]() {
      genericOp.getInsMutable().assign(genericOpInputs);
    });

    llvm::errs() << "updated generic: " << genericOp << "\n";

    for (Operation *op : llvm::reverse(*opsToClone)) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
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

    // Step 1: Create the generic ops
    patterns.add<WrapElementwiseChain>(context, benefitDefault);
    patterns.add<WrapReduceOp>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    // Step 2: Fuse generics + elementwise chains
    mlir::RewritePatternSet fusionPatterns(context);

    fusionPatterns.add<Fuse>(context, benefitDefault);
    if (applyPatternsGreedily(m, std::move(fusionPatterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
