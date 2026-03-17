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
    auto opIsElementwise = [&](Operation *defOp) -> bool {
      if (isa<arith::ArithDialect, math::MathDialect>(defOp->getDialect()) &&
          defOp->hasTrait<OpTrait::Elementwise>())
        return true;
      else if (isa<triton::AddPtrOp>(defOp))
        return true;
      return false;
    };

    // allow elementwise ops and push their operands to the queue
    if (opIsElementwise(defOp)) {
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

struct WrapStores : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    if (storeOp->getParentOfType<cpu::GenericOp>())
      return failure();

    auto value = storeOp.getValue();
    auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
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

    auto updateTensorType = [&](Type oldType) -> Type {
      auto oldTensorType = dyn_cast<RankedTensorType>(oldType);
      if (!oldTensorType)
        return oldType;
      return RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              vectorShape, [](int32_t s) { return int64_t(s); })),
          oldTensorType.getElementType(), oldTensorType.getEncoding());
    };
    ;

    SmallVector<Value> ins;
    for (auto operand : storeOp->getOperands()) {
      ins.push_back(operand);
    }

    auto generic = cpu::GenericOp::create(
        rewriter, loc, /*resultTypes= */ TypeRange{}, ins,
        /*params=*/ValueRange{}, blockShape, vectorShape);

    Block *body = rewriter.createBlock(&generic.getBody());
    IRMapping bodyMapping;
    for (Value v : ins) {
      Type argTy = v.getType();
      if (auto tt = dyn_cast<RankedTensorType>(argTy))
        argTy = updateTensorType(tt);
      bodyMapping.map(v, body->addArgument(argTy, v.getLoc()));
    }

    rewriter.setInsertionPointToStart(body);

    rewriter.clone(*storeOp, bodyMapping);
    cpu::YieldOp::create(rewriter, loc, /*values=*/ValueRange{});

    rewriter.replaceOp(storeOp, generic.getResults());
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

    return success();
  }
};

struct Fuse : public mlir::OpRewritePattern<triton::cpu::GenericOp> {
  using OpRewritePattern<triton::cpu::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
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

    for (Operation *op : llvm::reverse(*opsToClone)) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }

    return success();
  }
};

// Returns true if defOp can be cloned into a generic body during fusion.
// Reduction generics (cpu::GenericOp with scalar results) are not fusible —
// their scalar outputs become params of the consumer, not tiled ins.
static bool isFusible(Operation *defOp) {
  if (!defOp)
    return false;
  if ((isa<arith::ArithDialect, math::MathDialect>(defOp->getDialect())) &&
      defOp->hasTrait<OpTrait::Elementwise>())
    return true;
  if (isa<triton::LoadOp>(defOp))
    return true;
  // tt.splat broadcasts a scalar to a tensor; the scalar input becomes a param.
  if (isa<triton::SplatOp>(defOp))
    return true;
  return false;
}

// Fuse ops that produce the ins values of genericOp into its body.
//
// For each ins value, if its defining op is fusible (elementwise, load, splat),
// clone it into the body (transitively). Tensor operands of the fused ops
// become new ins (tiled); scalar operands become new params (untiled).
// Block args whose values are now produced inside the body are removed from
// ins and the body arg list. Original ops are erased if they have no remaining
// users after fusion (loads are kept if still used by other generics).
static void fuseInputs(IRRewriter &rewriter, cpu::GenericOp genericOp) {
  LDBG("fuseInputs: " << genericOp);

  auto sizePerThreadVec = genericOp.getVectorShape();
  auto updateTensorType = [&](Type t) -> Type {
    auto tt = dyn_cast<RankedTensorType>(t);
    if (!tt)
      return t;
    return RankedTensorType::get(
        llvm::to_vector(llvm::map_range(sizePerThreadVec,
                                        [](int32_t s) { return int64_t(s); })),
        tt.getElementType(), tt.getEncoding());
  };

  // Collect fusible ops reachable from the current ins values, in
  // def-before-use order (SetVector preserves insertion order; we walk
  // backwards so reversing gives topo order for cloning).
  SetVector<Operation *> opsToFuse;
  SmallVector<Value> worklist(genericOp.getIns().begin(),
                              genericOp.getIns().end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !isFusible(defOp) || opsToFuse.contains(defOp))
      continue;
    opsToFuse.insert(defOp);
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  if (opsToFuse.empty())
    return;

  // Build lookup: existing ins value → block arg index (before any changes).
  DenseMap<Value, unsigned> insToArgIdx;
  for (auto [idx, v] : llvm::enumerate(genericOp.getIns()))
    insToArgIdx[v] = idx;

  SmallVector<Value> newIns(genericOp.getIns().begin(),
                            genericOp.getIns().end());
  SmallVector<Value> newParams(genericOp.getParams().begin(),
                               genericOp.getParams().end());
  SmallVector<unsigned> insIdxToRemove;

  Block *body = &genericOp.getBody().front();
  // Insert cloned ops before the first existing body op.
  rewriter.setInsertionPointToStart(body);

  IRMapping mapping;
  // Clone in topological (def-before-use) order.
  for (Operation *op : llvm::reverse(opsToFuse)) {
    for (Value operand : op->getOperands()) {
      if (mapping.contains(operand))
        continue;

      newIns.push_back(operand);
      mapping.map(operand,
                  body->addArgument(updateTensorType(operand.getType()),
                                    operand.getLoc()));
    }

    Operation *newOp = rewriter.clone(*op, mapping);
    for (auto [origResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      newResult.setType(updateTensorType(newResult.getType()));
      // If this result was previously an ins, replace its block arg with the
      // newly cloned result and mark the arg for removal.
      if (auto it = insToArgIdx.find(origResult); it != insToArgIdx.end()) {
        body->getArgument(it->second).replaceAllUsesWith(newResult);
        insIdxToRemove.push_back(it->second);
      }
    }
    mapping.map(op->getResults(), newOp->getResults());
  }

  // Remove replaced ins entries and their block args (reverse order for index
  // stability).
  llvm::sort(insIdxToRemove);
  for (unsigned idx : llvm::reverse(insIdxToRemove)) {
    body->eraseArgument(idx);
    newIns.erase(newIns.begin() + idx);
  }

  rewriter.modifyOpInPlace(genericOp, [&]() {
    genericOp.getInsMutable().assign(newIns);
    genericOp.getParamsMutable().assign(newParams);
  });

  // Erase original ops that have no remaining users. Loads may still be used
  // by other generics (re-load semantics), so they are erased only if empty.
  for (Operation *op : llvm::reverse(opsToFuse)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }
}

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
    patterns.add<WrapStores>(context, benefitDefault);
    patterns.add<WrapReduceOp>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    // Step 2: Fuse elementwise ops and loads into each generic, bottom-up.
    // Collect once before fusion (the worklist is stable; new generics are not
    // created during fusion, only existing ops are cloned / erased).
    SmallVector<cpu::GenericOp> worklist;
    m.walk([&](cpu::GenericOp op) { worklist.push_back(op); });
    IRRewriter rewriter(context);
    for (cpu::GenericOp genericOp : llvm::reverse(worklist)) {
      fuseInputs(rewriter, genericOp);
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
