#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Replace the shape of a RankedTensorType with vectorShape, preserving element
// type and encoding. Non-tensor types are returned unchanged.
static Type updateTensorType(Type t, ArrayRef<int32_t> vectorShape) {
  auto tensorType = dyn_cast<RankedTensorType>(t);
  if (!tensorType)
    return t;
  assert(tensorType.getShape().size() == vectorShape.size() &&
         "expected tensor shape and vector shape to be the same during update");
  return RankedTensorType::get(
      llvm::to_vector(
          llvm::map_range(vectorShape, [](int32_t s) { return int64_t(s); })),
      tensorType.getElementType(), tensorType.getEncoding());
}

// Extract blockShape (full tensor shape) and vectorShape (sizePerThread) from
// a tensor type with BlockedEncoding.
static std::pair<SmallVector<int32_t>, SmallVector<int32_t>>
getBlockAndVectorShapes(RankedTensorType tensorTy,
                        gpu::BlockedEncodingAttr encoding) {
  auto shape = tensorTy.getShape();
  SmallVector<int32_t> blockShape(shape.begin(), shape.end());
  auto sizePerThread = encoding.getSizePerThread();
  SmallVector<int32_t> vectorShape(sizePerThread.begin(), sizePerThread.end());
  return {blockShape, vectorShape};
}

// Create the body block of a GenericOp, adding one block arg per ins value
// with tensor types replaced to the vector (chunk) shape. Populates `mapping`
// with ins value → block arg entries and sets the insertion point to the start
// of the block. Returns the new block.
static Block *initGenericBody(OpBuilder &rewriter, cpu::GenericOp generic,
                              ArrayRef<Value> ins,
                              ArrayRef<int32_t> vectorShape,
                              IRMapping &mapping) {
  Block *body = rewriter.createBlock(&generic.getBody());
  body->addArgument(rewriter.getI32Type(), generic.getLoc()); // chunk offset
  for (Value v : ins) {
    Type argTy = updateTensorType(v.getType(), vectorShape);
    mapping.map(v, body->addArgument(argTy, v.getLoc()));
  }
  rewriter.setInsertionPointToStart(body);
  return body;
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

    auto [blockShape, vectorShape] =
        getBlockAndVectorShapes(tensorTy, encoding);

    SmallVector<Value> ins(storeOp->getOperands().begin(),
                           storeOp->getOperands().end());

    auto generic = cpu::GenericOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, ins,
        /*params=*/ValueRange{}, blockShape, vectorShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, vectorShape, bodyMapping);

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

    // if the value being reduced is used elsewhere ttc.generic can materialize
    // the tensor
    // TODO: parametrize?
    const bool allowTensorMaterializationFlag = true;
    const bool srcIsLoad = isa<LoadOp>(srcs[0].getDefiningOp());
    const bool allowTensorMaterialization =
        allowTensorMaterializationFlag && !srcIsLoad;
    const bool srcUsedElsewhere =
        allowTensorMaterialization &&
        llvm::any_of(srcs[0].getUsers(), [&](Operation *user) {
          return user != reduceOp.getOperation(); // or just != reduceOp?
        });

    auto [blockShape, vectorShape] =
        getBlockAndVectorShapes(tensorTy, encoding);

    SmallVector<Value> ins = {srcs[0]};
    SmallVector<Type> resultTypes(reduceOp.getResultTypes().begin(),
                                  reduceOp.getResultTypes().end());
    if (srcUsedElsewhere)
      resultTypes.push_back(tensorTy);

    LDBG("Creating reduction generic op, result types: " << resultTypes.size());

    auto generic = cpu::GenericOp::create(rewriter, loc, resultTypes, ins,
                                          /*params=*/ValueRange{}, blockShape,
                                          vectorShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, vectorShape, bodyMapping);

    // Clone the reduce — it now operates on the tile-sized tensor.
    auto *newReduce = rewriter.clone(*reduceOp, bodyMapping);

    SmallVector<Value> partials(newReduce->getResults().begin(),
                                newReduce->getResults().end());
    if (srcUsedElsewhere)
      partials.push_back(bodyMapping.lookup(srcs[0]));
    cpu::YieldOp::create(rewriter, loc, partials);

    // Each block takes (acc, partial) and applies the same combining op as the
    // tt.reduce combiner, replacing tt.reduce.return with ttc.yield.
    Region &combiners = generic.getCombiners();
    Region &reduceCombiner = reduceOp.getCombineOp();
    Block &srcBlock = reduceCombiner.front();

    // Only scalar (reduction) results need combiner blocks. The tensor result
    // (if present) uses scatter semantics and has no combiner.
    for (Type resultTy : reduceOp.getResultTypes()) {
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

    // Replace uses of srcs[0] outside the generic with the materialized tensor
    // result. Must go through the rewriter and exclude the generic's own
    // operand so the generic still receives the original value as its ins.
    if (srcUsedElsewhere)
      rewriter.replaceUsesWithIf(
          srcs[0], generic.getResult(1), [&](OpOperand &use) {
            return use.getOwner() != generic.getOperation();
          });

    // Replace reduceOp with the scalar reduction result (generic result 0).
    rewriter.replaceOp(reduceOp, generic.getResult(0));

    return success();
  }
};

// Convert dot op nested in scf::ForOp loop to ttc.generic
struct WrapKLoopWithDotOp : public mlir::OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  struct KLoopDotInfo {
    triton::DotOp dotOp;
    unsigned cIterArgIdx;
    SmallVector<std::pair<unsigned, Value>>
        ptrArgs; // loop carried ptr values for A and B tile
  };

  static std::optional<KLoopDotInfo> matchKLoopWithDot(scf::ForOp forOp) {
    if (forOp->getParentOfType<cpu::GenericOp>())
      return std::nullopt;

    triton::DotOp dotOp;
    for (auto &op : *forOp.getBody()) {
      if (auto d = dyn_cast<triton::DotOp>(&op)) {
        if (dotOp)
          return std::nullopt; // TODO: handle multiple dots?
        dotOp = d;
      }
    }
    if (!dotOp)
      return std::nullopt;

    // Dot result must have BlockedEncoding so we can derive
    // blockShape/vectorShape.
    auto resultTy = dyn_cast<RankedTensorType>(dotOp.getType());
    if (!resultTy || !isa<gpu::BlockedEncodingAttr>(resultTy.getEncoding()))
      return std::nullopt;

    // C operand must be an iter_arg of this loop (not a constant or external
    // value).
    auto cArg = dyn_cast<BlockArgument>(dotOp.getC());
    if (!cArg || cArg.getOwner() != forOp.getBody())
      return std::nullopt;
    unsigned numIVs = forOp.getNumInductionVars();
    unsigned cArgNum = cArg.getArgNumber();
    if (cArgNum < numIVs)
      return std::nullopt;
    unsigned cIterArgIdx = cArgNum - numIVs;

    // C's init value must be a zero-splat constant — we reconstruct it at tile
    // size inside the generic body.
    auto cInitOp =
        forOp.getInitArgs()[cIterArgIdx].getDefiningOp<arith::ConstantOp>();
    if (!cInitOp)
      return std::nullopt;
    auto dense = dyn_cast<DenseElementsAttr>(cInitOp.getValue());
    if (!dense || !dense.isSplat())
      return std::nullopt;
    if (auto fp = dyn_cast<FloatAttr>(dense.getSplatValue<Attribute>()))
      if (!fp.getValue().isZero())
        return std::nullopt;

    // The yield must carry the dot result back as the updated C value.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yieldOp.getOperands()[cIterArgIdx] != dotOp.getResult())
      return std::nullopt;

    // Find pointer iter_args that are advanced by tt.addptr each iteration.
    SmallVector<std::pair<unsigned, Value>> ptrArgs;
    for (unsigned i = 0, e = forOp.getInitArgs().size(); i < e; ++i) {
      if (i == cIterArgIdx)
        continue;
      Value iterArgVal = forOp.getBody()->getArgument(numIVs + i);
      auto addPtrOp =
          yieldOp.getOperands()[i].getDefiningOp<triton::AddPtrOp>();
      if (!addPtrOp || addPtrOp.getPtr() != iterArgVal)
        continue;
      ptrArgs.push_back({i, addPtrOp.getOffset()});
    }
    if (ptrArgs.empty())
      return std::nullopt;

    return KLoopDotInfo{dotOp, cIterArgIdx, ptrArgs};
  }

  mlir::LogicalResult
  matchAndRewrite(scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {

#if 1
    auto info = matchKLoopWithDot(forOp);
    if (!info)
      return failure();

    Location loc = forOp.getLoc();
    triton::DotOp dotOp = info->dotOp;
    auto resultTy = cast<RankedTensorType>(dotOp.getType());
    auto encoding = cast<gpu::BlockedEncodingAttr>(resultTy.getEncoding());

    // TODO: this vector shape may be too small?
    auto [blockShape, vectorShape] =
        getBlockAndVectorShapes(resultTy, encoding);

    SmallVector<Value> ins;

    // add iter arg initializations to generic operand list first so we can keep
    // track of them later
    for (auto [argIdx, stride] : info->ptrArgs) {
      ins.push_back(forOp.getInitArgs()[argIdx]); // initial value
      ins.push_back(stride);                      // value to advance by
    }
    for (auto &op : *forOp.getBody()) {
      if (isa<scf::YieldOp>(op))
        continue;
      for (Value operand : op.getOperands()) {
        // Skip values defined inside the loop body (intermediate results,
        // the induction variable, and iter_args).
        if (operand.getParentBlock() == forOp.getBody())
          continue;
        ins.push_back(operand);
      }
    }
    ins.push_back(forOp.getLowerBound());
    ins.push_back(forOp.getUpperBound());
    ins.push_back(forOp.getStep());

    auto generic = cpu::GenericOp::create(rewriter, loc, TypeRange{resultTy},
                                          ins, /*params= */ ValueRange{},
                                          blockShape, vectorShape);

    IRMapping outerMapping;
    initGenericBody(rewriter, generic, ins, vectorShape, outerMapping);

    // setup arguments for the K loop
    SmallVector<Value> tilePtrInits, tilePtrStrides;
    for (auto [argIdx, stride] : info->ptrArgs) {
      tilePtrInits.push_back(outerMapping.lookup(forOp.getInitArgs()[argIdx]));
      tilePtrStrides.push_back(outerMapping.lookup(stride));
    }
    Value kLb = outerMapping.lookup(forOp.getLowerBound());
    Value kUb = outerMapping.lookup(forOp.getUpperBound());
    Value kStep = outerMapping.lookup(forOp.getStep());

    // copy the accumulator - TODO support slicing non-constant accumulators
    auto tileAccTy =
        cast<RankedTensorType>(updateTensorType(resultTy, vectorShape));
    auto cInitConst = forOp.getInitArgs()[info->cIterArgIdx]
                          .getDefiningOp<arith::ConstantOp>();
    auto origSplat = cast<DenseElementsAttr>(cInitConst.getValue());
    Value tileAcc = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(tileAccTy,
                               *origSplat.getValues<Attribute>().begin()));

    // clone the K loop
    unsigned numIVs = forOp.getNumInductionVars();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    SmallVector<Value> kIterInits = {tileAcc};
    kIterInits.append(tilePtrInits);
    auto kFor = scf::ForOp::create(rewriter, loc, kLb, kUb, kStep, kIterInits);

    IRMapping loopMapping;
    // map generic operand arguments to block arguments
    for (auto [i, v] : llvm::enumerate(ins))
      loopMapping.map(v, outerMapping.lookup(v));

    // map existing loop induction var to new induction var and existing loop
    // carried accumulator value to new loop region argument
    loopMapping.map(forOp.getInductionVar(), kFor.getInductionVar());
    loopMapping.map(forOp.getBody()->getArgument(numIVs + info->cIterArgIdx),
                    kFor.getRegionIterArgs()[0]);

    // map existing loop body arguments to cloned loop body arguments
    for (auto [pairIdx, p] : llvm::enumerate(info->ptrArgs)) {
      auto [argIdx, stride] = p;
      loopMapping.map(forOp.getBody()->getArgument(numIVs + argIdx),
                      kFor.getRegionIterArgs()[1 + pairIdx]);
    }

    // clone all loop body ops except yield
    rewriter.setInsertionPointToStart(kFor.getBody());
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto *cloned = rewriter.clone(op, loopMapping);
      for (Value result : cloned->getResults()) {
        // TODO: if we copy in 1D tensor types this won't work...
        if (auto resultTensorTy =
                dyn_cast<RankedTensorType>(result.getType())) {
          assert(resultTensorTy.getShape().size() == 2 &&
                 "expected only rank-2 tensors in K loop");
          result.setType(updateTensorType(result.getType(), vectorShape));
        }
      }
    }

    Value newAcc = loopMapping.lookup(dotOp.getResult());
    // new loop yields new accumulator + updated per-tile ptrs
    SmallVector<Value> kYields = {newAcc};
    for (auto [argIdx, stride] : info->ptrArgs)
      kYields.push_back(loopMapping.lookup(yieldOp.getOperands()[argIdx]));
    scf::YieldOp::create(rewriter, loc, kYields);

    // yield the accumulated tile results from the newly created generic
    rewriter.setInsertionPointAfter(kFor);
    cpu::YieldOp::create(rewriter, loc, ValueRange{kFor.getResult(0)});

    rewriter.replaceAllUsesWith(forOp.getResult(info->cIterArgIdx),
                                generic.getResult(0));

    // erase unused generic op arguments
    // TODO: is this really needed? what if instead of populating generic
    // operands up front we created empty operands and then updated them at the
    // end as we added ops into the block?
    SmallVector<Value> newIns(generic.getIns().begin(), generic.getIns().end());
    SmallVector<unsigned> argsToErase;
    Block *body = &generic.getBody().front();
    for (auto [idx, pair] : llvm::enumerate(
             llvm::zip(generic.getIns(), body->getArguments().drop_front()))) {
      auto [operand, arg] = pair;
      if (arg.use_empty())
        argsToErase.push_back(idx);
    }
    llvm::sort(argsToErase);
    for (auto idx : llvm::reverse(argsToErase)) {
      // offset by 1 since there is one induction variable (tile offset)
      body->eraseArgument(1 + idx);
      newIns.erase(newIns.begin() + idx);
    }
    rewriter.modifyOpInPlace(generic,
                             [&]() { generic.getInsMutable().assign(newIns); });

    rewriter.eraseOp(forOp);
    return success();

#else
    Location loc = dotOp.getLoc();

    if (dotOp->getParentOfType<cpu::GenericOp>())
      return failure();

    auto resultTensorTy = cast<RankedTensorType>(dotOp.getType());
    auto encoding =
        dyn_cast<gpu::BlockedEncodingAttr>(resultTensorTy.getEncoding());
    if (!encoding)
      return failure();

    auto [blockShape, vectorShape] =
        getBlockAndVectorShapes(resultTensorTy, encoding);

    SmallVector<Value> ins(dotOp->getOperands().begin(),
                           dotOp->getOperands().end());

    auto generic = cpu::GenericOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{resultTensorTy}, ins,
        /*params=*/ValueRange{}, blockShape, vectorShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, vectorShape, bodyMapping);
    auto newDot = rewriter.clone(*dotOp, bodyMapping);
    for (Value result : newDot->getResults())
      result.setType(updateTensorType(result.getType(), vectorShape));
    cpu::YieldOp::create(rewriter, loc, newDot->getResults());

    rewriter.replaceOp(dotOp, generic.getResults());
#endif
    return success();
  }
};

// Returns true if defOp can be cloned into a generic body during fusion.
// Reduction generics (cpu::GenericOp with scalar results) are not fusible —
// their scalar outputs become params of the consumer, not tiled ins.
static bool isFusible(Operation *defOp) {
  if (!defOp)
    return false;
  if (isa<arith::ConstantOp>(defOp))
    return true;
  if ((isa<arith::ArithDialect, math::MathDialect>(defOp->getDialect())) &&
      defOp->hasTrait<OpTrait::Elementwise>())
    return true;
  if (isa<triton::AddPtrOp>(defOp))
    return true;
  if (isa<triton::LoadOp>(defOp))
    return true;
  // tt.splat broadcasts a scalar to a tensor; the scalar input becomes a param.
  if (isa<triton::SplatOp>(defOp))
    return true;
  if (isa<triton::BroadcastOp>(defOp))
    return true;
  // tt.make_range can be fused by rewriting make_range to
  // ttc.make_dynamic_range, taking the chunk offset as a parameter
  if (isa<triton::MakeRangeOp>(defOp))
    return true;
#if 0
  // TODO: allow convert layout to be fused - but if the op is used elsewhere we should probably make a copy
  if (isa<triton::gpu::ConvertLayoutOp>(defOp))
      return true;
#endif
  return false;
}

// If v is a block argument of an scf.for (i.e. an iter_arg), return the
// corresponding initial value passed to the loop. Otherwise return v unchanged.
// This lets the fusion worklist see through loop-carried values to their
// defining ops (e.g. constants used as iter_arg initialisers).
static Value getIterArgInit(Value v) {
  auto blockArg = dyn_cast<BlockArgument>(v);
  if (!blockArg)
    return v;
  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp)
    return v;
  unsigned argIdx = blockArg.getArgNumber();
  unsigned numIV = forOp.getNumInductionVars();
  if (argIdx < numIV || (argIdx - numIV) >= forOp.getInitArgs().size())
    return v;
  return forOp.getInitArgs()[argIdx - numIV];
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

  // Collect fusible ops reachable from the current ins values, in
  // def-before-use order (SetVector preserves insertion order; we walk
  // backwards so reversing gives topo order for cloning).
  SetVector<Operation *> opsToFuse;
  SmallVector<Value> worklist(genericOp.getIns().begin(),
                              genericOp.getIns().end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    // See through scf.for iter_args to their initial values so we can fuse
    // the ops that produce those values (e.g. constants used as initialisers).
    Operation *defOp = getIterArgInit(v).getDefiningOp();
    if (!defOp || !isFusible(defOp) || opsToFuse.contains(defOp))
      continue;
    opsToFuse.insert(defOp);
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  if (opsToFuse.empty())
    return;

  // Build lookup: existing ins value → 0-based ins index (not block arg index).
  // The offset to convert to a block arg index is added at the two sites that
  // access block args, keeping the newIns indexing straightforward.
  unsigned numInductionVars = genericOp.getNumInductionVars();
  DenseMap<Value, unsigned> insToArgIdx;
  for (auto [idx, v] : llvm::enumerate(genericOp.getIns())) {
    insToArgIdx[v] = idx;
    // If v is an scf.for iter_arg, also map its initial value so that when the
    // defining op of the init is cloned its result is recognised as replacing
    // this ins slot.
    Value init = getIterArgInit(v);
    if (init != v)
      insToArgIdx[init] = idx;
  }

  SmallVector<Value> newIns(genericOp.getIns().begin(),
                            genericOp.getIns().end());
  SmallVector<Value> newParams(genericOp.getParams().begin(),
                               genericOp.getParams().end());
  SmallVector<unsigned> insIdxToRemove;

  Block *body = &genericOp.getBody().front();
  // Insert cloned ops before the first existing body op.
  rewriter.setInsertionPointToStart(body);

  // Sort opsToFuse into true topological order (defs before uses).
  // The worklist DFS can insert the same dependency via two different paths
  // (diamond-shaped graphs), so reversing the insertion order is not
  // guaranteed to give a valid topo order.  We do an explicit post-order DFS.
  SmallVector<Operation *> sortedOps;
  {
    DenseSet<Operation *> visited;
    std::function<void(Operation *)> visit = [&](Operation *op) {
      if (!opsToFuse.contains(op) || visited.contains(op))
        return;
      visited.insert(op);
      for (Value operand : op->getOperands()) {
        if (auto *def = operand.getDefiningOp())
          visit(def);
        // Also follow scf.for iter_arg initialisers.
        Value init = getIterArgInit(operand);
        if (init != operand)
          if (auto *def = init.getDefiningOp())
            visit(def);
      }
      sortedOps.push_back(op);
    };
    for (Operation *op : opsToFuse)
      visit(op);
  }

  IRMapping mapping;
  // Clone in topological (def-before-use) order.
  for (Operation *op : sortedOps) {
    if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
      // replace op with makeDynamicRange
      auto newMakeRangeResultType =
          updateTensorType(makeRangeOp.getResult().getType(), sizePerThreadVec);
      auto makeDynamicRangeOp = triton::cpu::MakeDynamicRangeOp::create(
          rewriter, makeRangeOp.getLoc(), newMakeRangeResultType,
          genericOp.getChunkOffset());
      mapping.map(makeRangeOp->getResults(), makeDynamicRangeOp->getResults());
      if (auto it = insToArgIdx.find(makeRangeOp.getResult());
          it != insToArgIdx.end()) {
        body->getArgument(it->second + numInductionVars)
            .replaceAllUsesWith(makeDynamicRangeOp.getResult());
        insIdxToRemove.push_back(it->second);
      }
      continue;
    }
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      auto tensorTy =
          dyn_cast<RankedTensorType>(constantOp.getResult().getType());
      if (tensorTy) {
        auto newTensorTy = cast<RankedTensorType>(
            updateTensorType(tensorTy, sizePerThreadVec));
        auto denseAttr = cast<DenseElementsAttr>(constantOp.getValue());
        assert(denseAttr.isSplat() &&
               "non-splat tensor constants not yet supported in fuseInputs");
        auto newAttr = DenseElementsAttr::get(
            newTensorTy, *denseAttr.getValues<Attribute>().begin());
        auto newConstant =
            arith::ConstantOp::create(rewriter, constantOp.getLoc(), newAttr);
        mapping.map(constantOp.getResult(), newConstant.getResult());
        if (auto it = insToArgIdx.find(constantOp.getResult());
            it != insToArgIdx.end()) {
          body->getArgument(it->second + numInductionVars)
              .replaceAllUsesWith(newConstant.getResult());
          insIdxToRemove.push_back(it->second);
        }
        continue;
      }
    }
#if 0
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        // update the result type
        auto inTy = cast<RankedTensorType>(cvtOp.getSrc().getType());
        auto outTy = cast<RankedTensorType>(cvtOp.getResult().getType());

        mapping.map(cvtOp.getSrc(), body->addArgument(updateTensorType(inTy, sizePerThreadVec), cvtOp.getLoc()));

        auto newCvt = triton::gpu::ConvertLayoutOp::create(rewriter, cvtOp.getLoc(), updateTensorType(outTy, sizePerThreadVec), mapping.lookup(cvtOp.getSrc()));
        mapping.map(cvtOp.getResult(), newCvt.getResult());
    }
#endif

    for (Value operand : op->getOperands()) {
      if (mapping.contains(operand))
        continue;

      newIns.push_back(operand);
      mapping.map(operand, body->addArgument(updateTensorType(operand.getType(),
                                                              sizePerThreadVec),
                                             operand.getLoc()));
    }

    Operation *newOp = rewriter.clone(*op, mapping);
    for (auto [origResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      newResult.setType(
          updateTensorType(newResult.getType(), sizePerThreadVec));
      // If this result was previously an ins, replace its block arg with the
      // newly cloned result and mark the arg for removal.
      if (auto it = insToArgIdx.find(origResult); it != insToArgIdx.end()) {
        body->getArgument(it->second + numInductionVars)
            .replaceAllUsesWith(newResult);
        insIdxToRemove.push_back(it->second);
      }
    }
    mapping.map(op->getResults(), newOp->getResults());
  }

  // Remove replaced ins entries and their block args (reverse order for index
  // stability). Both use the same 0-based ins index; block arg access adds the
  // induction var offset.
  llvm::sort(insIdxToRemove);
  for (unsigned idx : llvm::reverse(insIdxToRemove)) {
    body->eraseArgument(idx + numInductionVars);
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
    patterns.add<WrapKLoopWithDotOp>(context, benefitDefault);

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
