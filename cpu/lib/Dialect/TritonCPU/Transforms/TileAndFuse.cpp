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

// Replace the shape of a RankedTensorType with tileShape, preserving element
// type and encoding. Non-tensor types are returned unchanged.
static Type updateTensorType(Type t, ArrayRef<int32_t> tileShape) {
  auto tensorType = dyn_cast<RankedTensorType>(t);
  if (!tensorType)
    return t;
  return RankedTensorType::get(
      llvm::to_vector(
          llvm::map_range(tileShape, [](int32_t s) { return int64_t(s); })),
      tensorType.getElementType(), tensorType.getEncoding());
}

// Extract blockShape (full tensor shape) and tileShape (sizePerThread) from
// a tensor type with BlockedEncoding.
static std::pair<SmallVector<int32_t>, SmallVector<int32_t>>
getBlockAndTileShapes(RankedTensorType tensorTy,
                      gpu::BlockedEncodingAttr encoding) {
  auto shape = tensorTy.getShape();
  SmallVector<int32_t> blockShape(shape.begin(), shape.end());
  auto sizePerThread = encoding.getSizePerThread();
  SmallVector<int32_t> tileShape(sizePerThread.begin(), sizePerThread.end());
  return {blockShape, tileShape};
}

struct TiledInput {
  Value value;
  SmallVector<int32_t> shape;
};

// Create the body block of a GenericOp, adding one block arg per ins value
// with tensor types replaced to the vector (chunk) shape. Populates `mapping`
// with ins value → block arg entries and sets the insertion point to the start
// of the block. Returns the new block.
static Block *initGenericBody(OpBuilder &rewriter, cpu::GenericOp generic,
                              ArrayRef<TiledInput> ins,
                              ArrayRef<int32_t> tileShape, IRMapping &mapping) {
  Block *body = rewriter.createBlock(&generic.getBody());
  for (unsigned i = 0; i < tileShape.size(); i++)
    body->addArgument(rewriter.getI32Type(),
                      generic.getLoc()); // tile offset per vector shape dim

  for (auto pair : ins) {
    auto [v, tileShape] = pair;
    Type argTy = updateTensorType(v.getType(), tileShape);
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

    auto [blockShape, tileShape] = getBlockAndTileShapes(tensorTy, encoding);

    SmallVector<TiledInput> ins;
    for (auto value : storeOp->getOperands()) {
      ins.push_back(TiledInput{value, tileShape});
    }

    SmallVector<Value> insValues =
        llvm::map_to_vector(ins, [](const TiledInput &ti) { return ti.value; });

    auto generic =
        cpu::GenericOp::create(rewriter, loc, /*resultTypes=*/TypeRange{},
                               insValues, blockShape, tileShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, tileShape, bodyMapping);

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

    auto [blockShape, tileShape] = getBlockAndTileShapes(tensorTy, encoding);

    SmallVector<TiledInput> ins = {TiledInput{srcs[0], tileShape}};
    SmallVector<Type> resultTypes(reduceOp.getResultTypes().begin(),
                                  reduceOp.getResultTypes().end());
    if (srcUsedElsewhere)
      resultTypes.push_back(tensorTy);

    LDBG("Creating reduction generic op, result types: " << resultTypes.size());

    SmallVector<Value> insValues =
        llvm::map_to_vector(ins, [](const TiledInput &ti) { return ti.value; });
    auto generic = cpu::GenericOp::create(rewriter, loc, resultTypes, insValues,
                                          blockShape, tileShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, tileShape, bodyMapping);

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

struct WrapKLoopWithDotOp : public mlir::OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  struct KLoopDotInfo {
    triton::DotOp dotOp;
    unsigned cIterArgIdx;
    SmallVector<std::pair<unsigned, Value>>
        ptrArgs; // loop carried ptr values for A and B dot operands
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

    auto info = matchKLoopWithDot(forOp);
    if (!info)
      return failure();

    Location loc = forOp.getLoc();
    triton::DotOp dotOp = info->dotOp;
    auto resultTy = cast<RankedTensorType>(dotOp.getType());
    auto encoding = cast<gpu::BlockedEncodingAttr>(resultTy.getEncoding());

    // use the MxN (result) shape for block/tile shapes. The K loop is not
    // currently tiled.
    auto [blockShape, tileShape] = getBlockAndTileShapes(resultTy, encoding);

    auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
    assert(aTy.getRank() == 2 && "only 2D dot op supported");
    int32_t kSize = (int32_t)aTy.getShape()[1];

    SmallVector<int32_t> aTileShape = {tileShape[0], kSize};
    SmallVector<int32_t> bTileShape = {kSize, tileShape[1]};

    struct OperandChain {
      SetVector<Operation *> body;
      SmallVector<TiledInput> ins;
    };

    // Walk backward from rootOp (inside forOp's body) via getBackwardSlice.
    // Populates:
    //   body   — ops strictly inside forOp that are in this chain
    //   ins — values defined outside forOp consumed by this chain,
    //               including iter_arg init values traced across the boundary
    unsigned numIVs = forOp.getNumInductionVars();
    auto buildChain = [&](Operation *rootOp,
                          ArrayRef<int32_t> shape) -> OperandChain {
      OperandChain chain;

      BackwardSliceOptions opts;
      opts.omitBlockArguments = true;
      opts.filter = [&](Operation *op) {
        return op != forOp && forOp->isAncestor(op);
      };
      (void)getBackwardSlice(rootOp, &chain.body, opts);
      chain.body.insert(rootOp); // getBackwardSlice excludes the root itself

      for (Operation *op : chain.body) {
        for (Value operand : op->getOperands()) {
          if (auto barg = dyn_cast<BlockArgument>(operand)) {
            if (barg.getOwner() != forOp.getBody())
              continue;
            if (barg.getArgNumber() < numIVs)
              continue; // induction variable — defined by the loop, not
                        // external
            // iter_arg: cross the boundary to its init value
            Value initVal = forOp.getInitArgs()[barg.getArgNumber() - numIVs];
            bool isTensor = isa<RankedTensorType>(initVal.getType());
            chain.ins.push_back({initVal, isTensor ? SmallVector<int32_t>(shape)
                                                   : SmallVector<int32_t>{}});
          } else if (!forOp->isAncestor(operand.getDefiningOp())) {
            bool isTensor = isa<RankedTensorType>(operand.getType());
            chain.ins.push_back({operand, isTensor ? SmallVector<int32_t>(shape)
                                                   : SmallVector<int32_t>{}});
          }
        }
      }
      return chain;
    };

    // dotOp.getA() and getB() are results of loads inside the for body.
    // dotOp.getC() is an iter_arg (block argument) — no chain needed, we
    // reconstruct the zero accumulator at tile size inside the generic body.
    OperandChain aChain = buildChain(dotOp.getA().getDefiningOp(), aTileShape);
    OperandChain bChain = buildChain(dotOp.getB().getDefiningOp(), bTileShape);

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto [iterArgIdx, offset] : info->ptrArgs) {
      Operation *addPtrOp =
          yieldOp.getOperands()[iterArgIdx].getDefiningOp<triton::AddPtrOp>();
      assert(addPtrOp && "expected tt::add_ptr");
      // determine which chain owns this ptr arg by checking aChain's ins
      const auto iterArgIdx_ =
          iterArgIdx; // captured structured bindings workaround
      bool isAChain = llvm::any_of(aChain.ins, [&](const TiledInput &ti) {
        return ti.value == forOp.getInitArgs()[iterArgIdx_];
      });
      OperandChain &chain = isAChain ? aChain : bChain;
      ArrayRef<int32_t> shape = isAChain ? aTileShape : bTileShape;

      // pull in the addptr and its offset operand chain
      BackwardSliceOptions opts;
      opts.omitBlockArguments = true;
      opts.filter = [&](Operation *op) {
        return op != forOp && forOp->isAncestor(op);
      };
      (void)getBackwardSlice(addPtrOp, &chain.body, opts);
      chain.body.insert(addPtrOp);
      // offset may be external
      if (!forOp->isAncestor(offset.getDefiningOp()))
        chain.ins.push_back({offset, {}});
    }

    // Build ins: for loop bounds (scalars, needed to reconstruct the inner
    // scf.for) followed by A-chain then B-chain ins.
    SmallVector<TiledInput> ins;
    ins.push_back({forOp.getLowerBound(), {}});
    ins.push_back({forOp.getUpperBound(), {}});
    ins.push_back({forOp.getStep(), {}});
    for (auto &inValue : aChain.ins)
      ins.push_back(inValue);
    for (auto &inValue : bChain.ins)
      ins.push_back(inValue);

    SmallVector<Value> insValues =
        llvm::map_to_vector(ins, [](const TiledInput &ti) { return ti.value; });

    auto generic = cpu::GenericOp::create(rewriter, loc, TypeRange{resultTy},
                                          insValues, blockShape, tileShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, tileShape, bodyMapping);

    Block *body = &generic.getBody().front();
    unsigned argIdx = generic.getNumInductionVars();

    for (Value scalar :
         {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()})
      bodyMapping.map(scalar, body->getArgument(argIdx++));

    IRMapping aMapping, bMapping;
    for (auto &ti : aChain.ins)
      aMapping.map(ti.value, body->getArgument(argIdx++));
    for (auto &ti : bChain.ins)
      bMapping.map(ti.value, body->getArgument(argIdx++));

    rewriter.setInsertionPointToStart(body);

    // TODO: what if C is not loop carried?
    // Zero-splat C at tile size
    auto cInitConst = forOp.getInitArgs()[info->cIterArgIdx]
                          .getDefiningOp<arith::ConstantOp>();
    auto cDense = cast<DenseElementsAttr>(cInitConst.getValue());
    auto cTileTy =
        cast<RankedTensorType>(updateTensorType(resultTy, tileShape));
    Value cTileInit = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(cTileTy,
                               *cDense.getValues<Attribute>().begin()));

    // Assemble iter_args for inner for: same slots as original
    SmallVector<Value> innerInitArgs(forOp.getInitArgs().size());
    innerInitArgs[info->cIterArgIdx] = cTileInit;
    for (auto [iterArgIdx, _] : info->ptrArgs) {
      Value origInit = forOp.getInitArgs()[iterArgIdx];
      // init value lands in whichever chain owns it
      innerInitArgs[iterArgIdx] = aMapping.contains(origInit)
                                      ? aMapping.lookup(origInit)
                                      : bMapping.lookup(origInit);
    }

    auto innerFor = scf::ForOp::create(
        rewriter, loc, bodyMapping.lookup(forOp.getLowerBound()),
        bodyMapping.lookup(forOp.getUpperBound()),
        bodyMapping.lookup(forOp.getStep()), innerInitArgs);
    rewriter.setInsertionPointToStart(innerFor.getBody());

    // TODO: lambda-ify all this
    // Remap original IV and iter_args to inner for's block args
    auto remapIterArgs = [&](IRMapping &m) {
      m.map(forOp.getInductionVar(), innerFor.getInductionVar());
      for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i)
        m.map(forOp.getBody()->getArgument(numIVs + i),
              innerFor.getBody()->getArgument(numIVs + i));
    };

    remapIterArgs(aMapping);
    for (Operation *op : aChain.body) {
      Operation *cloned = rewriter.clone(*op, aMapping);
      for (auto [orig, res] : llvm::zip(op->getResults(), cloned->getResults()))
        res.setType(updateTensorType(orig.getType(), aTileShape));
      aMapping.map(op->getResults(), cloned->getResults());
    }

    remapIterArgs(bMapping);
    for (Operation *op : bChain.body) {
      Operation *cloned = rewriter.clone(*op, bMapping);
      for (auto [orig, res] : llvm::zip(op->getResults(), cloned->getResults()))
        res.setType(updateTensorType(orig.getType(), bTileShape));
      bMapping.map(op->getResults(), cloned->getResults());
    }

    bodyMapping.map(dotOp.getA(), aMapping.lookup(dotOp.getA()));
    bodyMapping.map(dotOp.getB(), bMapping.lookup(dotOp.getB()));
    bodyMapping.map(dotOp.getC(), innerFor.getBody()->getArgument(
                                      numIVs + info->cIterArgIdx));
    Operation *clonedDot = rewriter.clone(*dotOp, bodyMapping);
    clonedDot->getResult(0).setType(updateTensorType(resultTy, tileShape));

    auto origYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> yieldVals(forOp.getInitArgs().size());
    yieldVals[info->cIterArgIdx] = clonedDot->getResult(0);
    for (auto [iterArgIdx, _] : info->ptrArgs) {
      Value origYieldVal = origYield.getOperands()[iterArgIdx];
      // look up the cloned addptr in whichever chain owns it
      yieldVals[iterArgIdx] = aMapping.contains(origYieldVal)
                                  ? aMapping.lookup(origYieldVal)
                                  : bMapping.lookup(origYieldVal);
    }
    scf::YieldOp::create(rewriter, loc, yieldVals);

    rewriter.setInsertionPointAfter(innerFor);
    cpu::YieldOp::create(rewriter, loc,
                         ValueRange{innerFor.getResult(info->cIterArgIdx)});

    rewriter.replaceAllUsesWith(forOp.getResult(info->cIterArgIdx),
                                generic.getResult(0));
    rewriter.eraseOp(forOp);
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
  // tt.make_range can be fused by rewriting make_range to
  // ttc.make_dynamic_range, taking the chunk offset as a parameter
  if (isa<triton::MakeRangeOp>(defOp))
    return true;
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

class InputFuser {
public:
  InputFuser(IRRewriter &rewriter, cpu::GenericOp genericOp)
      : rewriter(rewriter), genericOp(genericOp) {}

  LogicalResult run();

private:
  struct Chain {
    Value root;
    unsigned insIdx;
    SmallVector<int32_t> tileShape;
  };

  SmallVector<Operation *> collectChain(Chain &chain);

  void cloneOp(Operation *op, Chain &chain, SmallVector<Value> &newIns,
               SmallVector<unsigned> &insIdxToRemove, IRMapping &mapping);

  IRRewriter &rewriter;
  cpu::GenericOp genericOp;
};

LogicalResult InputFuser::run() {
  Block *body = &genericOp.getBody().front();
  unsigned numIV = genericOp.getNumInductionVars();

  SmallVector<Value> newIns(genericOp.getIns());
  SmallVector<unsigned> insIdxToRemove;

  for (auto [insIdx, root] : llvm::enumerate(genericOp.getIns())) {
    BlockArgument blockArg = body->getArgument(numIV + insIdx);
    auto tensorTy = dyn_cast<RankedTensorType>(blockArg.getType());
    // only fuse tensor inputs
    if (!tensorTy)
      continue;

    SmallVector<int32_t> tileShape(tensorTy.getShape());
    Chain chain{root, (unsigned)insIdx, tileShape};
    SmallVector<Operation *> sorted = collectChain(chain);
    if (sorted.empty())
      continue;

    IRMapping mapping;
    rewriter.setInsertionPointToStart(body);
    for (Operation *op : sorted) {
      cloneOp(op, chain, newIns, insIdxToRemove, mapping);
    }
  }

  llvm::sort(insIdxToRemove);
  for (unsigned idx : llvm::reverse(insIdxToRemove)) {
    body->eraseArgument(idx + numIV);
    newIns.erase(newIns.begin() + idx);
  }

  rewriter.modifyOpInPlace(genericOp,
                           [&]() { genericOp.getInsMutable().assign(newIns); });

  return success();
}

SmallVector<Operation *> InputFuser::collectChain(Chain &chain) {
  SetVector<Operation *> opsToFuse;

  SmallVector<Value> worklist{chain.root};
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

  // Sort opsToFuse into true topological order (defs before uses).
  // The worklist DFS can insert the same dependency via two different paths
  // (diamond-shaped graphs), so reversing the insertion order is not
  // guaranteed to give a valid topo order.  We do an explicit post-order DFS.
  SmallVector<Operation *> sortedOps;
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

  return sortedOps;
}

void InputFuser::cloneOp(Operation *op, Chain &chain,
                         SmallVector<Value> &newIns,
                         SmallVector<unsigned> &insIdxToRemove,
                         IRMapping &mapping) {
  Block *body = &genericOp.getBody().front();
  unsigned numIV = genericOp.getNumInductionVars();

  // chain.root may be an scf.for iter_arg; the defining op produces the init
  // value, not root itself. Treat both as equivalent for block arg replacement.
  Value effectiveRoot = getIterArgInit(chain.root);

  auto replaceRootIfMatch = [&](Value origResult, Value clonedResult) {
    if (origResult != effectiveRoot)
      return;
    body->getArgument(numIV + chain.insIdx).replaceAllUsesWith(clonedResult);
    insIdxToRemove.push_back(chain.insIdx);
    // Also map root (the iter_arg) so downstream ops resolve through the
    // mapping rather than falling through to the "add as new ins" path.
    if (effectiveRoot != chain.root)
      mapping.map(chain.root, clonedResult);
  };

  // make range must be replaced with make dynamic range which takes the current
  // tile offset as a parameter
  if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
    auto resultType = makeRangeOp.getResult().getType();
    auto newResultType = updateTensorType(resultType, chain.tileShape);
    auto makeDynamicRangeOp = triton::cpu::MakeDynamicRangeOp::create(
        rewriter, makeRangeOp.getLoc(), newResultType,
        genericOp.getTileOffset(/*dim=*/0));
    mapping.map(makeRangeOp->getResults(), makeDynamicRangeOp->getResults());
    replaceRootIfMatch(makeRangeOp.getResult(), makeDynamicRangeOp.getResult());
    return;
  }

  // constant ops with tensor type get replaced with a constant op using the
  // tile size
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    auto resultTensorType =
        dyn_cast<RankedTensorType>(constantOp.getResult().getType());
    if (resultTensorType) {
      auto newTensorType = cast<RankedTensorType>(
          updateTensorType(resultTensorType, chain.tileShape));
      auto denseAttr = cast<DenseElementsAttr>(constantOp.getValue());
      assert(denseAttr.isSplat() &&
             "non-splat tensor constants not yet supported in fuseInputs");
      auto newAttr = DenseElementsAttr::get(
          newTensorType, *denseAttr.getValues<Attribute>().begin());
      auto newConstant =
          arith::ConstantOp::create(rewriter, constantOp.getLoc(), newAttr);
      mapping.map(constantOp.getResult(), newConstant.getResult());
      replaceRootIfMatch(constantOp.getResult(), newConstant.getResult());
      return;
    }
  }

  // general case - clone op
  for (Value operand : op->getOperands()) {
    if (mapping.contains(operand))
      continue;

    newIns.push_back(operand);
    mapping.map(operand, body->addArgument(updateTensorType(operand.getType(),
                                                            chain.tileShape),
                                           operand.getLoc()));
  }

  Operation *newOp = rewriter.clone(*op, mapping);
  for (auto [origResult, newResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    newResult.setType(updateTensorType(newResult.getType(), chain.tileShape));
    replaceRootIfMatch(origResult, newResult);
  }
  mapping.map(op->getResults(), newOp->getResults());
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

    // Step 2: Fuse elementwise ops and loads into each generic
    SmallVector<cpu::GenericOp> worklist;
    m.walk([&](cpu::GenericOp op) { worklist.push_back(op); });
    IRRewriter rewriter(context);
    for (cpu::GenericOp genericOp : llvm::reverse(worklist)) {
      InputFuser fuser(rewriter, genericOp);
      if (fuser.run().failed())
        signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
