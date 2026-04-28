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
  assert(tensorType.getRank() == tileShape.size() &&
         "expected tensor type and tile shape to have same rank");
  SmallVector<int64_t> newShape;
  // for broadcast/expanded dims (size 1) do not use the tile shape
  for (auto [s, tile] : llvm::zip(tensorType.getShape(), tileShape))
    newShape.push_back(std::min(s, (int64_t)tile));

  return RankedTensorType::get(newShape, tensorType.getElementType(),
                               tensorType.getEncoding());
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
    const bool srcIsLoad =
        srcs[0].getDefiningOp() && isa<LoadOp>(srcs[0].getDefiningOp());
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

    BackwardSliceOptions sliceOpts;
    sliceOpts.omitBlockArguments = true;
    sliceOpts.filter = [&](Operation *op) {
      return op != forOp && forOp->isAncestor(op);
    };

    // Scan `ops` for operands defined outside forOp and append them to
    // `chain.ins`. Crosses iter_arg boundaries to their init values; treats
    // other external block args (e.g. function arguments) as plain externals.
    auto collectExternals = [&](ArrayRef<Operation *> ops, OperandChain &chain,
                                ArrayRef<int32_t> shape) {
      DenseSet<Value> seen;
      for (auto &ti : chain.ins) // pre-populate from prior calls
        seen.insert(ti.value);

      for (Operation *op : ops) {
        for (Value operand : op->getOperands()) {
          if (auto barg = dyn_cast<BlockArgument>(operand)) {
            if (barg.getOwner() == forOp.getBody()) {
              if (barg.getArgNumber() < numIVs)
                continue; // induction variable
              Value initVal = forOp.getInitArgs()[barg.getArgNumber() - numIVs];
              if (seen.insert(initVal).second) {
                bool isTensor = isa<RankedTensorType>(initVal.getType());
                chain.ins.push_back({initVal, isTensor
                                                  ? SmallVector<int32_t>(shape)
                                                  : SmallVector<int32_t>{}});
              }
            } else {
              // external block arg (e.g. function argument)
              if (seen.insert(operand).second) {
                bool isTensor = isa<RankedTensorType>(operand.getType());
                chain.ins.push_back({operand, isTensor
                                                  ? SmallVector<int32_t>(shape)
                                                  : SmallVector<int32_t>{}});
              }
            }
          } else if (!forOp->isAncestor(operand.getDefiningOp())) {
            if (seen.insert(operand).second) {
              bool isTensor = isa<RankedTensorType>(operand.getType());
              chain.ins.push_back({operand, isTensor
                                                ? SmallVector<int32_t>(shape)
                                                : SmallVector<int32_t>{}});
            }
          }
        }
      }
    };

    auto buildChain = [&](Operation *rootOp,
                          ArrayRef<int32_t> shape) -> OperandChain {
      OperandChain chain;
      (void)getBackwardSlice(rootOp, &chain.body, sliceOpts);
      chain.body.insert(rootOp); // getBackwardSlice excludes the root itself
      collectExternals(chain.body.getArrayRef(), chain, shape);
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
      collectExternals(chain.body.getArrayRef(), chain, shape);
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

struct WrapConvertLayoutOp
    : mlir::OpRewritePattern<triton::gpu::ConvertLayoutOp> {
  using OpRewritePattern<triton::gpu::ConvertLayoutOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = cvtOp.getLoc();

    if (cvtOp->getParentOfType<cpu::GenericOp>())
      return failure();

    auto operand = cvtOp.getSrc();

    auto tensorTy = cast<RankedTensorType>(operand.getType());
    auto encoding = dyn_cast<gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
    if (!encoding)
      return failure();

    // only wrap blocked->blocked conversions
    auto convertedTensorTy =
        cast<RankedTensorType>(cvtOp.getResult().getType());
    if (!isa<triton::gpu::BlockedEncodingAttr>(convertedTensorTy.getEncoding()))
      return failure();

    // Note: this uses the source tensor ty to determine the tile shape. if we
    // tile over the source, we should be able to write to any location in the
    // output
    auto [blockShape, tileShape] = getBlockAndTileShapes(tensorTy, encoding);

    // but, overwrite the tileshape to match the block shape (i.e. just generate
    // a single tile generic for now)
    tileShape = blockShape;

    SmallVector<TiledInput> ins;
    for (auto value : cvtOp->getOperands()) {
      ins.push_back(TiledInput{value, tileShape});
    }
    SmallVector<Value> insValues =
        llvm::map_to_vector(ins, [](const TiledInput &ti) { return ti.value; });

    auto generic = cpu::GenericOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{convertedTensorTy}, insValues,
        blockShape, tileShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, tileShape, bodyMapping);

    auto newCvt = rewriter.clone(*cvtOp, bodyMapping);
    cpu::YieldOp::create(rewriter, loc, newCvt->getResults());

    rewriter.replaceOp(cvtOp, generic.getResults());
    return success();
  }
};

struct FuseElementwiseIntoGeneric : mlir::OpRewritePattern<cpu::GenericOp> {
  using OpRewritePattern<cpu::GenericOp>::OpRewritePattern;

  static bool isFusibleElementwise(Operation *op) {
    if (!op)
      return false;
    if (op->getNumResults() != 1)
      return false;
    if ((isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) &&
        op->hasTrait<OpTrait::Elementwise>())
      return true;
    if (isa<triton::AddPtrOp>(op))
      return true;
    if (isa<triton::SplatOp>(op))
      return true;
    // note: load isn't really "elementwise", but the tensor of ptrs can be
    // indexed elementwise and the output truncated based on the input size, so
    // we treat it as elementwise
    if (isa<triton::LoadOp>(op))
      return true;
    return false;
  }

  LogicalResult
  matchAndRewrite(cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    Block *body = &genericOp.getBody().front();
    unsigned numIV = genericOp.getNumInductionVars();

    for (auto [i, insVal] : llvm::enumerate(genericOp.getIns())) {
      Operation *op = insVal.getDefiningOp();
      if (!isFusibleElementwise(op))
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = dyn_cast<RankedTensorType>(blockArg.getType());
      if (!tiledType)
        continue;

      SmallVector<int32_t> tileShape(tiledType.getShape());
      SmallVector<Value> newIns(genericOp.getIns());

      IRMapping mapping;
      // 1. Add new block args for source op inputs at body end
      for (Value operand : op->getOperands()) {
        newIns.push_back(operand);
        mapping.map(operand, body->addArgument(
                                 updateTensorType(operand.getType(), tileShape),
                                 operand.getLoc()));
      }

      // 2. clone
      rewriter.setInsertionPointToStart(body);
      Operation *newOp = rewriter.clone(*op, mapping);
      Type origResultType = newOp->getResult(0).getType();
      newOp->getResult(0).setType(updateTensorType(origResultType, tileShape));

      // 3. replace block arg and clean up
      // note that newOp must have 1 and only 1 result due to isFusible above
      blockArg.replaceAllUsesWith(newOp->getResult(0));
      body->eraseArgument(numIV + i);
      newIns.erase(newIns.begin() + i);

      rewriter.modifyOpInPlace(
          genericOp, [&]() { genericOp.getInsMutable().assign(newIns); });

      return success();
    }

    return failure();
  }
};

struct FuseMakeRangeIntoGeneric : mlir::OpRewritePattern<cpu::GenericOp> {
  using OpRewritePattern<cpu::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    Block *body = &genericOp.getBody().front();
    unsigned numIV = genericOp.getNumInductionVars();

    for (auto [i, insVal] : llvm::enumerate(genericOp.getIns())) {
      Operation *op = insVal.getDefiningOp();
      if (!op)
        continue;
      triton::MakeRangeOp makeRangeOp = dyn_cast<triton::MakeRangeOp>(op);
      if (!makeRangeOp)
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = cast<RankedTensorType>(blockArg.getType());

      SmallVector<int32_t> tileShape(tiledType.getShape());
      SmallVector<Value> newIns(genericOp.getIns());

      // 1. clone (make range has no operands)
      rewriter.setInsertionPointToStart(body);

      auto resultType =
          cast<RankedTensorType>(makeRangeOp.getResult().getType());

      const bool isNotTiled = llvm::all_of(
          llvm::zip(tileShape, resultType.getShape()), [](auto pair) {
            auto [t, s] = pair;
            return t == s;
          });

      IRMapping mapping;
      Operation *newOp;
      if (isNotTiled) {
        // just fuse the existing op
        newOp = rewriter.clone(*op, mapping);
      } else {
        auto newResultType = updateTensorType(resultType, tileShape);
        auto sliceEncodingAttr =
            dyn_cast<triton::gpu::SliceEncodingAttr>(resultType.getEncoding());
        unsigned dim = sliceEncodingAttr ? sliceEncodingAttr.getDim() : 0;
        newOp = triton::cpu::MakeDynamicRangeOp::create(
            rewriter, makeRangeOp.getLoc(), newResultType,
            genericOp.getTileOffset(dim));
      }
      assert(newOp && "expected make range op to be replaced or fused");

      // 3. update existing uses
      blockArg.replaceAllUsesWith(newOp->getResult(0));
      body->eraseArgument(numIV + i);
      newIns.erase(newIns.begin() + i);

      rewriter.modifyOpInPlace(
          genericOp, [&]() { genericOp.getInsMutable().assign(newIns); });
      return success();
    }
    return failure();
  }
};

struct FuseConstantIntoGeneric : mlir::OpRewritePattern<cpu::GenericOp> {
  using OpRewritePattern<cpu::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    Block *body = &genericOp.getBody().front();
    unsigned numIV = genericOp.getNumInductionVars();

    for (auto [i, insVal] : llvm::enumerate(genericOp.getIns())) {
      Operation *op = insVal.getDefiningOp();
      if (!op)
        continue;

      auto constantOp = dyn_cast<arith::ConstantOp>(op);
      if (!constantOp)
        continue;

      auto resultTensorType =
          dyn_cast<RankedTensorType>(constantOp.getResult().getType());
      if (!resultTensorType)
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = cast<RankedTensorType>(blockArg.getType());

      SmallVector<int32_t> tileShape(tiledType.getShape());
      SmallVector<Value> newIns(genericOp.getIns());

      // 1. clone. constants have no operands to update
      rewriter.setInsertionPointToStart(body);
      auto newTensorType =
          cast<RankedTensorType>(updateTensorType(resultTensorType, tileShape));
      auto denseAttr = cast<DenseElementsAttr>(constantOp.getValue());
      assert(denseAttr.isSplat() &&
             "non-splat tensor constants not yet supported in fuseInputs");
      auto newAttr = DenseElementsAttr::get(
          newTensorType, *denseAttr.getValues<Attribute>().begin());
      auto newConstant =
          arith::ConstantOp::create(rewriter, constantOp.getLoc(), newAttr);

      // 3. update existing uses
      blockArg.replaceAllUsesWith(newConstant.getResult());
      body->eraseArgument(numIV + i);
      newIns.erase(newIns.begin() + i);

      rewriter.modifyOpInPlace(
          genericOp, [&]() { genericOp.getInsMutable().assign(newIns); });

      return success();
    }
    return failure();
  }
};

struct FuseBroadcastIntoGeneric
    : public mlir::OpRewritePattern<cpu::GenericOp> {
  using OpRewritePattern<cpu::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    Block *body = &genericOp.getBody().front();
    unsigned numIV = genericOp.getNumInductionVars();

    for (auto [i, insVal] : llvm::enumerate(genericOp.getIns())) {
      Operation *op = insVal.getDefiningOp();
      if (!op)
        continue;

      auto broadcastOp = dyn_cast<triton::BroadcastOp>(op);
      if (!broadcastOp)
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = cast<RankedTensorType>(blockArg.getType());
      SmallVector<int32_t> tileShape(tiledType.getShape());
      SmallVector<Value> newIns(genericOp.getIns());

      // broadcast source operand: only tile the non-broadcast dim
      RankedTensorType sourceTensorType =
          cast<RankedTensorType>(broadcastOp.getSrc().getType());
      SmallVector<int32_t> sourceTileShape = llvm::to_vector(
          llvm::map_range(llvm::zip(sourceTensorType.getShape(), tileShape),
                          [](auto pair) -> int32_t {
                            auto [s, t] = pair;
                            return s == 1 ? s : t;
                          }));

      IRMapping mapping;
      // 1. map src operand to block args
      newIns.push_back(broadcastOp.getSrc());
      mapping.map(
          broadcastOp.getSrc(),
          body->addArgument(updateTensorType(sourceTensorType, sourceTileShape),
                            broadcastOp.getSrc().getLoc()));

      // 2. clone the broadcast
      rewriter.setInsertionPointToStart(body);
      Operation *newBroadcast = rewriter.clone(*op, mapping);
      Type origResultType = broadcastOp.getResult().getType();
      newBroadcast->getResult(0).setType(
          updateTensorType(origResultType, tileShape));

      // 3. replace block arg and clean up
      blockArg.replaceAllUsesWith(newBroadcast->getResult(0));
      body->eraseArgument(numIV + i);
      newIns.erase(newIns.begin() + i);

      rewriter.modifyOpInPlace(
          genericOp, [&]() { genericOp.getInsMutable().assign(newIns); });

      return success();
    }

    return failure();
  }
};

struct FuseExpandDimsIntoGeneric
    : public mlir::OpRewritePattern<cpu::GenericOp> {
  using OpRewritePattern<cpu::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cpu::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    Block *body = &genericOp.getBody().front();
    unsigned numIV = genericOp.getNumInductionVars();

    for (auto [i, insVal] : llvm::enumerate(genericOp.getIns())) {
      Operation *op = insVal.getDefiningOp();
      if (!op)
        continue;

      auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op);
      if (!expandDimsOp)
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = cast<RankedTensorType>(blockArg.getType());
      SmallVector<int32_t> tileShape(tiledType.getShape());
      SmallVector<Value> newIns(genericOp.getIns());

      unsigned axis = expandDimsOp.getAxis();
      assert(tileShape[axis] == 1 &&
             "expected expand dims axis tile shape to be 1");

      SmallVector<int32_t> sourceTileShape;
      for (auto [j, t] : llvm::enumerate(tileShape)) {
        if (j == axis)
          continue;
        sourceTileShape.push_back(t);
      }

      RankedTensorType sourceTensorType =
          cast<RankedTensorType>(expandDimsOp.getSrc().getType());
      IRMapping mapping;
      // 1. map src operand to block args
      newIns.push_back(expandDimsOp.getSrc());
      mapping.map(
          expandDimsOp.getSrc(),
          body->addArgument(updateTensorType(sourceTensorType, sourceTileShape),
                            expandDimsOp.getSrc().getLoc()));

      // 2. clone expand dims
      rewriter.setInsertionPointToStart(body);
      Operation *newExpandDims = rewriter.clone(*op, mapping);
      Type origResultType = expandDimsOp.getResult().getType();
      newExpandDims->getResult(0).setType(
          updateTensorType(origResultType, tileShape));

      blockArg.replaceAllUsesWith(newExpandDims->getResult(0));
      body->eraseArgument(numIV + i);
      newIns.erase(newIns.begin() + i);

      rewriter.modifyOpInPlace(
          genericOp, [&]() { genericOp.getInsMutable().assign(newIns); });

      return success();
    }

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

    // Step 1: Create the generic ops
    patterns.add<WrapStores>(context, benefitDefault + 1);
    patterns.add<WrapReduceOp>(context, benefitDefault + 1);
    patterns.add<WrapKLoopWithDotOp>(context, benefitDefault + 1);
    patterns.add<WrapConvertLayoutOp>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    // Step 2: Fuse elementwise ops and loads into each generic
    RewritePatternSet fusePatterns(context);

    fusePatterns.add<FuseElementwiseIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseBroadcastIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseExpandDimsIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseMakeRangeIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseConstantIntoGeneric>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(fusePatterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
