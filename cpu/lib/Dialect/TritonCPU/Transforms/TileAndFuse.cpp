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

// Create the body block of a GenericOp, adding one block arg per ins value
// with tensor types replaced to the vector (chunk) shape. Populates `mapping`
// with ins value → block arg entries and sets the insertion point to the start
// of the block. Returns the new block.
static Block *initGenericBody(OpBuilder &rewriter, cpu::GenericOp generic,
                              ArrayRef<Value> ins, ArrayRef<int32_t> tileShape,
                              IRMapping &mapping) {
  Block *body = rewriter.createBlock(&generic.getBody());
  body->addArgument(rewriter.getI32Type(), generic.getLoc()); // chunk offset
  for (Value v : ins) {
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

    SmallVector<Value> ins(storeOp->getOperands().begin(),
                           storeOp->getOperands().end());

    auto generic = cpu::GenericOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, ins, blockShape, tileShape);

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

    SmallVector<Value> ins = {srcs[0]};
    SmallVector<Type> resultTypes(reduceOp.getResultTypes().begin(),
                                  reduceOp.getResultTypes().end());
    if (srcUsedElsewhere)
      resultTypes.push_back(tensorTy);

    LDBG("Creating reduction generic op, result types: " << resultTypes.size());

    auto generic = cpu::GenericOp::create(rewriter, loc, resultTypes, ins,
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

    // Collect all values the for loop captures from outside its scope.
    //   1. The forOp's explicit operands (lb, ub, step, init_args).
    //   2. Free variables: values used by ops inside the body that are
    //      defined outside the for op (implicit captures).
    auto isDefinedInsideForOp = [&](Value v) -> bool {
      if (auto blockArg = dyn_cast<BlockArgument>(v))
        return forOp->isAncestor(blockArg.getOwner()->getParentOp());
      return forOp->isAncestor(v.getDefiningOp());
    };

    SetVector<Value> capturedSet;
    for (Value v : forOp.getOperands())
      capturedSet.insert(v);
    forOp.getBody()->walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands())
        if (!isDefinedInsideForOp(operand))
          capturedSet.insert(operand);
    });
    SmallVector<Value> ins(capturedSet.begin(), capturedSet.end());

    auto generic = cpu::GenericOp::create(rewriter, loc, TypeRange{resultTy},
                                          ins, blockShape, tileShape);

    llvm::errs() << "generic: " << generic << "\n";
    assert(false && "TODO");
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

      auto resultType = makeRangeOp.getResult().getType();
      auto newResultType = updateTensorType(resultType, tileShape);
      cpu::MakeDynamicRangeOp makeDynamicRangeOp =
          triton::cpu::MakeDynamicRangeOp::create(
              rewriter, makeRangeOp.getLoc(), newResultType,
              genericOp.getChunkOffset());

      // 3. update existing uses
      blockArg.replaceAllUsesWith(makeDynamicRangeOp.getResult());
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
    RewritePatternSet fusePatterns(context);

    fusePatterns.add<FuseElementwiseIntoGeneric>(context, benefitDefault);
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
