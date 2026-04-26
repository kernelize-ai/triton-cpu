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
    const bool srcIsLoad = isa<LoadOp>(srcs[0].getDefiningOp());
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
