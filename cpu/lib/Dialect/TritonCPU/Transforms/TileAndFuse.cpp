#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "triton/Analysis/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"

#include <numeric>

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
  auto memDescType = dyn_cast<gpu::MemDescType>(t);
  if (memDescType) {
    // for memdesc tensor types we need to update the memdesc shape and the
    // encoding. note that only shared linear encoding is supported.
    auto shape = memDescType.getShape();
    if (llvm::all_of(llvm::zip(shape, tileShape), [](auto p) {
          auto [s, t] = p;
          return s == t;
        })) {
      return t;
    }

    auto encoding =
        cast<gpu::SharedLinearEncodingAttr>(memDescType.getEncoding());
    auto layout = encoding.getLinearLayout();
    assert(layout.getNumOutDims() == tileShape.size());

    // TODO: this breaks the surjectivity of the layout. do we care that the in
    // dims don't get updated? update the layout in dim size to equal the number
    // of elements in the tensor layout =
    // layout.resizeInDim(mlir::StringAttr::get(t.getContext(), "offset"),
    // std::accumulate(tileShape.begin(), tileShape.end(), 1,
    // std::multiplies<int32_t>()));
    auto dims = standardOutDimNames(t.getContext(), tileShape.size());
    for (auto [idx, dim] : llvm::enumerate(dims)) {
      layout = layout.resizeOutDim(dim, tileShape[idx]);
    }
    llvm::errs() << "new layout = " << layout << "\n";
    auto newEncoding = gpu::SharedLinearEncodingAttr::get(
        t.getContext(), layout, encoding.getLayoutAlignment());
    SmallVector<int64_t> tileShape64(tileShape.begin(), tileShape.end());
    return gpu::MemDescType::get(tileShape64, memDescType.getElementType(),
                                 newEncoding, memDescType.getMemorySpace(),
                                 memDescType.getMutableMemory());
  }
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

static SmallVector<Value>
buildBlockShapeValues(Location loc, ArrayRef<int32_t> blockShape,
                      mlir::PatternRewriter &rewriter) {
  return llvm::map_to_vector(blockShape, [&](int32_t s) {
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getI32IntegerAttr(s))
        .getResult();
  });
}

// Note: this helper assumes the conversion only involves reordering of
// registers
static std::optional<SmallVector<int32_t>>
getBlockedRegisterConversionTileShape(RankedTensorType srcTy,
                                      RankedTensorType dstTy) {
  // only support blocked -> blocked conversions
  auto srcEnc = dyn_cast<gpu::BlockedEncodingAttr>(srcTy.getEncoding());
  if (!srcEnc)
    return std::nullopt;

  auto srcSPT = srcEnc.getSizePerThread();

  SmallVector<unsigned> dstOrder;
  auto dstEnc = dyn_cast<gpu::BlockedEncodingAttr>(dstTy.getEncoding());
  if (!dstEnc) {
    auto dstDotEncoding =
        dyn_cast<gpu::DotOperandEncodingAttr>(dstTy.getEncoding());
    if (dstDotEncoding) {
      dstEnc = dyn_cast<gpu::BlockedEncodingAttr>(dstDotEncoding.getParent());
      dstOrder = gpu::getOrderForDotOperand(dstDotEncoding.getOpIdx(),
                                            dstTy.getRank(), /*kContig*/ false);
    }
  }
  if (!dstEnc)
    return std::nullopt;

  if (dstOrder.empty())
    dstOrder = llvm::to_vector(dstEnc.getOrder());

  // Both getSizePerThread() arrays are tensor-dimension indexed, so the LCM
  // per tensor dimension is computed by direct positional comparison.
  auto dstSPT = dstEnc.getSizePerThread();
  assert(srcSPT.size() == dstSPT.size());

  auto shape = srcTy.getShape();
  unsigned rank = shape.size();

  // Compute LCM of src and dst sizePerThread per tensor dimension.
  SmallVector<int32_t> tileNaive(rank);
  for (unsigned dim = 0; dim < rank; dim++) {
    int32_t tile = std::lcm((int32_t)srcSPT[dim], (int32_t)dstSPT[dim]);
    if (shape[dim] % tile != 0)
      return std::nullopt;
    tileNaive[dim] = tile;
  }

  LDBG("Tile shape without order: " << triton::join(tileNaive));

  // Permute tileNaive by dstOrder to produce a tile shape in the dst
  // encoding's storage-order space. Callers compare this result positionally
  // against the outer generic's tileShape (which is also order-indexed via
  // sizePerThread), so the permutation is required for the fits check to be
  // correct. Returning tileNaive directly would misalign the dimensions.
  SmallVector<int32_t> tileShape(rank);
  for (unsigned i = 0; i < rank; i++)
    tileShape[i] = tileNaive[dstOrder[i]];

  LDBG("Tile shape after applying dst order: " << triton::join(tileShape));

  return tileShape;
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
    auto [v, inputTileShape] = pair;
    Type argTy = updateTensorType(v.getType(), inputTileShape);
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
    SmallVector<Value> blockShapeValues =
        buildBlockShapeValues(loc, blockShape, rewriter);

    auto generic =
        cpu::GenericOp::create(rewriter, loc, /*resultTypes=*/TypeRange{},
                               insValues, blockShapeValues, tileShape);

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
    SmallVector<Value> blockShapeValues =
        buildBlockShapeValues(loc, blockShape, rewriter);
    auto generic = cpu::GenericOp::create(rewriter, loc, resultTypes, insValues,
                                          blockShapeValues, tileShape);

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

struct WrapDotOp : public mlir::OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = dotOp.getLoc();

    if (dotOp->getParentOfType<cpu::GenericOp>())
      return failure();

    // Dot result must have BlockedEncoding so we can derive
    // blockShape/vectorShape.
    auto resultTy = dyn_cast<RankedTensorType>(dotOp.getType());
    if (!resultTy || !isa<gpu::BlockedEncodingAttr>(resultTy.getEncoding()))
      return failure();

    auto encoding = cast<gpu::BlockedEncodingAttr>(resultTy.getEncoding());

    // use the MxN (result) shape for block/tile shapes. The K loop is not
    // currently tiled.
    auto [blockShape, tileShape] = getBlockAndTileShapes(resultTy, encoding);

    auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
    assert(aTy.getRank() == 2 && "only 2D dot op supported");
    int32_t kSize = (int32_t)aTy.getShape()[1];

    SmallVector<int32_t> aTileShape = {tileShape[0], kSize};
    SmallVector<int32_t> bTileShape = {kSize, tileShape[1]};

    SmallVector<TiledInput> ins;
    ins.push_back(TiledInput{dotOp.getA(), aTileShape});
    ins.push_back(TiledInput{dotOp.getB(), bTileShape});
    ins.push_back(TiledInput{dotOp.getC(), tileShape});
    SmallVector<Value> insValues =
        llvm::map_to_vector(ins, [](const TiledInput &ti) { return ti.value; });

    SmallVector<Value> blockShapeValues =
        buildBlockShapeValues(loc, blockShape, rewriter);

    auto generic = cpu::GenericOp::create(rewriter, loc, {resultTy}, insValues,
                                          blockShapeValues, tileShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, tileShape, bodyMapping);

    // TODO: if we have a materialized accumulator, clone that chain too
    // clone the dot op
    auto *newDot = rewriter.clone(*dotOp, bodyMapping);
    newDot->getResult(0).setType(updateTensorType(resultTy, tileShape));
    cpu::YieldOp::create(rewriter, loc, newDot->getResults());

    rewriter.replaceOp(dotOp, generic.getResults());
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

    auto src = cvtOp.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(cvtOp.getType());

    // src encodings are validated in getBlockedRegisterConversionTileShape, but
    // dst encodings other than blocked are allowed. Disable those in standalone
    // generics as we can usually fuse blocked -> non-blocked cvts (i.e. blocked
    // -> dot_op)
    if (!isa<gpu::BlockedEncodingAttr>(dstTy.getEncoding()))
      return failure();

    auto layout = minimalCvtLayout(srcTy, dstTy);
    auto outDims = to_vector(layout.getOutDimNames());
    MLIRContext *ctx = srcTy.getContext();
    auto kRegister = StringAttr::get(ctx, "register");
    // only wrap cvts that reorder registers. if the cvt does nothing (empty out
    // dims), return failure as we should be able to fuse it
    if (outDims.empty() || (ArrayRef(outDims) != ArrayRef({kRegister})))
      return failure();

    LDBG("Getting required tile shape for " << cvtOp);

    //  get blocked register conversion tile shape
    auto requiredTileShape =
        getBlockedRegisterConversionTileShape(srcTy, dstTy);
    if (!requiredTileShape)
      return failure();

    // Use the destination ty to get the tile shape. the destination ty will be
    // tiled in any cvt evaluated for fusion, so we want to use the same
    // criteria for "fits" to avoid wrapping fusible cvts
    auto [blockShape, defaultTileShape] = getBlockAndTileShapes(
        dstTy, cast<gpu::BlockedEncodingAttr>(dstTy.getEncoding()));

    // return failure as we should be able to fuse this cvt op
    if (ArrayRef(*requiredTileShape) == ArrayRef(defaultTileShape))
      return failure();

    SmallVector<int32_t> tileShape = blockShape;
    SmallVector<TiledInput> ins;
    for (auto value : cvtOp->getOperands()) {
      ins.push_back(TiledInput{value, tileShape});
    }

    SmallVector<Value> insValues =
        llvm::map_to_vector(ins, [](const TiledInput &ti) { return ti.value; });
    SmallVector<Value> blockShapeValues =
        buildBlockShapeValues(loc, blockShape, rewriter);
    auto generic =
        cpu::GenericOp::create(rewriter, loc, /*resultTypes=*/TypeRange{dstTy},
                               insValues, blockShapeValues, tileShape);

    IRMapping bodyMapping;
    initGenericBody(rewriter, generic, ins, tileShape, bodyMapping);

    auto newCvt = rewriter.clone(*cvtOp, bodyMapping);
    newCvt->getResult(0).setType(updateTensorType(dstTy, tileShape));
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

      if (!mlir::isMemoryEffectFree(op) && op->getBlock() != genericOp->getBlock())
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = dyn_cast<RankedTensorType>(blockArg.getType());
      SmallVector<int32_t> tileShape;
      if (!tiledType) {
        const bool hasTensorOperands =
            llvm::any_of(op->getOperands(), [&](Value operand) {
              return isa<RankedTensorType>(operand.getType());
            });
        const bool isArithElementwise =
            (isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) &&
            op->hasTrait<OpTrait::Elementwise>();

        // fuse scalar elementwise ops too
        if (hasTensorOperands || !isArithElementwise)
          continue;
      } else {
        tileShape = llvm::map_to_vector(tiledType.getShape(), [](int64_t dim) {
          return static_cast<int32_t>(dim);
        });
      }

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
      if (op->getBlock() != genericOp->getBlock())
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
        auto rank = tileShape.size();
        auto newResultType = updateTensorType(resultType, tileShape);
        auto sliceEncodingAttr =
            dyn_cast<triton::gpu::SliceEncodingAttr>(resultType.getEncoding());
        if (!sliceEncodingAttr) {
          // check to see if a slice encoding attr exists downstream from a cvt
          // op that was previously rematerialized
          ForwardSliceOptions fwdOpt;
          fwdOpt.filter = [&](Operation *op) {
            auto cvtInSlice = dyn_cast<gpu::ConvertLayoutOp>(op);
            if (cvtInSlice) {
              auto localSliceEncodingAttr = dyn_cast<gpu::SliceEncodingAttr>(
                  cast<RankedTensorType>(cvtInSlice.getType()).getEncoding());
              if (localSliceEncodingAttr) {
                sliceEncodingAttr = localSliceEncodingAttr;
                return false;
              }
            }
            return true;
          };
          SetVector<Operation *> slice;
          getForwardSlice(blockArg, &slice, fwdOpt);
        }
        unsigned dim;
        if (!sliceEncodingAttr) {
          assert(rank == 1 && "make dynamic range op without slice encoding "
                              "should be inside a 1D generic");
          dim = 0;
        } else {
          // Body induction vars are ordered outermost-first, matching
          // blockShape. A SliceEncodingAttr with dim=d encodes a 1D tensor in
          // the direction of tensor-dim (1-d),
          // which sits at blockShape index (rank-2) + (1-d) = rank-1-d.
          dim =
              genericOp.getBlockShape().size() - 1 - sliceEncodingAttr.getDim();
        }
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

struct FuseConvertLayoutOpIntoGeneric : mlir::OpRewritePattern<cpu::GenericOp> {
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

      auto cvtOp = dyn_cast<gpu::ConvertLayoutOp>(op);
      if (!cvtOp)
        continue;

      LDBG("Evaluate cvt for fusion: " << cvtOp);
      auto srcTy = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(cvtOp.getType());

      // get the existing tile type
      BlockArgument blockArg = body->getArgument(numIV + i);
      auto tiledType = cast<RankedTensorType>(blockArg.getType());

      SmallVector<int32_t> tileShape(tiledType.getShape());

      // determine if the register shuffle required fits inside our generic
      // Note: the logic here is very similar to cvtReordersRegisters
      auto layout = minimalCvtLayout(srcTy, dstTy);
      auto outDims = to_vector(layout.getOutDimNames());

      if (!outDims.empty()) {
        LDBG("Non-empty out dims, layout: " << layout);
        MLIRContext *ctx = srcTy.getContext();
        auto kRegister = StringAttr::get(ctx, "register");

        // layout must only reorder registers
        if (ArrayRef(outDims) != ArrayRef({kRegister}))
          continue;

        // must be able to determine the required tile shape and it must fit
        // inside this generic
        auto requiredTileShape =
            getBlockedRegisterConversionTileShape(srcTy, dstTy);
        if (!requiredTileShape)
          continue;

        auto required = *requiredTileShape;
        LDBG("Required tile shape for register shuffle: "
             << triton::join(required));
        bool fits = llvm::all_of(llvm::zip(tileShape, required), [](auto pair) {
          auto [cur, req] = pair;
          return cur % req == 0;
        });
        if (!fits)
          continue;

        LDBG("Cvt can be legally fused");
      }

      SmallVector<Value> newIns(genericOp.getIns());

      IRMapping mapping;
      // 1. Add new block args for source op inputs at body end
      newIns.push_back(cvtOp.getSrc());
      mapping.map(cvtOp.getSrc(),
                  body->addArgument(updateTensorType(srcTy, tileShape),
                                    cvtOp.getSrc().getLoc()));

      // 2. clone
      rewriter.setInsertionPointToStart(body);
      Operation *newOp = rewriter.clone(*op, mapping);
      newOp->getResult(0).setType(updateTensorType(dstTy, tileShape));

      // 3. replace block arg and clean up
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

struct FuseLocalAllocIntoGeneric : mlir::OpRewritePattern<cpu::GenericOp> {
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

      auto localAllocOp = dyn_cast<gpu::LocalAllocOp>(op);
      if (!localAllocOp)
        continue;

      if (op->getBlock() != genericOp->getBlock())
        continue;

      BlockArgument blockArg = body->getArgument(numIV + i);

      auto memDescType = cast<gpu::MemDescType>(blockArg.getType());
      SmallVector<int32_t> tileShape(memDescType.getShape());
      SmallVector<Value> newIns(genericOp.getIns());

      IRMapping mapping;
      // 1. Add new block args for source op inputs at body end
      if (auto src = localAllocOp.getSrc()) {
        auto srcTy = cast<RankedTensorType>(src.getType());
        newIns.push_back(src);
        mapping.map(src, body->addArgument(updateTensorType(srcTy, tileShape),
                                           src.getLoc()));
      }

      // 2. clone
      rewriter.setInsertionPointToStart(body);
      Operation *newOp = rewriter.clone(*op, mapping);
      newOp->getResult(0).setType(
          updateTensorType(localAllocOp.getType(), tileShape));

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

struct FuseParentForOpIntoGeneric : mlir::OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  std::optional<cpu::GenericOp> findTargetGenericOp(scf::ForOp forOp) const {
    // (1) For loop must have iter args — excludes persistent block-dispatch
    // loops which have no iter args.
    if (forOp.getNumRegionIterArgs() == 0)
      return std::nullopt;

    // (2) For step must be the constant 1 (can probably relax this later)
    auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantOp>();
    if (!stepCst || cast<IntegerAttr>(stepCst.getValue()).getInt() != 1)
      return std::nullopt;

    Block *forBody = forOp.getBody();

    // (3) Match a very specific genericOp/addPtr pattern. Again, this can
    // probably be relaxed as we see more examples that are good candidates
    // for fusion
    cpu::GenericOp genericOp;
    bool bodyValid = true;
    for (Operation &bodyOp : forBody->without_terminator()) {
      if (auto bodyGenericOp = dyn_cast<cpu::GenericOp>(bodyOp)) {
        if (genericOp) {
          bodyValid = false; // >1 generic
          break;
        }
        genericOp = bodyGenericOp;
      } else if (auto addptr = dyn_cast<triton::AddPtrOp>(bodyOp)) {
        if (!llvm::is_contained(forOp.getRegionIterArgs(), addptr.getPtr())) {
          bodyValid = false;
          break;
        }
      } else if (auto localStoreOp = dyn_cast<gpu::LocalStoreOp>(bodyOp)) {
        if (localStoreOp.getSrc().getDefiningOp() != genericOp) {
          bodyValid = false;
          break;
        }
      } else {
        bodyValid = false;
        break;
      }
    }
    if (!bodyValid)
      return std::nullopt;
    if (!genericOp)
      return std::nullopt;

    // (4) Every scf.yield operand must come from either the inner generic
    // or an addptr that advances a for iter arg.
    auto yieldOp = cast<scf::YieldOp>(forBody->getTerminator());
    bool yieldValid = llvm::all_of(yieldOp.getOperands(), [&](Value v) {
      Operation *def = v.getDefiningOp();
      if (!def)
        return false;
      if (def == genericOp.getOperation())
        return true;
      if (auto addptr = dyn_cast<triton::AddPtrOp>(def))
        return llvm::is_contained(forOp.getRegionIterArgs(), addptr.getPtr());
      return false;
    });
    if (!yieldValid)
      return std::nullopt;

    return genericOp;
  }

  LogicalResult
  matchAndRewrite(scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {
    // check fusion criteria
    auto targetGenericOp = findTargetGenericOp(forOp);
    if (!targetGenericOp)
      return failure();

    cpu::GenericOp genericOp = *targetGenericOp;

    unsigned numIV = genericOp.getNumInductionVars();
    Block &genericBody = genericOp.getBody().front();

    // 1. forward forOp upper/lower/step args and init args through the generic
    // op so the cloned forOp can re-use them

    SmallVector<Value> newIns(genericOp.getIns());
    SmallVector<Value> newForControlOperands;
    SmallVector<Value>
        newForInitVals; // tiled init values for the fused for loop

    IRMapping mapping;
    for (auto [i, forOpOperand] : llvm::enumerate(forOp.getOperands())) {
      if (i > forOp.getNumControlOperands() + forOp.getNumRegionIterArgs())
        break; // only consider control operands args and init args

      if (i < forOp.getNumControlOperands()) {
        BlockArgument bodyArg = genericBody.addArgument(forOpOperand.getType(),
                                                        forOpOperand.getLoc());
        newIns.push_back(forOpOperand); // forward control operands to generic
                                        // op directly, no need to update types
        newForControlOperands.push_back(bodyArg);
      } else {
        auto forOpIterArg =
            forOp.getRegionIterArg(i - forOp.getNumControlOperands());
        // use the for op iter arg to lookup the tiling for this operand
        for (auto [j, operand] : llvm::enumerate(genericOp.getIns())) {
          if (operand == forOpIterArg) {

            auto existingGenericBodyArg = genericBody.getArgument(numIV + j);
            SmallVector<int32_t> tileShape;
            if (auto rankedTensorType = dyn_cast<RankedTensorType>(
                    existingGenericBodyArg.getType()))
              tileShape = llvm::map_to_vector(
                  rankedTensorType.getShape(),
                  [](int64_t dim) { return static_cast<int32_t>(dim); });

            BlockArgument bodyArg = genericBody.addArgument(
                updateTensorType(forOpOperand.getType(), tileShape),
                forOpOperand.getLoc());
            mapping.map(forOpIterArg, bodyArg);
            newIns.push_back(forOpOperand);
            newForInitVals.push_back(bodyArg);
            break;
          }
        }
      }
    }

    // 2. clone the for op into the generic body

    // Snapshot old body ops before inserting newFor
    SmallVector<Operation *> oldBodyOps;
    for (Operation &op : genericBody.without_terminator())
      oldBodyOps.push_back(&op);

    // TODO: let's make sure we check to see that the generic op is the first op
    // in the for op body
    rewriter.setInsertionPointToStart(&genericBody);
    assert(newForControlOperands.size() == forOp.getNumControlOperands() &&
           "expected to forward all for op control operands to generic");
    auto newFor = scf::ForOp::create(
        rewriter, forOp.getLoc(), newForControlOperands[0],
        newForControlOperands[1], newForControlOperands[2], newForInitVals);

    // 3. map old for-arg body args to new iter args / induction vars

    // Map all generic body args that received the for IV → new for IV.
    // The IV may appear multiple times in the generic's ins (no break).
    mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
    for (auto [j, operand] : llvm::enumerate(genericOp.getIns()))
      if (operand == forOp.getInductionVar())
        mapping.map(genericBody.getArgument(numIV + j),
                    newFor.getInductionVar());

    // Map generic body args at iter-arg positions → new for iter args.
    // Also map the for iter args themselves → new for iter args so that ops
    // cloned from the old for body (e.g. addptr advancing pointers) resolve
    // to the current-iteration value rather than the initial value.
    for (auto [i, forOpIterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      for (auto [j, operand] : llvm::enumerate(genericOp.getIns())) {
        if (operand == forOpIterArg) {
          mapping.map(genericBody.getArgument(numIV + j),
                      newFor.getRegionIterArgs()[i]);
        }
      }
      mapping.map(forOpIterArg, newFor.getRegionIterArgs()[i]);
    }

    // 4. clone body ops
    rewriter.setInsertionPointToStart(newFor.getBody());
    // Clone old generic body ops (without yield)
    for (Operation *op : oldBodyOps)
      rewriter.clone(*op, mapping);

    // map generics results to cloned values so the other for body ops can
    // reference them
    auto genericYield = cast<cpu::YieldOp>(genericBody.getTerminator());
    for (auto [genericResult, yieldOperand] :
         llvm::zip(genericOp.getResults(), genericYield.getValues()))
      mapping.map(genericResult, mapping.lookup(yieldOperand));

    // Clone addptr ops from the old for body
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<cpu::GenericOp>(op)) {
        SmallVector<int32_t> tileShape;
        for (auto operand : op.getOperands()) {
          if (mapping.contains(operand)) {
            auto mapped = mapping.lookup(operand);
            if (auto mappedTensorTy =
                    dyn_cast<RankedTensorType>(mapped.getType())) {
              // assuming all tensor operands have the same shape
              tileShape = llvm::map_to_vector(
                  mappedTensorTy.getShape(),
                  [](int64_t dim) { return static_cast<int32_t>(dim); });
              break;
            }
          }
        }
        if (!tileShape.empty()) {
          for (auto operand : op.getOperands()) {
            if (mapping.contains(operand))
              continue; // already mapped (e.g. for iter arg) — don't re-add
            newIns.push_back(operand);
            mapping.map(operand,
                        genericBody.addArgument(
                            updateTensorType(operand.getType(), tileShape),
                            operand.getLoc()));
          }
        }
        Operation *newOp = rewriter.clone(op, mapping);
        if (newOp->getNumResults() > 0) {
          assert(newOp->getNumResults() == 1 &&
                 "expected cloned for op body ops to have only 1 result");
          if (!tileShape.empty())
            newOp->getResult(0).setType(
                updateTensorType(newOp->getResult(0).getType(), tileShape));
        } else {
          // TODO: can we update op type generically?
          auto localStoreOp = cast<gpu::LocalStoreOp>(op);
          // TODO: update type

          // TODO: we need to update the memdesc type for these stores,
          // unfortunately. this is no good: fortunately, we can probbably apply
          // the same fix to handling the tiled types like for the accumulator
          // allocator though this might be tricky - what happens if the local
          // load tensor size does not match the alloc? I guess this is actually
          // not a problem -- we can resize the accumulator for one iteration of
          // the generic and the inter-loop boundaries will handle writing back
          // to the outputs. so yeah - support memdesc updates - in update
          // tensor type maybe?
          /**
           *
            %218 = "ttg.local_load"(%arg79) : (!ttg.memdesc<4x16xf32, #shared,
          #smem, mutable>) -> tensor<4x4xf32, #blocked> %219 = "tt.dot"(%217,
          %208, %218) <{inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}>
          : (tensor<4x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
          tensor<8x4xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
          tensor<4x4xf32, #blocked>) -> tensor<4x4xf32, #blocked>
          "ttg.local_store"(%219, %arg86) : (tensor<4x4xf32, #blocked>,
          !ttg.memdesc<4x16xf32, #shared, #smem, mutable>) -> ()
           *
           */
        }
      }
    }

    // clone the for op yield
    Block *forBody = forOp.getBody();
    rewriter.clone(*forBody->getTerminator(), mapping);

    // 5. build the generic ttc.yield op
    scf::YieldOp oldForYield = cast<scf::YieldOp>(forBody->getTerminator());
    auto oldGenericYield = cast<cpu::YieldOp>(genericBody.getTerminator());
    SmallVector<Value> newGenericYieldVals;
    for (auto [j, oldResult] : llvm::enumerate(genericOp.getResults())) {
      for (auto [i, operand] : llvm::enumerate(oldForYield.getOperands())) {
        if (operand == oldResult) {
          newGenericYieldVals.push_back(newFor.getResult(i));
          break;
        }
      }
    }
    rewriter.setInsertionPoint(oldGenericYield);
    rewriter.replaceOpWithNewOp<cpu::YieldOp>(oldGenericYield,
                                              newGenericYieldVals);

    // 6. clean up
    // erase old body ops
    for (Operation *op : llvm::reverse(oldBodyOps))
      rewriter.eraseOp(op);

    // drop generic ins operands that come from the old for op
    SetVector<unsigned> operandsToDrop;
    for (auto arg : forOp.getBody()->getArguments()) {
      for (auto [i, operand] : llvm::enumerate(genericOp.getIns())) {
        if (operand == arg)
          operandsToDrop.insert(i);
      }
    }
    SmallVector<unsigned> sortedToDrop(operandsToDrop.begin(),
                                       operandsToDrop.end());
    llvm::sort(sortedToDrop, std::greater<unsigned>());
    for (auto idx : sortedToDrop) {
      genericBody.eraseArgument(numIV + idx);
      newIns.erase(newIns.begin() + idx);
    }

    rewriter.modifyOpInPlace(
        genericOp, [&]() { genericOp.getInsMutable().assign(newIns); });

    rewriter.moveOpBefore(genericOp, forOp);

    // update for op users to use the generic op results. if the pattern match
    // is correct, this should result in the existing for op being unused
    for (auto [j, newForResult] : llvm::enumerate(newGenericYieldVals)) {
      unsigned i = cast<OpResult>(newForResult).getResultNumber();
      rewriter.replaceAllUsesWith(forOp.getResult(i), genericOp.getResult(j));
    }
    rewriter.eraseOp(forOp);

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
    patterns.add<WrapStores>(context, benefitDefault + 1);
    patterns.add<WrapReduceOp>(context, benefitDefault + 1);
    patterns.add<WrapDotOp>(context, benefitDefault + 1);
    patterns.add<WrapConvertLayoutOp>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    LDBG("Module before fusion " << m);

    // Step 2: Fuse ops into each generic
    RewritePatternSet fusePatterns(context);

    fusePatterns.add<FuseElementwiseIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseBroadcastIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseExpandDimsIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseMakeRangeIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseConstantIntoGeneric>(context, benefitDefault);
    fusePatterns.add<FuseConvertLayoutOpIntoGeneric>(context, benefitDefault);
    // fusePatterns.add<FuseLocalAllocIntoGeneric>(context, benefitDefault);

    fusePatterns.add<FuseParentForOpIntoGeneric>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(fusePatterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
