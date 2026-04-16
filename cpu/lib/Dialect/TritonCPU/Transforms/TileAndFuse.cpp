#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

// TODO: rmeove if we use utility to load the analysis
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "triton/Analysis/Utility.h"

#define DEBUG_TYPE "tritoncpu-tile-and-fuse"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TRITONCPUTILEANDFUSE
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

// Each dimension of a ranked tensor is mapped to either:
//   Some(i) — this dim corresponds to tile dimension i (i.e. vectorShape[i])
//   None    — this dim is untiled (full size)
class TileInfo {
public:
  using TileShapeT = SmallVector<int32_t>;

  TileInfo() = default; // uninitialized (bottom)
  explicit TileInfo(TileShapeT tiledShape)
      : tiledShape(std::move(tiledShape)), initialized(true) {}

  static TileInfo getNoTile() {
    TileInfo t;
    t.initialized = true;
    return t;
  }

  static TileInfo getPessimisticValueState(Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t)
      return getNoTile();
    return TileInfo(TileShapeT(t.getShape()));
  }

  static TileInfo join(const TileInfo &lhs, const TileInfo &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs.isEmpty())
      return rhs;
    if (rhs.isEmpty())
      return lhs;
    assert(lhs.tiledShape.size() == rhs.tiledShape.size());
    TileShapeT result;
    for (auto [l, r] : llvm::zip(lhs.tiledShape, rhs.tiledShape))
      result.push_back(
          std::min(l, r)); // min: 0 (not-tiled) wins over any tile size
    // TODO: what about if l ==0 || r == 0 ? 0 : max()? or maybe we should just
    // pick 0 or assert equality
    // TODO: also why do we have both empty and 0?
    return TileInfo(result);
  }

  bool operator==(const TileInfo &other) const {
    return initialized == other.initialized && tiledShape == other.tiledShape;
  }

  // decltype?
  ArrayRef<int32_t> getTiledShape() const { return tiledShape; }

  int32_t getShapeForDim(const unsigned index) const {
    return tiledShape[index];
  }

  bool isUninitialized() const { return !initialized; }
  bool isEmpty() const { return initialized && tiledShape.empty(); }
  unsigned getRank() const { return tiledShape.size(); }

  bool dimIsTiled(const unsigned dim) const {
    assert(!isUninitialized() && dim < tiledShape.size());
    return tiledShape[dim] != 0;
  }

  // initialized but not tiled returns true, all other returns false
  bool isNotTiled() const {
    return llvm::all_of(tiledShape, [](int64_t t) { return t == 0; });
  }

  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "UNINITIALIZED";
      return;
    }
    os << "[";
    llvm::interleaveComma(tiledShape, os);
    os << "]";
  }

private:
  TileShapeT tiledShape;
  bool initialized = false;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const TileInfo &ti) {
  ti.print(os);
  return os;
}

class TileInfoAnalysis : public dataflow::SparseBackwardDataFlowAnalysis<
                             dataflow::Lattice<TileInfo>> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<dataflow::Lattice<TileInfo> *> operands,
      ArrayRef<const dataflow::Lattice<TileInfo> *> results) override;

protected:
  void visitCallOperand(OpOperand &operand) override {}

  void propagateToYield(scf::YieldOp yieldOp, SmallVector<TileInfo> &lattices) {
    for (auto [lattice, yieldOperand] :
         llvm::zip_equal(lattices, yieldOp->getOperands())) {
      auto yieldLattice = getLatticeElement(yieldOperand);
      ChangeResult changed = yieldLattice->join(lattice);
      propagateIfChanged(yieldLattice, changed);
    }
  }

  void visitBranchOperand(OpOperand &operand) override {
    auto defOp = operand.getOwner();
    assert(isa<scf::IfOp>(defOp) || isa<scf::ForOp>(defOp));

    SmallVector<TileInfo> lattices(defOp->getNumResults(), TileInfo());
    for (auto [i, result] : llvm::enumerate(defOp->getResults())) {
      auto resultLattice = getLatticeElement(result);
      // Wait for all the results to be initialized.
      if (resultLattice->getValue().isUninitialized())
        return;
      lattices[i] = resultLattice->getValue().join(lattices[i],
                                                   resultLattice->getValue());
    }

    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      propagateToYield(yieldOp, lattices);
      // Also propagate result lattices back to the iter_arg initial values so
      // that tile shapes escape the loop to upstream chains. Use
      // propagateTileShape (not bare propagateIfChanged) so that if an init
      // arg is a generic body block arg, the extra ins-value propagation fires
      // and the tile shape crosses the generic region boundary too.
      for (auto [i, lat] : llvm::enumerate(lattices)) {
        if (lat.isUninitialized() || lat.isEmpty())
          continue;
        auto *initLattice = getLatticeElement(forOp.getInitArgs()[i]);
        propagateTileShape(initLattice, llvm::to_vector(lat.getTiledShape()));
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      propagateToYield(ifOp.thenYield(), lattices);
      if (!ifOp.getElseRegion().empty())
        propagateToYield(ifOp.elseYield(), lattices);
    } else {
      llvm_unreachable("Unknown branch operation");
    }
    return;
  }

  void setToExitState(dataflow::Lattice<TileInfo> *lattice) override {
    // Non-tensors have no tile shape — mark initialized so they don't block
    // visitBranchOperand's wait-for-all-results check.
    if (!isa<RankedTensorType>(lattice->getAnchor().getType()))
      propagateIfChanged(lattice, lattice->join(TileInfo::getNoTile()));
    // Tensors: leave uninitialized; downstream demands will populate.
  }

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {
    Region *region = successor.getSuccessor();
    if (!region)
      return;
    auto genericOp = dyn_cast<cpu::GenericOp>(region->getParentOp());
    if (!genericOp)
      return;

    unsigned numInductionVars = genericOp.getNumInductionVars();
    for (auto [i, insVal] : llvm::enumerate(genericOp.getIns())) {
      unsigned argIdx = numInductionVars + i;
      if (argIdx >= arguments.size())
        break;
      auto *blockArgLattice = getLatticeElement(arguments[argIdx]);
      auto *insLattice = getLatticeElement(insVal);
      propagateIfChanged(insLattice,
                         insLattice->join(blockArgLattice->getValue()));
    }
  }

private:
  void propagateTileShape(dataflow::Lattice<TileInfo> *lattice,
                          TileInfo::TileShapeT shape) {
    propagateIfChanged(lattice, lattice->join(TileInfo(shape)));
    // If this is an scf.for iter_arg, also eagerly propagate to the
    // corresponding yield operand. visitOperation for scf.yield is only
    // triggered when yield's *results* change, but yield has no results —
    // so it is never re-triggered after iter_args are populated. We close
    // the loop-back edge here instead.
    if (auto blockArg = dyn_cast<BlockArgument>(lattice->getAnchor())) {
      Block *block = blockArg.getOwner();
      unsigned argIdx = blockArg.getArgNumber();

      // scf.for iter_arg: close the loop-back edge to the yield operand.
      // visitOperation for scf.yield is only triggered when yield's *results*
      // change, but yield has no results, so we eagerly propagate here.
      if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp())) {
        unsigned numIVs = forOp.getNumInductionVars();
        if (argIdx < numIVs)
          return; // induction variable, not an iter_arg
        unsigned iterIdx = argIdx - numIVs;
        // Close the loop-back edge: iter_arg → yield operand.
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        Value yieldOperand = yieldOp.getOperand(iterIdx);
        auto *yieldLattice = getLatticeElement(yieldOperand);
        propagateIfChanged(yieldLattice, yieldLattice->join(TileInfo(shape)));
        // Also propagate to the corresponding scf.for result so that
        // visitBranchOperand is triggered. Without this the K-loop results
        // stay UNINITIALIZED and visitBranchOperand (which waits for all
        // results) never fires, blocking propagation out of the loop.
        auto *resultLattice = getLatticeElement(forOp.getResult(iterIdx));
        propagateIfChanged(resultLattice, resultLattice->join(TileInfo(shape)));
        return;
      }

      // cpu::GenericOp body block arg: eagerly propagate to the corresponding
      // ins value. visitNonControlFlowArguments is only called at region-entry
      // initialisation and is not re-triggered when individual block arg
      // lattices change later, so the tile shape would otherwise never escape
      // the generic body to reach the ops outside it (e.g. make_range →
      // expand_dims → generic ins).
      if (auto genericOp = dyn_cast<cpu::GenericOp>(block->getParentOp())) {
        unsigned numIVs = genericOp.getNumInductionVars();
        if (argIdx < numIVs)
          return; // induction variable
        unsigned insIdx = argIdx - numIVs;
        if (insIdx >= genericOp.getIns().size())
          return;
        Value insVal = genericOp.getIns()[insIdx];
        auto *insLattice = getLatticeElement(insVal);
        propagateIfChanged(insLattice, insLattice->join(TileInfo(shape)));
      }
    }
  }
};

LogicalResult TileInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<dataflow::Lattice<TileInfo> *> operands,
    ArrayRef<const dataflow::Lattice<TileInfo> *> results) {

  if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
    TileInfo::TileShapeT tileShapeMN;
    if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("ttcpu.tile_shape")) {
      tileShapeMN.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
      // propagate the tile shape signaled by the attribute to dot op results
      auto *resLattice = getLatticeElement(dotOp.getResult());
      propagateIfChanged(resLattice, resLattice->join(TileInfo(tileShapeMN)));
    } else {
      // dot ops are anchor ops - propagate the result tiled shape to the
      // operands
      auto resTy = cast<RankedTensorType>(dotOp.getResult().getType());
      auto enc = dyn_cast<gpu::BlockedEncodingAttr>(resTy.getEncoding());
      if (!enc)
        return success();
      tileShapeMN.assign(enc.getSizePerThread()); // [M_tile, N_tile]
      assert(tileShapeMN.size() == 2 && "only rank 2 dot op results supported");
    }
    auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = cast<RankedTensorType>(dotOp.getB().getType());

    // A [M, K]: tile M from encoding, K is not tiled (reduction dim)
    propagateTileShape(operands[0], {tileShapeMN[0], 0});
    // B [K, N]: K is not tiled (reduction dim), tile N from encoding
    propagateTileShape(operands[1], {0, tileShapeMN[1]});
    // C [M, N]: accumulator has the same tile shape as the result
    if (operands.size() > 2)
      propagateTileShape(operands[2], {tileShapeMN[0], tileShapeMN[1]});
    return success();
  }

  // handle generics in non-control flow argument processing
  if (isa<cpu::GenericOp>(op))
    return success();

  if (results.empty())
    return success();

  assert(results.size() == 1 && "only single-result ops supported");
  const TileInfo &resultTileInfo = results[0]->getValue();
  if (resultTileInfo.isUninitialized())
    return success();

  if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(op)) {
    unsigned axis = expandOp.getAxis();
    TileInfo::TileShapeT srcShape;
    for (unsigned d = 0; d < resultTileInfo.getRank(); ++d)
      if (d != axis)
        srcShape.push_back(resultTileInfo.getShapeForDim(d));
    propagateTileShape(operands[0], srcShape);
    return success();
  }

  if (auto bcOp = dyn_cast<triton::BroadcastOp>(op)) {
    auto srcTy = cast<RankedTensorType>(bcOp.getSrc().getType());
    TileInfo::TileShapeT srcShape;
    for (auto [srcDim, tiledDim] :
         llvm::zip(srcTy.getShape(), resultTileInfo.getTiledShape()))
      srcShape.push_back(srcDim == 1 ? 1 : tiledDim);
    propagateTileShape(operands[0], srcShape);
    return success();
  }

  if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
    // no tile shape propagation needed to operand
    return success();
  }

  // Default: shape-preserving / elementwise — same tile shape for all operands.
  for (auto *opLattice : operands)
    propagateTileShape(opLattice,
                       llvm::to_vector(resultTileInfo.getTiledShape()));
  return success();
}

namespace {

// Replace the shape of a RankedTensorType with vectorShape, preserving element
// type and encoding. Non-tensor types are returned unchanged.
static Type updateTensorType(Type t, ArrayRef<int32_t> vectorShape) {
  auto tensorType = dyn_cast<RankedTensorType>(t);
  if (!tensorType)
    return t;
  assert(tensorType.getShape().size() == vectorShape.size() &&
         "expected tensor shape and vector shape to be the same during update");
  SmallVector<int64_t> newShape;
  for (auto [origDim, tileDim] : llvm::zip(tensorType.getShape(), vectorShape))
    newShape.push_back(tileDim == 0 ? origDim : (int64_t)tileDim);
  return RankedTensorType::get(newShape, tensorType.getElementType(),
                               tensorType.getEncoding());
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

static SmallVector<int32_t>
setTiledShapeForOperand(Operation *op,
                        const SmallVector<int32_t> &resultTiledShape,
                        Value operand) {
  // Drop the axis dimension from the result tiled shape to recover the
  // operand tiled shape. e.g. result=[4,1], axis=1 -> operand=[4].
  if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op)) {
    assert(operand == expandDimsOp.getSrc());
    uint32_t axis = expandDimsOp.getAxis();
    SmallVector<int32_t> operandShape(resultTiledShape);
    operandShape.erase(operandShape.begin() + axis);
    return operandShape;
  }

  // For each dim: if the *original* src dim is 1, keep 1 (we don't tile
  // across a broadcast dim); otherwise propagate the result tiled size.
  if (isa<triton::BroadcastOp>(op)) {
    auto srcTy = cast<RankedTensorType>(operand.getType());
    SmallVector<int32_t> operandShape;
    for (auto [srcDim, tiledDim] :
         llvm::zip(srcTy.getShape(), resultTiledShape))
      operandShape.push_back(srcDim == 1 ? 1 : tiledDim);
    return operandShape;
  }

  return resultTiledShape;
}

// Create the body block of a GenericOp, adding one block arg per ins value
// with tensor types replaced to the tile shape returned by `getTileShape`.
// Populates `mapping` with ins value → block arg entries and sets the
// insertion point to the start of the body. Returns the new block.
//
// `getTileShape` is called for each tensor-typed ins value and returns the
// tiled shape to use for its block arg. Defaults to returning `vectorShape`
// for all values (existing behaviour).
static Block *
initGenericBody(OpBuilder &rewriter, cpu::GenericOp generic,
                ArrayRef<Value> ins, ArrayRef<int32_t> vectorShape,
                IRMapping &mapping,
                llvm::function_ref<ArrayRef<int32_t>(Value, ArrayRef<int32_t>)>
                    getTileShape = nullptr) {
  Block *body = rewriter.createBlock(&generic.getBody());
  for (unsigned i = 0; i < vectorShape.size(); i++)
    body->addArgument(rewriter.getI32Type(),
                      generic.getLoc()); // chunk offset per vector shape dim
  for (Value v : ins) {
    ArrayRef<int32_t> shape =
        (getTileShape && isa<RankedTensorType>(v.getType()))
            ? getTileShape(v, vectorShape)
            : ArrayRef<int32_t>(vectorShape);
    Type argTy = updateTensorType(v.getType(), shape);
    mapping.map(v, body->addArgument(argTy, v.getLoc()));
  }
  rewriter.setInsertionPointToStart(body);
  return body;
}

// Query the tiled shape for `v` from `solver`. Falls back to `vectorShape` if
// the lattice is absent, uninitialized, or empty (e.g. scalars / no-tile).
static SmallVector<int32_t>
getTileShapeFromSolver(const DataFlowSolver &solver, Value v,
                       ArrayRef<int32_t> vectorShape) {
  auto *lattice = solver.lookupState<dataflow::Lattice<TileInfo>>(v);
  if (lattice && !lattice->getValue().isUninitialized() &&
      !lattice->getValue().isEmpty())
    return llvm::to_vector(lattice->getValue().getTiledShape());
  return llvm::to_vector(vectorShape);
}

static Type updateTensorTypeFromSolver(const DataFlowSolver &solver, Value v) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType)
    return v.getType();

  SmallVector<int32_t> origShape32(tensorType.getShape().begin(),
                                   tensorType.getShape().end());
  SmallVector<int64_t> newShape;
  auto tileShape = getTileShapeFromSolver(solver, v, origShape32);
  for (auto [origDim, tileDim] : llvm::zip(tensorType.getShape(), tileShape))
    newShape.push_back(tileDim == 0 ? origDim : (int64_t)tileDim);
  return RankedTensorType::get(newShape, tensorType.getElementType(),
                               tensorType.getEncoding());
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

// TODO: rename cvtChangesVectorSize or some such?
static bool shouldWrapCvt(triton::gpu::ConvertLayoutOp cvtOp) {
  auto sourceType = cast<RankedTensorType>(cvtOp.getSrc().getType());
  auto destEncoding =
      cast<RankedTensorType>(cvtOp.getResult().getType()).getEncoding();

  // TODO: will the convert layout fold away if this is true? maybe this entire
  // function is not necessary
  const bool areLayoutsEquivalent = triton::gpu::areLayoutsEquivalent(
      sourceType.getShape(),
      cast<triton::gpu::LayoutEncodingTrait>(sourceType.getEncoding()),
      cast<triton::gpu::LayoutEncodingTrait>(destEncoding));
  return !areLayoutsEquivalent;
}

struct WrapConvertLayoutOp
    : public mlir::OpRewritePattern<triton::gpu::ConvertLayoutOp> {
  using OpRewritePattern<triton::gpu::ConvertLayoutOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    // Don't re-wrap reductions already inside a ttc.generic.
    if (op->getParentOfType<cpu::GenericOp>())
      return failure();

    if (shouldWrapCvt(op)) {
      auto operand = op.getSrc();

      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto encoding =
          dyn_cast<gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
      if (!encoding)
        return failure();

      auto convertedTensorTy = cast<RankedTensorType>(op.getResult().getType());

      LinearLayout conversion = minimalCvtLayout(tensorTy, convertedTensorTy);
      LinearLayout srcLayout = triton::gpu::toLinearLayout(tensorTy);
      LinearLayout dstLayout = triton::gpu::toLinearLayout(convertedTensorTy);
      llvm::errs() << "conversion layout = " << conversion << "\n";
      llvm::errs() << "src layout = " << srcLayout << "\n";
      llvm::errs() << "dst layout = " << dstLayout << "\n";
      auto dims = llvm::to_vector(conversion.getInDimNames());
      assert(dims.size() == 1 &&
             "expected only register-dim layout conversion");
      auto kRegister = StringAttr::get(ctx, "register");
      auto bases = conversion.getBases();
      auto &regBase = bases[kRegister];

      int identityStartReg = -1;
      for (int idx = (int)regBase.size() - 1; idx >= 0; --idx) {
        assert(regBase[idx].size() == 1);
        if ((1 << idx) != regBase[idx][0]) {
          identityStartReg = 1 << idx;
          break;
        }
      }

      // TODO: feed identityStartReg back into the src layout to find x,y coords
      // and use as vector size.
      auto kBlock = StringAttr::get(ctx, "block");
      auto kWarp = StringAttr::get(ctx, "warp");
      auto kLane = StringAttr::get(ctx, "lane");
#if 0
      auto srcMapping = srcLayout.apply({{kRegister, identityStartReg}, {kBlock, 0}, {kWarp, 0}, {kLane, 0}});
      for (auto [k, v] : srcMapping)
        llvm::errs() << k << " = " << v << "\n";

    auto dstMapping = dstLayout.apply({{kRegister, identityStartReg}, {kBlock, 0}, {kWarp, 0}, {kLane, 0}});
    for (auto [k, v] : dstMapping)
        llvm::errs() << k << " = " << v << "\n";
#endif

      // numBits = log2(identityStartReg), e.g. log2(128) = 7
      int numBits = llvm::Log2_32(identityStartReg);

      // For a layout, sum basis contributions across the first numBits register
      // bits per output dimension. Since all contributions are distinct powers
      // of 2, the sum equals the max reachable coordinate, so tile extent = sum
      // + 1.
      auto tileFootprint = [&](const LinearLayout &layout) {
        int numOutDims = layout.getNumOutDims();
        SmallVector<int32_t> extents(numOutDims, 0);
        for (int i = 0; i < numBits; ++i) {
          ArrayRef<int32_t> basisVec = layout.getBasis(kRegister, i);
          for (int d = 0; d < numOutDims; ++d)
            extents[d] += basisVec[d];
        }
        for (auto &e : extents)
          ++e; // 0..max → size = max + 1
        return extents;
      };

      auto srcExtents = tileFootprint(srcLayout);
      auto dstExtents = tileFootprint(dstLayout);

      SmallVector<int32_t> tileSize(srcExtents.size());
      for (auto [d, _] : llvm::enumerate(tileSize))
        tileSize[d] = std::max(srcExtents[d], dstExtents[d]);

      // only wrap blocked->blocked conversions
      if (!isa<triton::gpu::BlockedEncodingAttr>(
              convertedTensorTy.getEncoding()))
        return failure();

      auto [srcBlockShape, srcVectorShape] =
          getBlockAndVectorShapes(tensorTy, encoding);
      auto [destBlockShape, destVectorShape] = getBlockAndVectorShapes(
          convertedTensorTy, cast<triton::gpu::BlockedEncodingAttr>(
                                 convertedTensorTy.getEncoding()));

#if 1
      auto vectorShape = srcBlockShape;
      auto blockShape = srcBlockShape;
#else

      assert(llvm::all_of(llvm::zip(srcBlockShape, destBlockShape),
                          [](auto pair) -> bool {
                            auto [s, d] = pair;
                            return s == d;
                          }) &&
             "expected cvt src and dest block shapes to be equal");
      auto blockShape = srcBlockShape;

      auto vectorShape = llvm::to_vector(llvm::map_range(
          llvm::zip(srcVectorShape, destVectorShape), [](auto pair) {
            auto [s, d] = pair;
            return std::max(s, d);
          }));
#endif
      SmallVector<Value> ins(op->getOperands().begin(),
                             op->getOperands().end());
      auto generic = cpu::GenericOp::create(
          rewriter, loc, /*resultTypes=*/TypeRange{op.getType()}, ins,
          /*params=*/ValueRange{}, blockShape, vectorShape);

      IRMapping bodyMapping;
      initGenericBody(rewriter, generic, ins, vectorShape, bodyMapping);

      auto newCvt = rewriter.clone(*op, bodyMapping);
      // newCvt->getResult(0).setType(
      // updateTensorType(op.getResult().getType(), vectorShape));
      cpu::YieldOp::create(rewriter, loc, newCvt->getResults());

      rewriter.replaceOp(op, generic.getResult(0));
      return success();
    }

    return failure();
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

static DenseMap<Value, llvm::SmallDenseSet<unsigned>>
buildKDimMap(scf::ForOp forOp, triton::DotOp dotOp) {
  DenseMap<Value, llvm::SmallDenseSet<unsigned>> kDims;

  kDims[dotOp.getA()].insert(
      cast<RankedTensorType>(dotOp.getA().getType()).getRank() - 1);
  // TODO: are B operand K indices always 0? Need to check layout
  kDims[dotOp.getB()].insert(0);

  auto updateKDimIndices = [&](Operation *op, int kDimIndex) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.omitUsesFromAbove = true;
    (void)getBackwardSlice(op, &backwardSlice, opt);
    for (auto op : backwardSlice)
      llvm::errs() << "op : " << *op << "\n";
    assert(false && "TODO");
  };

  updateKDimIndices(dotOp.getA().getDefiningOp(),
                    cast<RankedTensorType>(dotOp.getA().getType()).getRank() -
                        1);
  assert(false && "TODO");
}

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

    auto info = matchKLoopWithDot(forOp);
    if (!info)
      return failure();

    Location loc = forOp.getLoc();
    triton::DotOp dotOp = info->dotOp;
    auto resultTy = cast<RankedTensorType>(dotOp.getType());
    auto encoding = cast<gpu::BlockedEncodingAttr>(resultTy.getEncoding());

    // find the K dimension ops for the A and B inputs to the Dot op.
    // We track the K dimension and avoid modifying those layouts as the K loop
    // is unchanged during tiling auto kDimMap = buildKDimMap(forOp, dotOp);

    // TODO: this vector shape may be too small?
    auto [blockShape, vectorShape] =
        getBlockAndVectorShapes(resultTy, encoding);

    // TODO: run the dataflow analysis on the dot op
    // need to seed the analysis with the tile info for the dot op too
    dotOp->setAttr("ttcpu.tile_shape",
                   DenseI32ArrayAttr::get(rewriter.getContext(), vectorShape));

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TileInfoAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(forOp)))
      return failure();

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
    SmallVector<int32_t> tileShapeBuf;
    initGenericBody(
        rewriter, generic, ins, vectorShape, outerMapping,
        [&](Value v, ArrayRef<int32_t> vectorShape) -> ArrayRef<int32_t> {
          tileShapeBuf = getTileShapeFromSolver(solver, v, vectorShape);
          return tileShapeBuf;
        });

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
      for (auto [i, result] : llvm::enumerate(cloned->getResults())) {
        auto *lattice =
            solver.lookupState<dataflow::Lattice<TileInfo>>(op.getResult(i));
        if (!lattice)
          llvm_unreachable("Latice not found.");
        auto tileShape = lattice->getValue();
        // llvm::errs() << "op = " << op << "\n";
        // llvm::errs() << "tile shape = " << tileShape << "\n";

        if (auto resultTensorTy =
                dyn_cast<RankedTensorType>(result.getType())) {
          assert(resultTensorTy.getRank() == tileShape.getRank());
          result.setType(
              updateTensorType(result.getType(), tileShape.getTiledShape()));
        }
      }
      if (isa<triton::DotOp>(cloned))
        op.removeAttr("ttcpu.tile_shape");
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
    for (auto [idx, pair] : llvm::enumerate(llvm::zip(
             generic.getIns(),
             body->getArguments().drop_front(generic.getNumInductionVars())))) {
      auto [operand, arg] = pair;
      if (arg.use_empty())
        argsToErase.push_back(idx);
    }
    llvm::sort(argsToErase);
    for (auto idx : llvm::reverse(argsToErase)) {
      body->eraseArgument(generic.getNumInductionVars() + idx);
      newIns.erase(newIns.begin() + idx);
    }
    rewriter.modifyOpInPlace(generic,
                             [&]() { generic.getInsMutable().assign(newIns); });
    LDBG("Created generic op to replace K loop with dot op " << generic);
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
  if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(defOp)) {
    // only fuse cvt ops which are not wrapped
    return !shouldWrapCvt(cvtOp);
  }
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
  if (isa<triton::ExpandDimsOp>(defOp))
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

class InputFusion {
public:
  InputFusion(GenericOp G) : genericOp(G) {}

  // compute tiled shapes for fusion. returns false if there's nothing to fuse.
  bool analyze();

  // clone fusible ops into the body using the tiledShapes map
  void rewrite(IRRewriter &rewriter);

private:
  // seed the worklist with the current set of generic op inputs, clamping size
  // 1 dimensions to the block shape instead of the tiled shape
  void seedFromIns(SmallVectorImpl<Value> &worklist) {
    auto vectorShape = genericOp.getVectorShape();
    for (Value v : genericOp.getIns()) {
      auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
      if (!tensorTy)
        continue;
      auto shape = tensorTy.getShape();
      SmallVector<int32_t> tiledShape;
      for (auto [vs, dim] : llvm::zip(vectorShape, shape))
        tiledShape.push_back(std::min(vs, (int32_t)dim));

      // update the map
      LDBG("Initialize tiledShapes for " << v);
      LLVM_DEBUG({
        for (auto s : tiledShape)
          DBGS() << s << "\n";
      });
      tiledShapes[v] = tiledShape;

      Value init = getIterArgInit(v);
      if (init != v)
        tiledShapes[init] = tiledShape;
      worklist.push_back(v);
    }
  }

  SmallVector<int32_t> computeOperandTiledShape(Operation *op, Value result,
                                                Value operand) const;

  void topologicalSort();

  GenericOp genericOp;
  SmallVector<Operation *> sortedOps; // def-before-use
  SetVector<Operation *> fusibleOps;
  DenseMap<Value, SmallVector<int32_t>> tiledShapes; // Value -> tiled shape
};

bool InputFusion::analyze() {
  SmallVector<Value> worklist;
  seedFromIns(worklist);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();

    Operation *defOp = getIterArgInit(v).getDefiningOp();
    if (!defOp || !isFusible(defOp) || fusibleOps.contains(defOp))
      continue;
    fusibleOps.insert(defOp);
    for (Value operand : defOp->getOperands()) {
      auto shape = computeOperandTiledShape(defOp, v, operand);
      if (shape.empty())
        continue;
      LDBG("Update tiled shape for operand " << operand << " to:");
      LLVM_DEBUG({
        for (auto s : shape)
          DBGS() << s << "\n";
      });
      tiledShapes[operand] = shape;
      worklist.push_back(operand);
    }
  }
  if (fusibleOps.empty())
    return false;
  topologicalSort();
  return true;
}

void InputFusion::rewrite(IRRewriter &rewriter) {
  LDBG("Fusing inputs into generic " << genericOp);

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

  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<TileInfoAnalysis>(symbolTable);
  // Run on the enclosing function so that sortedOps and any values defined
  // outside the genericOp's immediate parent (e.g. function arguments) are
  // all visited and their operand lattices populated.
  auto funcOp = genericOp->getParentOfType<triton::FuncOp>();
  assert(funcOp && "expected genericOp to be inside a func");
  if (failed(solver.initializeAndRun(funcOp)))
    llvm_unreachable("solver should not fail!");

  SmallVector<Value> newIns(genericOp.getIns().begin(),
                            genericOp.getIns().end());
  SmallVector<Value> newParams(genericOp.getParams().begin(),
                               genericOp.getParams().end());
  SmallVector<unsigned> insIdxToRemove;

  Block *body = &genericOp.getBody().front();
  // Insert cloned ops before the first existing body op.
  rewriter.setInsertionPointToStart(body);

  IRMapping mapping;
  for (Operation *op : sortedOps) {
    LDBG("Fuse op " << *op);
    if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
      // replace op with makeDynamicRange

      auto makeRangeTensorTy =
          cast<RankedTensorType>(makeRangeOp.getResult().getType());
      auto sliceEncoding = dyn_cast<triton::gpu::SliceEncodingAttr>(
          makeRangeTensorTy.getEncoding());
#if 1
      unsigned dim = sliceEncoding ? sliceEncoding.getDim() : 0;
#else
      unsigned dim =
          sliceEncoding ? numInductionVars - 1 - sliceEncoding.getDim() : 0;
#endif

      auto *lattice = solver.lookupState<dataflow::Lattice<TileInfo>>(
          makeRangeOp.getResult());
      assert(lattice && "expected make range op lattice");
      auto tileInfo = lattice->getValue();
      llvm::errs() << "make range op from solver: " << lattice->getValue()
                   << "\n";

      Operation *newOp;
      if (tileInfo.isNotTiled()) {
        // clone without changing type?
        newOp = rewriter.clone(*op, mapping);
      } else {
        auto newMakeRangeResultType =
            updateTensorType(makeRangeTensorTy, tileInfo.getTiledShape());
        llvm::errs() << "new make range type: " << newMakeRangeResultType
                     << "\n";
        newOp = triton::cpu::MakeDynamicRangeOp::create(
            rewriter, makeRangeOp.getLoc(), newMakeRangeResultType,
            genericOp.getChunkOffset(dim));
        LDBG("Rewrite make range to dynamic make range with type "
             << newMakeRangeResultType);
      }

      mapping.map(makeRangeOp->getResults(), newOp->getResults());
      if (auto it = insToArgIdx.find(makeRangeOp.getResult());
          it != insToArgIdx.end()) {
        body->getArgument(it->second + numInductionVars)
            .replaceAllUsesWith(newOp->getResult(0));
        insIdxToRemove.push_back(it->second);
      }
      continue;
    }
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      auto tensorTy =
          dyn_cast<RankedTensorType>(constantOp.getResult().getType());
      if (tensorTy) {
        auto newTensorTy = cast<RankedTensorType>(
            updateTensorTypeFromSolver(solver, constantOp.getResult()));
        auto denseAttr = cast<DenseElementsAttr>(constantOp.getValue());
        assert(denseAttr.isSplat() &&
               "non-splat tensor constants not yet supported in fuseInputs");
        auto newAttr = DenseElementsAttr::get(
            newTensorTy, *denseAttr.getValues<Attribute>().begin());
        auto newConstant =
            arith::ConstantOp::create(rewriter, constantOp.getLoc(), newAttr);
        LDBG("Replace constant with " << newConstant);
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
    for (Value operand : op->getOperands()) {
      if (mapping.contains(operand))
        continue;
      newIns.push_back(operand);

      auto *lattice = solver.lookupState<dataflow::Lattice<TileInfo>>(operand);
      if (!lattice)
        llvm_unreachable("Latice not found.");
      auto tileShape = lattice->getValue();
      llvm::errs() << "tile shape from solver: " << tileShape << "\n";
      llvm::errs() << " vs tiledShape: ";
      auto tiledShape = tiledShapes.lookup(operand);
      for (auto s : tiledShape)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";

      mapping.map(operand,
                  body->addArgument(updateTensorTypeFromSolver(solver, operand),
                                    operand.getLoc()));
    }

    Operation *newOp = rewriter.clone(*op, mapping);
    for (auto [origResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      auto tiledShape = tiledShapes.lookup(origResult);
      auto *lattice =
          solver.lookupState<dataflow::Lattice<TileInfo>>(origResult);
      if (!lattice)
        llvm_unreachable("Latice not found.");
      auto tileShape = lattice->getValue();

      llvm::errs() << "tile shape from solver: " << tileShape << "\n";
      llvm::errs() << " vs tiledShape: ";
      for (auto s : tiledShape)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";

      newResult.setType(updateTensorTypeFromSolver(solver, origResult));

      LDBG("Update result type to " << newResult.getType());
      if (auto it = insToArgIdx.find(origResult); it != insToArgIdx.end()) {
        BlockArgument bodyArg =
            body->getArgument(it->second + numInductionVars);
        Value replacement = newResult;
        // The generic may have broadcast 1-dims to the full vectorShape in the
        // body arg type (e.g. tensor<1x4xi32> fused from tensor<1x64xi32> but
        // body arg is tensor<4x4xi32>). Insert a broadcast to match.
        if (false && bodyArg.getType() != newResult.getType()) {
          replacement = triton::BroadcastOp::create(
              rewriter, op->getLoc(), bodyArg.getType(), replacement);
        }
        bodyArg.replaceAllUsesWith(replacement);
        insIdxToRemove.push_back(it->second);
      }
    }
    mapping.map(op->getResults(), newOp->getResults());
  }

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
  for (Operation *op : llvm::reverse(fusibleOps)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }
}

SmallVector<int32_t>
InputFusion::computeOperandTiledShape(Operation *op, Value result,
                                      Value operand) const {
  if (!isa<RankedTensorType>(result.getType()))
    return {};

  auto resultTiledShape = tiledShapes.lookup(result);
  return setTiledShapeForOperand(op, resultTiledShape, operand);
}

void InputFusion::topologicalSort() {
  // TODO: do we still need visited here if fusibleops is a set vector?
  DenseSet<Operation *> visited;
  std::function<void(Operation *)> visit = [&](Operation *op) {
    if (!fusibleOps.contains(op) || visited.contains(op))
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
  for (Operation *op : fusibleOps)
    visit(op);
  return;
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
    patterns.add<WrapConvertLayoutOp>(context, benefitDefault);
    patterns.add<WrapKLoopWithDotOp>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    // Debug helper: run the solver on each function and annotate every op
    // result with its tile info as a "ttcpu.tile_info" string attribute.
    // Remove before submitting.
    m.walk([&](triton::FuncOp funcOp) {
      SymbolTableCollection symbolTable;
      DataFlowSolver solver;
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      solver.load<TileInfoAnalysis>(symbolTable);
      if (failed(solver.initializeAndRun(funcOp)))
        return;
      funcOp.walk([&](Operation *op) {
        for (auto [i, result] : llvm::enumerate(op->getResults())) {
          auto *lattice =
              solver.lookupState<dataflow::Lattice<TileInfo>>(result);
          std::string s;
          llvm::raw_string_ostream os(s);
          if (!lattice)
            os << "no-lattice";
          else
            lattice->getValue().print(os);
          op->setAttr(("ttcpu.tile_info." + llvm::Twine(i)).str(),
                      StringAttr::get(context, s));
        }
      });
    });

    llvm::errs() << "### MODULE BEFORE FUSION ###\n";
    m.dump();
    llvm::errs() << "### END MODULE BEFORE FUSION ###\n";

    // Step 2: Fuse elementwise ops and loads into each generic, bottom-up.
    SmallVector<cpu::GenericOp> worklist;
    m.walk([&](cpu::GenericOp op) { worklist.push_back(op); });
    IRRewriter rewriter(context);
    for (cpu::GenericOp genericOp : llvm::reverse(worklist)) {
      InputFusion fusion(genericOp);
      if (!fusion.analyze())
        return;
      fusion.rewrite(rewriter);
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
