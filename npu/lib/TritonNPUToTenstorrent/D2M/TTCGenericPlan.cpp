#include "TTCGenericPlan.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <deque>

namespace mlir {
namespace triton {
namespace npu {

namespace ttcore = ::mlir::tt::ttcore;

namespace {
struct Planes {
  SmallVector<Operation *> data; // topologically ordered
  SetVector<Operation *> control;
  SmallVector<Operation *> boundary; // loads/stores
};

static bool isBoundaryOp(Operation *op) {
  return isa<triton::LoadOp, triton::StoreOp>(op);
}

static mlir::FailureOr<Planes> classify(cpu::GenericOp generic) {
  Block &body = generic.getBody().front();

  SmallVector<Operation *> boundary;
  // boundary: the ops that cross between address space and value space
  for (Operation &op : body.without_terminator())
    if (isBoundaryOp(&op))
      boundary.push_back(&op);

  SetVector<Operation *> control;
  auto ptrOperandOf = [](Operation *root) -> Value {
    return TypeSwitch<Operation *, Value>(root)
        .Case<triton::LoadOp>(
            [&](triton::LoadOp load) { return load.getPtr(); })
        .Case<triton::StoreOp>(
            [&](triton::StoreOp store) { return store.getPtr(); })
        .Default([&](Operation *) {
          llvm_unreachable("expected only load or store op in boundary");
          return Value{};
        });
  };
  auto maskOperandOf = [](Operation *root) -> Value {
    return TypeSwitch<Operation *, Value>(root)
        .Case<triton::LoadOp>(
            [&](triton::LoadOp load) { return load.getMask(); })
        .Case<triton::StoreOp>(
            [&](triton::StoreOp store) { return store.getMask(); })
        .Default([&](Operation *) { return Value{}; });
  };
  // control: backward slice of every pointer and mask operand, clipped to body
  BackwardSliceOptions opts;
  opts.inclusive = true;
  opts.filter = [&](Operation *op) { return op->getBlock() == &body; };
  for (Operation *b : boundary) {
    (void)getBackwardSlice(ptrOperandOf(b), &control, opts);
    if (Value m = maskOperandOf(b))
      (void)getBackwardSlice(m, &control, opts);
  }

  DenseSet<Operation *> dataOps;
  // data: forward closure from load results, stopping at tt.store
  std::deque<Operation *> worklist;
  for (auto op : boundary) {
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      Operation *owner = load.getResult().getDefiningOp();
      // ignores block args
      if (owner)
        worklist.push_back(owner);
    }
  }
  while (!worklist.empty()) {
    Operation *cur = worklist.back();
    worklist.pop_back();
    for (Value result : cur->getResults())
      for (Operation *user : result.getUsers())
        if (!isa<triton::StoreOp>(user) && user->getBlock() == &body)
          if (dataOps.insert(user).second)
            worklist.push_back(user);
  }

  SmallVector<Operation *> orderedDataOps;
  for (Operation &op : body.without_terminator())
    if (dataOps.contains(&op))
      orderedDataOps.push_back(&op);

  // ensure no ops are unclassified and planes are disjoint
  for (Operation &op : body.without_terminator()) {
    SmallVector<StringRef, 3> planes;
    if (dataOps.contains(&op))
      planes.push_back("data");
    if (control.count(&op))
      planes.push_back("control");
    if (isBoundaryOp(&op))
      planes.push_back("boundary");

    if (planes.size() == 1)
      continue;

    InFlightDiagnostic diag =
        planes.empty()
            ? op.emitOpError(
                  "is not reachable from any tt.load result, nor from "
                  "any pointer or mask operand, so it would be "
                  "silently dropped when the ttc.generic is erased")
            : op.emitOpError("belongs to multiple planes (")
                  << llvm::join(planes, ", ")
                  << "); a value feeding both a computation and an address "
                     "requires cloning, which is not yet supported";
    diag.attachNote(generic.getLoc()) << "while classifying this ttc.generic";
    return failure();
  }

  return Planes{orderedDataOps, control, boundary};
}

/// Read the logical (scalar-element) shape of kernel argument `argIndex` from
/// the tt.tensor_rank / tt.tensor_shape_N attributes attached by the backend's
/// tensor specialization.
///
/// A rank-1 shape {N} is promoted to {N/32, 32}. That is not a reinterpretation
/// of the data: a row-major {N/32, 32} tensor tiled into 32x32 tiles places
/// elements [1024*i, 1024*i + 1024) into tile i, which is exactly how the host
/// packs a 1-D tensor. Leaving it rank-1 would hit MetalLayoutAttr's "rank-1 is
/// a single logical row in a 2D tile plane" rule and describe the same buffer
/// as N/32 tiles of 32 useful elements each.
static FailureOr<SmallVector<int64_t>>
readLogicalShape(triton::FuncOp funcOp, unsigned argIndex,
                 Operation *diagnosticAnchorOp) {
  auto rankAttr =
      funcOp.getArgAttrOfType<IntegerAttr>(argIndex, "tt.tensor_rank");
  if (!rankAttr)
    return diagnosticAnchorOp->emitError()
           << "kernel argument #" << argIndex
           << " has no tt.tensor_rank attribute; tensor shape specialization "
              "did not run for it (do_not_specialize?)";

  SmallVector<int64_t> shape;
  for (int64_t dim = 0, rank = rankAttr.getInt(); dim < rank; ++dim) {
    auto dimAttr = funcOp.getArgAttrOfType<IntegerAttr>(
        argIndex, ("tt.tensor_shape_" + Twine(dim)).str());
    if (!dimAttr)
      return diagnosticAnchorOp->emitError()
             << "kernel argument #" << argIndex
             << " declares tt.tensor_rank = " << rank
             << " but is missing tt.tensor_shape_" << dim;
    shape.push_back(dimAttr.getInt());
  }

  auto tile = ttcore::TileType::getDefaultShape();
  if (shape.size() == 1) {
    const int64_t tileVolume = tile[0] * tile[1];
    if (shape[0] % tileVolume != 0)
      return diagnosticAnchorOp->emitError()
             << "rank-1 kernel argument #" << argIndex << " has " << shape[0]
             << " elements, which is not a whole number of " << tileVolume
             << "-element tiles";
    shape = {shape[0] / tile[0], tile[0]};
  }
  return shape;
}

/// Turn the classified planes into the plan's operand list.
///
/// Operand order is loads-then-store, which is the ins-then-outs order that
/// d2m.generic and its indexing_maps require.
static LogicalResult populateOperands(cpu::GenericOp generic, GenericPlan &plan,
                                      const Planes &planes) {
  auto funcOp = generic->getParentOfType<triton::FuncOp>();
  if (!funcOp)
    return generic.emitError("ttc.generic is not inside a triton function");

  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::StoreOp> stores;
  for (Operation *op : planes.boundary) {
    if (auto load = dyn_cast<triton::LoadOp>(op))
      loads.push_back(load);
    else
      stores.push_back(cast<triton::StoreOp>(op));
  }

  if (loads.empty())
    return generic.emitError("expected at least one tt.load in the body");
  if (stores.size() != 1)
    return generic.emitError()
           << "expected exactly one tt.store in the body, got " << stores.size()
           << "; multi-output generics are not yet supported";

  auto tile = ttcore::TileType::getDefaultShape();

  auto addOperand = [&](Operation *boundaryOp, Value ptr,
                        RankedTensorType valueTy) -> LogicalResult {
    GenericPlan::Operand operand;
    operand.boundaryOp = boundaryOp;
    operand.elementType = valueTy.getElementType();

    // TODO: trace `ptr` back to the kernel argument. Two legs:
    //   1. inside the body, walk addptr/splat/broadcast back to a body
    //      BlockArgument -- TagInputOutputs.cpp:23 already has this half;
    //   2. map that block arg to an `ins` operand via
    //      argIndex - generic.getInsArgOffset(), then walk that operand out to
    //      the enclosing triton::FuncOp BlockArgument.
    operand.funcArg = BlockArgument();
    if (!operand.funcArg)
      return boundaryOp->emitOpError(
          "TODO: pointer-to-kernel-argument tracing is not implemented");

    FailureOr<SmallVector<int64_t>> logicalShape =
        readLogicalShape(funcOp, operand.funcArg.getArgNumber(), boundaryOp);
    if (failed(logicalShape))
      return failure();
    operand.logicalShape = std::move(*logicalShape);

    // Only the trailing two dimensions are tiled; any leading dims are batch.
    const size_t rank = operand.logicalShape.size();
    for (auto [dim, extent] : llvm::enumerate(operand.logicalShape)) {
      int64_t divisor = dim + 2 >= rank ? tile[dim - (rank - 2)] : 1;
      if (extent % divisor != 0)
        return boundaryOp->emitOpError()
               << "kernel argument #" << operand.funcArg.getArgNumber()
               << " dimension " << dim << " (" << extent
               << ") is not a multiple of the tile extent " << divisor;
      operand.tensorTiles.push_back(extent / divisor);
    }

    plan.operands.push_back(std::move(operand));
    return success();
  };

  // TODO: two loads of the same kernel argument currently become two distinct
  // operands. Legal, but it hands d2m two `ins` that alias; dedup once we
  // can prove the two accesses have the same indexing map.
  for (triton::LoadOp load : loads)
    if (failed(addOperand(load, load.getPtr(),
                          cast<RankedTensorType>(load.getType()))))
      return failure();
  plan.numInputs = plan.operands.size();

  triton::StoreOp store = stores.front();
  if (failed(addOperand(store, store.getPtr(),
                        cast<RankedTensorType>(store.getValue().getType()))))
    return failure();

  plan.dataOps.assign(planes.data.begin(), planes.data.end());
  return success();
}

// Largest `d` such that `d` divides `n` and `d <= limit`. Always >= 1.
static int64_t largestDivisorAtMost(int64_t n, int64_t limit) {
  for (int64_t d = std::min(n, limit); d > 1; --d)
    if (n % d == 0)
      return d;
  return 1;
}

} // namespace

LogicalResult GenericPlan::setIterationSpace(ArrayRef<int64_t> workerGrid,
                                             Operation *diagnosticAnchorOp) {
  MLIRContext *context = diagnosticAnchorOp->getContext();

  if (operands.empty())
    return diagnosticAnchorOp->emitError(
        "cannot derive an iteration space: plan has no operands");

  ArrayRef<int64_t> tiles = operands.front().tensorTiles;
  if (tiles.size() < 2)
    return diagnosticAnchorOp->emitError()
           << "expected an operand tile shape of rank >= 2, got rank "
           << tiles.size();
  for (const GenericPlan::Operand &operand : operands)
    if (!llvm::equal(operand.tensorTiles, tiles))
      return diagnosticAnchorOp->emitError(
          "operands do not all cover the same tile shape; broadcasting is not "
          "yet supported");

  if (workerGrid.size() != tiles.size())
    return diagnosticAnchorOp->emitError()
           << "worker grid rank (" << workerGrid.size()
           << ") does not match operand tile rank (" << tiles.size() << ")";

  gridShape.clear();
  blockFactors.clear();
  iteratorTypes.clear();

  // Bounding each grid dim by the corresponding worker grid dim also bounds
  // the grid volume by the core count, so no separate volume check is needed.
  for (auto [dim, extent] : llvm::enumerate(tiles)) {
    int64_t grid = largestDivisorAtMost(extent, workerGrid[dim]);
    gridShape.push_back(grid);
    blockFactors.push_back(extent / grid);
    // TODO: support reductions
    iteratorTypes.push_back(
        ttcore::IteratorTypeAttr::get(context, ttcore::IteratorType::Parallel));
  }

  // Never let a silent under-allocation look like full occupancy.
  int64_t used = 1, available = 1;
  for (int64_t g : gridShape)
    used *= g;
  for (int64_t g : workerGrid)
    available *= g;
  if (used < available)
    diagnosticAnchorOp->emitRemark()
        << "iteration space of " << tiles[0] << "x" << tiles[1]
        << " tiles maps onto " << used << " of " << available
        << " cores; no larger divisor of the tile extents "
           "fits the worker grid";

  return success();
}

mlir::FailureOr<GenericPlan> GenericPlan::build(cpu::GenericOp generic) {
  // check generic op metadata criteria

  // TODO: support generics with reductions
  if (generic.getReductionDims().size() != 0) {
    generic.emitError("reduction dims not yet supported in D2M lowering");
    return failure();
  }
  if (generic.getInitVals().size() != 0) {
    generic.emitError("init vals not yet supported in D2M lowering");
    return failure();
  }

  for (auto ins : generic.getIns()) {
    if (isa<RankedTensorType>(ins.getType())) {
      generic.emitError(
          "tensor type inputs to generic op not yet supported in D2M lowering");
      return failure();
    }
  }

  // TODO: we can likely relax this restriction, but for now it's not
  // unreasonable
  for (auto [bVal, t] :
       llvm::zip(generic.getBlockShape(), generic.getTileShape())) {
    // block shape is a value, so check for a constant
    APInt val;
    if (!matchPattern(bVal, m_ConstantInt(&val))) {
      generic.emitError("expected block shape to be constants");
      return failure();
    }
    int64_t b = val.getSExtValue();
    if (b != t) {
      generic.emitError("expected generic block shape and tile shape to be "
                        "equal for D2M lowering");
      return failure();
    }
  }

  GenericPlan plan;

  auto planeResult = classify(generic);
  if (failed(planeResult))
    return failure();

  Planes planes = *planeResult;

  // Operands first: setIterationSpace reads their tensorTiles.
  if (failed(populateOperands(generic, plan, planes)))
    return failure();

  ArrayRef<int64_t> workerGrid = tt::TritonTenstorrentDialect::getGridAttr(
                                     generic->getParentOfType<ModuleOp>())
                                     .getShape();
  if (failed(plan.setIterationSpace(workerGrid, generic)))
    return failure();

  // TODO: indexingMap per operand, plus the persistent-loop mapping.

  return plan;
}

} // namespace npu
} // namespace triton
} // namespace mlir
