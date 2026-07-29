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

  ModuleOp mod = diagnosticAnchorOp->getParentOfType<ModuleOp>();
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

  // TODO: populate the plan
  ArrayRef<int64_t> workerGrid = tt::TritonTenstorrentDialect::getGridAttr(
                                     generic->getParentOfType<ModuleOp>())
                                     .getShape();
  if (failed(plan.setIterationSpace(workerGrid, generic)))
    return failure();

  return plan;
}

} // namespace npu
} // namespace triton
} // namespace mlir
