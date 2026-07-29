#include "TTCGenericPlan.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"

#include <deque>
namespace mlir {
namespace triton {
namespace npu {

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

} // namespace

mlir::FailureOr<GenericPlan> buildPlan(cpu::GenericOp generic) {
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

  return plan;
}

} // namespace npu
} // namespace triton
} // namespace mlir
