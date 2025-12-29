#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

struct LaneInfo {
  enum Kind { Unknown, Uniform, AffineLane } kind = Unknown;

  // For Uniform: baseScalar may hold the defining scalar Value (optional).
  // For AffineLane: baseScalar is a uniform scalar Value (e.g., block_start),
  // plus a constant offset, plus integer stride.
  Value baseScalar; // may be null
  int64_t constOffset = 0;
  int64_t stride = 0; // valid iff kind==AffineLane

  static LaneInfo getUnknown() { return LaneInfo(); }
  static LaneInfo getUniform(Value baseScalar = Value()) {
    LaneInfo f;
    f.kind = Uniform;
    f.baseScalar = baseScalar;
    return f;
  }
  static LaneInfo getAffine(Value baseScalar, int64_t c, int64_t s) {
    LaneInfo f;
    f.kind = AffineLane;
    f.baseScalar = baseScalar;
    f.constOffset = c;
    f.stride = s;
    return f;
  }

  bool operator==(const LaneInfo &o) const {
    return kind == o.kind && baseScalar == o.baseScalar &&
           constOffset == o.constOffset && stride == o.stride;
  }

  // Conservative join.
  static LaneInfo join(const LaneInfo &a, const LaneInfo &b) {
    if (a == b)
      return a;

    if (a.kind == Unknown && b.kind != Unknown)
      return b;
    if (b.kind == Unknown && a.kind != Unknown)
      return a;
    if (a.kind == Unknown && b.kind == Unknown)
      return getUnknown();

    // Uniform join: if both uniform but different bases, keep uniform (still
    // pointwise)
    if (a.kind == Uniform && b.kind == Uniform)
      return getUniform();

    // If one is uniform and other is affine, keep affine (still pointwise)
    // If the affine LaneInfo does not have a base scalar use the uniform's base
    // scalar
    if (a.kind == Uniform && b.kind == AffineLane)
      return b.baseScalar ? b
                          : getAffine(a.baseScalar, b.constOffset, b.stride);
    if (b.kind == Uniform && a.kind == AffineLane)
      return a.baseScalar ? a
                          : getAffine(b.baseScalar, a.constOffset, a.stride);

    // Two different affine forms => unknown (conservative)
    return getUnknown();
  }

  void print(raw_ostream &os) const {
    os << "LaneInfo(";
    switch (kind) {
    case LaneInfo::Unknown:
      os << "Unknown";
      break;
    case LaneInfo::Uniform:
      os << "Uniform";
      break;
    case LaneInfo::AffineLane:
      os << "Affine(base=" << baseScalar << ", c=" << constOffset
         << ", s=" << stride << ")";
      break;
    }
    os << ")";
  }
};

class LaneMapAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                            dataflow::Lattice<LaneInfo>> {
public:
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<LaneInfo>>::SparseForwardDataFlowAnalysis;
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<LaneInfo>>::getLatticeElement;

  void setToEntryState(dataflow::Lattice<LaneInfo> *lattice) override {
    propagateIfChanged(lattice, lattice->join(LaneInfo::getUnknown()));
  }

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<LaneInfo> *> operands,
                 ArrayRef<dataflow::Lattice<LaneInfo> *> results) override {
    LaneInfo laneInfo;
    auto result = op->getResult(0);

    if (isa<arith::ConstantOp>(op)) {
      // constants are always uniform across lanes
      for (auto *r : results)
        if (r)
          propagateIfChanged(r, r->join(LaneInfo::getUniform()));
      return success();
    }
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      // splat is uniform across lanes, keep the baseScalar
      for (auto *r : results)
        if (r)
          propagateIfChanged(r, r->join(LaneInfo::getUniform(splat.getSrc())));
      return success();
    }
    if (auto makeRange = dyn_cast<triton::MakeRangeOp>(op)) {
      // make_range produces affine lane mapping
      auto start = makeRange.getStart();
      for (auto *r : results)
        if (r)
          propagateIfChanged(
              r, r->join(LaneInfo::getAffine(Value(), start, /*step=*/1)));
      return success();
    }

    // For elementwise ops, joining operand lane-forms is the canonical,
    // conservative rule.
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::CmpIOp,
            arith::CmpFOp, arith::SelectOp>(op)) {
      LaneInfo a =
          operands[0] ? operands[0]->getValue() : LaneInfo::getUnknown();
      LaneInfo b =
          operands[1] ? operands[1]->getValue() : LaneInfo::getUnknown();
      LaneInfo joined = LaneInfo::join(a, b);
      for (auto *r : results)
        if (r)
          propagateIfChanged(r, r->join(joined));
      return success();
    }
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
      LaneInfo base =
          operands[0] ? operands[0]->getValue() : LaneInfo::getUnknown();
      LaneInfo offs =
          operands[1] ? operands[1]->getValue() : LaneInfo::getUnknown();
      LaneInfo joined = LaneInfo::join(base, offs);
      for (auto *r : results)
        if (r)
          propagateIfChanged(r, r->join(joined));
      return success();
    }

    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      LaneInfo ptr =
          operands[0] ? operands[0]->getValue() : LaneInfo::getUnknown();
      // load keeps the lane mapping of the pointer
      for (auto *r : results)
        if (r)
          propagateIfChanged(r, r->join(ptr));
      return success();
    }

    if (isa<triton::StoreOp>(op)) {
      for (auto *r : results)
        if (r)
          propagateIfChanged(r, r->join(LaneInfo::getUnknown()));
      return success();
    }

    // unknown op kind => set unknown
    for (auto *r : results)
      propagateIfChanged(r, r->join(LaneInfo::getUnknown()));

    return success();
  }
};

bool isPointwiseStore(triton::StoreOp storeOp, LaneMapAnalysis &analysis) {
  auto *ptrLatElement = analysis.getLatticeElement(storeOp.getPtr());
  auto *valLatElement = analysis.getLatticeElement(storeOp.getValue());

  if (!ptrLatElement || !valLatElement)
    return false;

  const LaneInfo &p = ptrLatElement->getValue();
  const LaneInfo &v = valLatElement->getValue();

  if (p.kind == LaneInfo::Unknown || v.kind == LaneInfo::Unknown)
    return false;

  // equivalent non-unknown lane mappings are pointwise
  if (p == v)
    return true;

  // one uniform and the other affine is still pointwise.
  if (p.kind == LaneInfo::Uniform && v.kind == LaneInfo::AffineLane)
    return true;
  if (v.kind == LaneInfo::Uniform && p.kind == LaneInfo::AffineLane)
    return true;

  return false;
}

} // namespace triton
} // namespace mlir
