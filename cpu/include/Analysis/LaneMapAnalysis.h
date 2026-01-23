#ifndef TRITONCPU_LANEMAPANALYSIS_H
#define TRITONCPU_LANEMAPANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

struct LaneInfo {
  enum Kind { Unknown, Uniform, AffineLane } kind = Unknown;

  // Uniform: baseScalar optionally holds the defining scalar Value.
  // AffineLane: laneVal = baseScalar + constOffset + stride * laneId
  Value baseScalar;
  int64_t constOffset = 0;
  int64_t stride = 0;

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

  static LaneInfo join(const LaneInfo &a, const LaneInfo &b) {
    if (a == b)
      return a;

    if (a.kind == Unknown)
      return b;
    if (b.kind == Unknown)
      return a;

    // Two uniforms => still uniform (TODO: propagate baseScalar?)
    if (a.kind == Uniform && b.kind == Uniform)
      return getUniform();

    // Uniform + affine => affine (fill baseScalar if missing)
    if (a.kind == Uniform && b.kind == AffineLane)
      return b.baseScalar ? b
                          : getAffine(a.baseScalar, b.constOffset, b.stride);
    if (b.kind == Uniform && a.kind == AffineLane)
      return a.baseScalar ? a
                          : getAffine(b.baseScalar, a.constOffset, a.stride);

    // Two different affine => unknown (conservative)
    return getUnknown();
  }

  void print(llvm::raw_ostream &os) const {
    os << "LaneInfo(";
    switch (kind) {
    case Unknown:
      os << "Unknown";
      break;
    case Uniform:
      os << "Uniform";
      break;
    case AffineLane:
      os << "Affine(base=" << baseScalar << ", c=" << constOffset
         << ", s=" << stride << ")";
      break;
    }
    os << ")";
  }
};

#if 0
/// Lattice wrapper that uses LaneInfo::join.
class LaneLattice : public mlir::dataflow::Lattice<LaneInfo> {
public:
  using Lattice::Lattice;

  mlir::ChangeResult join(const LaneInfo &rhs) override {
    LaneInfo cur = getValue();
    LaneInfo next = LaneInfo::join(cur, rhs);
    if (next == cur) return mlir::ChangeResult::NoChange;
    setValue(next);
    return mlir::ChangeResult::Change;
  }
};
#endif

using LaneLattice = dataflow::Lattice<LaneInfo>;

class LaneMapAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<LaneLattice> {
public:
  using SparseForwardDataFlowAnalysis<
      LaneLattice>::SparseForwardDataFlowAnalysis;
  using SparseForwardDataFlowAnalysis<LaneLattice>::getLatticeElement;

  void setToEntryState(LaneLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(LaneInfo::getUnknown()));
  }

  LogicalResult visitOperation(mlir::Operation *op,
                               llvm::ArrayRef<const LaneLattice *> operands,
                               llvm::ArrayRef<LaneLattice *> results) override;

private:
  static std::optional<int64_t> getI64Const(mlir::Value v) {
    mlir::Attribute attr;
    if (mlir::matchPattern(v, mlir::m_Constant(&attr))) {
      if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(attr))
        return ia.getInt();
    }
    return std::nullopt;
  }

  void setAllUnknown(llvm::ArrayRef<LaneLattice *> results) {
    for (auto *r : results)
      if (r)
        propagateIfChanged(r, r->join(LaneInfo::getUnknown()));
  }

  static LaneInfo evalAddI(const LaneInfo &a, const LaneInfo &b);
  static LaneInfo evalSubI(const LaneInfo &a, const LaneInfo &b);
  static LaneInfo evalMulI(const LaneInfo &a, const LaneInfo &b);
  static LaneInfo evalAddPtr(const LaneInfo &basePtr, const LaneInfo &offs);
};

// TODO: maybe unused
bool isPointwiseStore(triton::StoreOp storeOp, LaneMapAnalysis &analysis);

} // namespace triton
} // namespace mlir

#endif // TRITONCPU_LANEMAPANALYSIS_H
