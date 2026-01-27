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
namespace cpu {

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

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const LaneInfo &info) {
  info.print(os);
  return os;
}

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
};

bool isPointwiseStore(triton::StoreOp storeOp, LaneMapAnalysis &analysis);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITONCPU_LANEMAPANALYSIS_H
