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

struct UniformLinearExpr {
  UniformLinearExpr() = default;

  int64_t c = 0;
  DenseMap<Value, int64_t>
      coeffs; // sum coeff[v] * v where v must be scalar-uniform

  bool operator==(const UniformLinearExpr &other) const {
    return c == other.c && coeffs == other.coeffs;
  }

  static UniformLinearExpr add(const UniformLinearExpr &a,
                               const UniformLinearExpr &b) {
    UniformLinearExpr result;
    result.c = a.c + b.c;
    result.coeffs = a.coeffs;
    for (auto &kv : b.coeffs) {
      result.coeffs[kv.first] += kv.second;
    }
    return result;
  }

  static UniformLinearExpr fromConst(int64_t c) {
    UniformLinearExpr result;
    result.c = c;
    return result;
  }

  static UniformLinearExpr fromValue(Value v, int64_t coeff = 1) {
    UniformLinearExpr result;
    result.coeffs[v] = coeff;
    return result;
  }

  void print(llvm::raw_ostream &os) const {
    os << "{c=" << c;
    if (!coeffs.empty()) {
      os << ", coeffs=[";
      bool first = true;
      for (auto &kv : coeffs) {
        if (!first)
          os << ", ";
        first = false;
        os << kv.second << "*" << kv.first;
      }
      os << "]";
    }
    os << "}";
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const UniformLinearExpr &expr) {
  expr.print(os);
  return os;
}

struct LaneInfo {
  enum Kind {
    Uninitialized,
    Uniform,
    Pointwise,
    Overdefined
  } kind = Uninitialized;

  // Pointwise: value(lane) = base(lane) + uniform + laneStride * laneId
  // Uniform:   value(lane) = uniform
  Value base;                // may be scalar or tensor
  UniformLinearExpr uniform; // lane-invariant linear expression
  int64_t laneStride = 0; // TODO: do we need this? make_range step is always 1

  static LaneInfo getUninitialized() { return LaneInfo(); }

  static LaneInfo getOverdefined() {
    LaneInfo f;
    f.kind = Overdefined;
    return f;
  }

  static LaneInfo getUniform(UniformLinearExpr u = UniformLinearExpr()) {
    LaneInfo f;
    f.kind = Uniform;
    f.uniform = std::move(u);
    return f;
  }

  static LaneInfo getPointwise(Value base,
                               UniformLinearExpr u = UniformLinearExpr(),
                               int64_t s = 0) {
    LaneInfo f;
    f.kind = Pointwise;
    f.base = base;
    f.uniform = std::move(u);
    f.laneStride = s;
    return f;
  }

  static LaneInfo getPessimisticValueState(Value v) {
    if (!isa<TensorType>(v.getType())) {
      return getUniform(UniformLinearExpr::fromValue(v));
    }
    return getUninitialized();
  }

  bool operator==(const LaneInfo &o) const {
    if (kind != o.kind)
      return false;
    if (!(uniform == o.uniform))
      return false;

    if (kind == Pointwise) {
      return base == o.base && laneStride == o.laneStride;
    }

    return true;
  }

  static LaneInfo join(const LaneInfo &a, const LaneInfo &b) {
    if (a == b)
      return a;

    // ⊥ rules
    if (a.kind == Uninitialized)
      return b;
    if (b.kind == Uninitialized)
      return a;

    // ⊤ rules
    if (a.kind == Overdefined || b.kind == Overdefined)
      return getOverdefined();

    // Uniform ⊔ Uniform
    if (a.kind == Uniform && b.kind == Uniform) {
      if (a.uniform == b.uniform)
        return a;
      return getUniform(); // "some uniform", drop details
    }

    // Promote Uniform to Pointwise(null base) for merging with Pointwise
    auto asPointwise = [](const LaneInfo &x) -> LaneInfo {
      if (x.kind == Uniform)
        return getPointwise(Value(), x.uniform, /*s=*/0);
      return x;
    };
    LaneInfo ap = asPointwise(a);
    LaneInfo bp = asPointwise(b);

    // Pointwise ⊔ Pointwise
    if (ap.kind == Pointwise && bp.kind == Pointwise) {
      // Base: if either is unknown-base or bases disagree => unknown-base
      Value outBase;
      if (!ap.base || !bp.base)
        outBase = Value();
      else if (ap.base == bp.base)
        outBase = ap.base;
      else
        outBase = Value();

      // Uniform part: keep if identical else drop
      UniformLinearExpr outU =
          (ap.uniform == bp.uniform) ? ap.uniform : UniformLinearExpr();

      int64_t outStride = (ap.laneStride == bp.laneStride) ? ap.laneStride : 0;
      return getPointwise(outBase, outU, outStride);
    }

    return getOverdefined();
  }

  void print(llvm::raw_ostream &os) const {
    os << "LaneInfo(";
    switch (kind) {
    case Uninitialized:
      os << "Uninitialized";
      break;
    case Overdefined:
      os << "Overdefined";
      break;
    case Uniform: {
      os << "Uniform";
      if (base)
        os << ", base=" << base;
      os << ", u=" << uniform;
      break;
    }
    case Pointwise: {
      os << "Pointwise";
      os << ", base=" << base;
      os << ", u=" << uniform;
      if (laneStride != 0)
        os << ", laneStride=" << laneStride;
      break;
    }
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
    propagateIfChanged(
        lattice, lattice->join(
                     LaneInfo::getPessimisticValueState(lattice->getAnchor())));
  }

  LogicalResult visitOperation(mlir::Operation *op,
                               llvm::ArrayRef<const LaneLattice *> operands,
                               llvm::ArrayRef<LaneLattice *> results) override;

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<LaneLattice *> argLattices,
                                    unsigned firstIndex) override;

private:
  void setAllUnknown(llvm::ArrayRef<LaneLattice *> results) {
    for (auto *r : results)
      if (r)
        propagateIfChanged(r, r->join(LaneInfo::getOverdefined()));
  }
};

bool isPointwiseStore(triton::StoreOp storeOp, LaneMapAnalysis &analysis);
bool isPointwiseLoad(triton::LoadOp loadOp, LaneMapAnalysis &analysis);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITONCPU_LANEMAPANALYSIS_H
