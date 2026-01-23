#include "cpu/include/Analysis/LaneMapAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

// TODO: it would be nice to embed these all in the generic LaneInfo.join
LaneInfo LaneMapAnalysis::evalAddI(const LaneInfo &a, const LaneInfo &b) {
  // Affine + constant uniform => affine with shifted c
  if (a.kind == LaneInfo::AffineLane && b.kind == LaneInfo::Uniform) {
    if (auto k = getI64Const(b.baseScalar))
      return LaneInfo::getAffine(a.baseScalar, a.constOffset + *k, a.stride);
  }
  if (b.kind == LaneInfo::AffineLane && a.kind == LaneInfo::Uniform) {
    if (auto k = getI64Const(a.baseScalar))
      return LaneInfo::getAffine(b.baseScalar, b.constOffset + *k, b.stride);
  }

  // Affine + affine: only if bases match (or one base is null)
  if (a.kind == LaneInfo::AffineLane && b.kind == LaneInfo::AffineLane) {
    if (a.baseScalar == b.baseScalar)
      return LaneInfo::getAffine(a.baseScalar, a.constOffset + b.constOffset,
                                 a.stride + b.stride);
  }

  // Otherwise conservative join
  return LaneInfo::join(a, b);
}

LaneInfo LaneMapAnalysis::evalSubI(const LaneInfo &a, const LaneInfo &b) {
  if (a.kind == LaneInfo::AffineLane && b.kind == LaneInfo::Uniform) {
    if (auto k = getI64Const(b.baseScalar))
      return LaneInfo::getAffine(a.baseScalar, a.constOffset - *k, a.stride);
  }
  if (a.kind == LaneInfo::AffineLane && b.kind == LaneInfo::AffineLane) {
    if (a.baseScalar == b.baseScalar)
      return LaneInfo::getAffine(a.baseScalar, a.constOffset - b.constOffset,
                                 a.stride - b.stride);
  }
  return LaneInfo::join(a, b);
}

LaneInfo LaneMapAnalysis::evalMulI(const LaneInfo &a, const LaneInfo &b) {
  // Only handle multiply by constant
  if (a.kind == LaneInfo::AffineLane && b.kind == LaneInfo::Uniform) {
    if (auto k = getI64Const(b.baseScalar))
      return LaneInfo::getAffine(a.baseScalar, a.constOffset * (*k),
                                 a.stride * (*k));
  }
  if (b.kind == LaneInfo::AffineLane && a.kind == LaneInfo::Uniform) {
    if (auto k = getI64Const(a.baseScalar))
      return LaneInfo::getAffine(b.baseScalar, b.constOffset * (*k),
                                 b.stride * (*k));
  }
  return LaneInfo::join(a, b);
}

LaneInfo LaneMapAnalysis::evalAddPtr(const LaneInfo &basePtr,
                                     const LaneInfo &offs) {
  // The useful case: basePtr is uniform scalar pointer; offs is affine lane
  // (i32 offsets).
  if (offs.kind != LaneInfo::AffineLane)
    return LaneInfo::getUnknown();

  if (basePtr.kind == LaneInfo::Uniform) {
    // ptr(i) = basePtr + (offs.c + offs.s * lane)
    // treat as affine with baseScalar = base pointer and c/s from offs.
    return LaneInfo::getAffine(basePtr.baseScalar, offs.constOffset,
                               offs.stride);
  }

  // If basePtr itself is affine, you can combine if same baseScalar (rare).
  if (basePtr.kind == LaneInfo::AffineLane &&
      basePtr.baseScalar == offs.baseScalar) {
    return LaneInfo::getAffine(basePtr.baseScalar,
                               basePtr.constOffset + offs.constOffset,
                               basePtr.stride + offs.stride);
  }

  return LaneInfo::getUnknown();
}

LogicalResult
LaneMapAnalysis::visitOperation(Operation *op,
                                ArrayRef<const LaneLattice *> operands,
                                ArrayRef<LaneLattice *> results) {
  auto getOperand = [&](unsigned i) -> LaneInfo {
    return (i < operands.size() && operands[i]) ? operands[i]->getValue()
                                                : LaneInfo::getUnknown();
  };
  auto joinToAll = [&](const LaneInfo &v) {
    for (auto *r : results)
      if (r)
        propagateIfChanged(r, r->join(v));
  };

  // No results => nothing to do.
  if (results.empty())
    return success();

  if (isa<arith::ConstantOp>(op)) {
    // Keep baseScalar as the constant SSA value, so evalAddI/MulI can read it.
    joinToAll(LaneInfo::getUniform(op->getResult(0)));
    return success();
  }

  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    // tt.splat %scalar => uniform with that scalar
    joinToAll(LaneInfo::getUniform(splat.getSrc()));
    return success();
  }

  if (auto makeRange = dyn_cast<triton::MakeRangeOp>(op)) {
    int32_t start = makeRange.getStart();
    // Result lanes: start + laneId
    joinToAll(
        LaneInfo::getAffine(/*baseScalar=*/Value(), /*c=*/start, /*s=*/1));
    return success();
  }

  if (isa<arith::AddIOp>(op)) {
    joinToAll(evalAddI(getOperand(0), getOperand(1)));
    return success();
  }
  if (isa<arith::SubIOp>(op)) {
    joinToAll(evalSubI(getOperand(0), getOperand(1)));
    return success();
  }
  if (isa<arith::MulIOp>(op)) {
    joinToAll(evalMulI(getOperand(0), getOperand(1)));
    return success();
  }

  if (isa<arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op)) {
    // Compare/select/etc: pointwise if operands are pointwise
    joinToAll(LaneInfo::join(getOperand(0), getOperand(1)));
    return success();
  }

  if (isa<triton::BroadcastOp, triton::ExpandDimsOp>(op)) {
    // Simple shape ops that preserve lane mapping (conservative forward)
    joinToAll(getOperand(0));
    return success();
  }

  if (isa<triton::AddPtrOp>(op)) {
    joinToAll(evalAddPtr(getOperand(0), getOperand(1)));
    return success();
  }

  if (isa<triton::LoadOp>(op)) {
    // result lanes follow pointer lanes
    joinToAll(getOperand(0));
    return success();
  }

  if (isa<triton::StoreOp>(op)) {
    // stores have no results; nothing to propagate.
    return success();
  }

  // Unknown op kind => unknown
  setAllUnknown(results);
  return success();
}

bool mlir::triton::isPointwiseStore(triton::StoreOp storeOp,
                                    LaneMapAnalysis &analysis) {
  auto *ptrLat = analysis.getLatticeElement(storeOp.getPtr());
  auto *valLat = analysis.getLatticeElement(storeOp.getValue());

  if (!ptrLat || !valLat)
    return false;

  const LaneInfo &p = ptrLat->getValue();
  const LaneInfo &v = valLat->getValue();

  if (p.kind == LaneInfo::Unknown || v.kind == LaneInfo::Unknown)
    return false;

  if (p == v)
    return true;

  if (p.kind == LaneInfo::Uniform && v.kind == LaneInfo::AffineLane)
    return true;
  if (v.kind == LaneInfo::Uniform && p.kind == LaneInfo::AffineLane)
    return true;

  return false;
}
