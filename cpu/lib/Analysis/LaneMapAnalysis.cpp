#include "cpu/include/Analysis/LaneMapAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritoncpu-lane-map-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace cpu {

LogicalResult
LaneMapAnalysis::visitOperation(Operation *op,
                                ArrayRef<const LaneLattice *> operands,
                                ArrayRef<LaneLattice *> results) {
  LDBG("Visiting operation " << *op);
  LLVM_DEBUG({
    for (auto *operand : operands) {
      if (operand)
        DBGS() << "Operand " << *operand << " has value " << operand->getValue()
               << "\n";
    }
    for (auto *result : results) {
      if (result)
        DBGS() << "Result " << *result << " has value " << result->getValue()
               << "\n";
    }
  });

  auto getOperand = [&](unsigned i) -> LaneInfo {
    return (i < operands.size() && operands[i]) ? operands[i]->getValue()
                                                : LaneInfo::getUnknown();
  };
  auto joinToAll = [&](const LaneInfo &v) {
    for (auto *r : results) {
      if (r) {
        LDBG("Propagating to result " << *r << " with value "
                                      << LaneInfo::join(r->getValue(), v));
        propagateIfChanged(r, r->join(v));
      }
    }
  };

  // No results => nothing to do.
  if (results.empty())
    return success();

  if (isa<arith::ConstantOp>(op)) {
    // constants are always uniform across lanes
    // TODO: keep the base scalar?
    joinToAll(LaneInfo::getUniform());
    return success();
  }

  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    // splat is uniform across lanes, keep the baseScalar
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

  // For elementwise ops, joining operand lane-forms is the canonical,
  // conservative rule.
  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::CmpIOp,
          arith::CmpFOp, arith::SelectOp>(op)) {
    LaneInfo a = operands[0] ? operands[0]->getValue() : LaneInfo::getUnknown();
    LaneInfo b = operands[1] ? operands[1]->getValue() : LaneInfo::getUnknown();
    LaneInfo joined = LaneInfo::join(a, b);
    joinToAll(joined);
    return success();
  }

  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
    LaneInfo base =
        operands[0] ? operands[0]->getValue() : LaneInfo::getUnknown();
    LaneInfo offs =
        operands[1] ? operands[1]->getValue() : LaneInfo::getUnknown();
    LaneInfo joined = LaneInfo::join(base, offs);
    joinToAll(joined);
    return success();
  }

  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    LaneInfo ptr =
        operands[0] ? operands[0]->getValue() : LaneInfo::getUnknown();
    // load keeps the lane mapping of the pointer
    LDBG("Load op ptr lane info: " << ptr);
    joinToAll(ptr);
    return success();
  }

  if (isa<triton::StoreOp>(op)) {
    joinToAll(LaneInfo::getUnknown());
    return success();
  }

  if (isa<cpu::BlockStartOp>(op) || isa<cpu::BlockEndOp>(op)) {
    joinToAll(LaneInfo::getUniform(op->getResult(0)));
  }

  // Unknown op kind => unknown
  setAllUnknown(results);
  return success();
}

bool isPointwiseStore(triton::StoreOp storeOp, LaneMapAnalysis &analysis) {
  LDBG("Evaluating storeOp for pointwise store");

  auto *ptrLat = analysis.getLatticeElement(storeOp.getPtr());
  auto *valLat = analysis.getLatticeElement(storeOp.getValue());

  if (!ptrLat || !valLat)
    return false;

  const LaneInfo &p = ptrLat->getValue();
  const LaneInfo &v = valLat->getValue();
  LDBG("StoreOp ptr lane info: " << p);
  LDBG("StoreOp value lane info: " << v);
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

} // namespace cpu
} // namespace triton
} // namespace mlir
