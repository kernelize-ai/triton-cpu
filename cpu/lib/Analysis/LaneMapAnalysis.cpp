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
    joinToAll(LaneInfo::getUniform(op->getResult(0)));
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

  if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
    joinToAll(getOperand(0));
    return success();
  }

  if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(op)) {
    joinToAll(getOperand(0));
    return success();
  }

  if (isa<arith::CmpIOp>(op)) {
    LaneInfo a = getOperand(0);
    LaneInfo b = getOperand(1);
    LDBG("Cmp operands " << a << " " << cast<arith::CmpIOp>(op).getPredicate()
                         << " " << b);
    // TODO: this specifically matches the mask compare order. Are there others
    // we should match?
    if (a.kind == LaneInfo::AffineLane && b.kind == LaneInfo::Uniform) {
      LaneInfo joined = LaneInfo::getAffine(
          /*baseScalar=*/op->getResult(0), /*c=*/a.constOffset, /*s=*/a.stride);
      joinToAll(joined);
      return success();
    }
    joinToAll(LaneInfo::getUnknown());
    return success();
  }

  // For elementwise ops, joining operand lane-forms is the canonical,
  // conservative rule.
  if (isa<arith::ArithDialect>(op->getDialect()) &&
      op->hasTrait<OpTrait::Elementwise>()) {
    LaneInfo a = getOperand(0);
    LaneInfo b = getOperand(1);
    const bool hasLoadOperand =
        llvm::any_of(op->getOperands(), [](Value operand) {
          if (!operand.getDefiningOp())
            return false;
          return isa<triton::LoadOp>(operand.getDefiningOp());
        });
    if (hasLoadOperand) {
      a = a.kind == LaneInfo::AffineLane
              ? LaneInfo::getAffine(/*baseScalar=*/Value(), /*c=*/a.constOffset,
                                    /*s=*/a.stride)
              : a;
      b = b.kind == LaneInfo::AffineLane
              ? LaneInfo::getAffine(/*baseScalar=*/Value(), /*c=*/b.constOffset,
                                    /*s=*/b.stride)
              : b;
    }
    LaneInfo joined = LaneInfo::join(a, b);
    joinToAll(joined);
    return success();
  }

  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
#if 1
    LaneInfo offs = getOperand(1);
    joinToAll(offs);
#else
    LaneInfo base = getOperand(0);
    LaneInfo offs = getOperand(1);
    LaneInfo joined = LaneInfo::join(base, offs);
    joinToAll(joined);
#endif
    return success();
  }

  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    LaneInfo ptr = getOperand(0);
    // load keeps the lane mapping of the pointer
    LDBG("Load op ptr lane info: " << ptr);
    joinToAll(ptr);
    return success();
  }

  if (isa<triton::StoreOp>(op)) {
    joinToAll(LaneInfo::getUnknown());
    return success();
  }

  // block index is uniform across all lanes
  if (isa<cpu::BlockStartOp>(op) || isa<cpu::BlockEndOp>(op) ||
      isa<cpu::CurrentBlockOp>(op)) {
    joinToAll(LaneInfo::getUniform(op->getResult(0)));
  }

  // Unknown op kind => unknown
  setAllUnknown(results);
  return success();
}

void LaneMapAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<LaneLattice *> argLattices, unsigned firstIndex) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto getLaneOrUnknown = [](const LaneLattice *lat) {
      return lat ? lat->getValue() : LaneInfo::getUnknown();
    };

    Block *body = forOp.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    llvm::errs() << "yield = " << yield << "\n";
    for (auto [idx, arg] : llvm::enumerate(body->getArguments())) {
      llvm::errs() << "checkout arg: " << idx << " ==> " << arg << "\n";
      if (idx == 0) {
        // induction var
        continue;
      }

      LaneInfo fromInit =
          getLaneOrUnknown(getLatticeElementFor(getProgramPointAfter(op), arg));
      llvm::errs() << "fromInitLat: " << fromInit << "\n";

      auto yieldVal = yield.getOperand(idx - forOp.getNumInductionVars());
      auto yieldLat = getLatticeElementFor(getProgramPointAfter(op), yieldVal);
      LaneInfo yieldInfo = getLaneOrUnknown(yieldLat);

      // LaneInfo merged = LaneInfo::join(fromInit, fromYield);
      llvm::errs() << "yieldVal: " << yieldVal << "\n";
      llvm::errs() << "fromYieldLat: " << yieldInfo << "\n";

      // TODO: how can we merge fromInit and yieldInfo?
    }
    llvm::errs() << "done\n";
  }
  setAllToEntryStates(argLattices.take_front(firstIndex));
  setAllToEntryStates(argLattices.drop_front(
      firstIndex + successor.getSuccessorInputs().size()));
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

  if (storeOp.getMask()) {
    auto *maskLat = analysis.getLatticeElement(storeOp.getMask());
    if (!maskLat)
      return false;
    const LaneInfo &m = maskLat->getValue();
    LDBG("StoreOp mask lane info: " << m);
    if (m.kind == LaneInfo::Unknown)
      return false;
    // since the mask is applied elementwise any other combo of mask laneinfo +
    // p/v laneinfo should match the p/v result.
  }

  return true;
}

bool isPointwiseLoad(triton::LoadOp loadOp, LaneMapAnalysis &analysis) {
  LDBG("Evaluating loadOp for pointwise load");

  auto *ptr = analysis.getLatticeElement(loadOp.getPtr());

  if (!ptr)
    return false;

  const LaneInfo &p = ptr->getValue();
  LDBG("LoadOp ptr lane info: " << p);
  if (p.kind == LaneInfo::Unknown)
    return false;

  if (!loadOp.getMask())
    return true;
  // check the mask
  auto *mask = analysis.getLatticeElement(loadOp.getMask());

  if (!mask)
    return false;

  const LaneInfo &m = mask->getValue();
  LDBG("LoadOp mask lane info: " << m);
  if (m.kind == LaneInfo::Unknown)
    return false;

  return true;
}

} // namespace cpu
} // namespace triton
} // namespace mlir
