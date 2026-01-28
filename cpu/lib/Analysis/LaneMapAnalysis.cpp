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

namespace {

// Cases:
// Uniform + Uniform -> Uniform(sum)
// Pointwise(baseX) + Uniform(u) -> Pointwise(baseX, uniform += u)
// Uniform + Pointwise(baseY) -> Pointwise(baseY, uniform += u)
// Pointwise(baseX) + Pointwise(baseX) with same base -> Pointwise(baseX,
// uniform += uniform) otherwise -> Unknown
inline LaneInfo addElementwiseAdd(const LaneInfo &a, const LaneInfo &b) {
// If either is ⊥, treat as no information; the framework join will handle it,
  // but returning the other is often fine too.
  if (a.kind == LaneInfo::Uninitialized) return b;
  if (b.kind == LaneInfo::Uninitialized) return a;

  if (a.kind == LaneInfo::Overdefined || b.kind == LaneInfo::Overdefined)
    return LaneInfo::getOverdefined();

  if (a.kind == LaneInfo::Uniform && b.kind == LaneInfo::Uniform)
    return LaneInfo::getUniform(UniformLinearExpr::add(a.uniform, b.uniform));

  auto toPointwise = [](const LaneInfo &x) {
    if (x.kind == LaneInfo::Uniform) return LaneInfo::getPointwise(Value(), x.uniform, 0);
    return x;
  };
  LaneInfo ap = toPointwise(a);
  LaneInfo bp = toPointwise(b);

  UniformLinearExpr outU = UniformLinearExpr::add(ap.uniform, bp.uniform);

  Value outBase;
  if (ap.kind == LaneInfo::Pointwise && bp.kind == LaneInfo::Pointwise &&
      ap.base && bp.base && ap.base == bp.base)
    outBase = ap.base;
  else
    outBase = Value(); // base-agnostic pointwise (safe fallback)

  return LaneInfo::getPointwise(outBase, outU, /*laneStride=*/0);
}

} // namespace

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
                                                : LaneInfo::getUninitialized();
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

  if (results.empty())
    return success();

  if (isa<arith::ConstantOp>(op)) {
    auto result = op->getResult(0);
    if (isa<TensorType>(result.getType())) {
      joinToAll(LaneInfo::getPointwise(result));
    } else {
      joinToAll(LaneInfo::getUniform(UniformLinearExpr::fromValue(result)));
    }
    return success();
  }

  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    LaneInfo src = getOperand(0);
    if (src.kind == LaneInfo::Uniform) {
      joinToAll(LaneInfo::getUniform(src.uniform));
    } else {
      // splat of non-uniform is a pointwise base
      joinToAll(LaneInfo::getPointwise(op->getResult(0)));
    }
    return success();
  }

  if (auto makeRange = dyn_cast<triton::MakeRangeOp>(op)) {
    int32_t start = makeRange.getStart();
    joinToAll(LaneInfo::getPointwise(op->getResult(0),
                                     UniformLinearExpr::fromConst(start)));
    return success();
  }

  if (isa<triton::BroadcastOp, triton::ExpandDimsOp>(op)) {
    LaneInfo in = getOperand(0);
    if (in.kind == LaneInfo::Pointwise) {
      joinToAll(
          LaneInfo::getPointwise(op->getResult(0), in.uniform, in.laneStride));
    } else {
      joinToAll(in);
    }
    return success();
  }

  // uniform across all lanes
  if (isa<cpu::BlockStartOp>(op) || isa<cpu::BlockEndOp>(op) ||
      isa<cpu::CurrentBlockOp>(op)) {
    joinToAll(
        LaneInfo::getUniform(UniformLinearExpr::fromValue(op->getResult(0))));
    return success();
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
    LaneInfo a = getOperand(0);
    LaneInfo b = getOperand(1);
    LDBG("Cmp operands " << a << " " << cast<arith::CmpIOp>(op).getPredicate()
                         << " " << b);

    // a mask of two uniforms is uniform
    if (a.kind == LaneInfo::Uniform && b.kind == LaneInfo::Uniform) {
      joinToAll(LaneInfo::getUniform()); // "some uniform i1"
      return success();
    }

    // otherwise the mask is lane varying (the more typical case, e.g. if
    // tensor[i] < num_elems)
    joinToAll(LaneInfo::getPointwise(op->getResult(0)));
    return success();
  }

  // general elementwise arithmetic operations with special handling for Add
  if (isa<arith::ArithDialect>(op->getDialect()) &&
      op->hasTrait<OpTrait::Elementwise>()) {
    // Unary elementwise: just forward operand for now.
    if (op->getNumOperands() == 1) {
      joinToAll(getOperand(0));
      return success();
    }

    LaneInfo a = getOperand(0);
    LaneInfo b = getOperand(1);

    if (isa<arith::AddIOp, arith::AddFOp>(op)) {
      joinToAll(addElementwiseAdd(a, b));
      return success();
    }

    if (isa<arith::SubIOp, arith::SubFOp>(op)) {
      // Similar to add, but negate rhs uniform.
      // TODO: use addElementwiseAdd but find some way to negate the second
      // operand
      auto negate = [](UniformLinearExpr u) {
        u.c = -u.c;
        for (auto &kv : u.coeffs)
          kv.second = -kv.second;
        return u;
      };

      if (a.kind == LaneInfo::Uninitialized || b.kind == LaneInfo::Uninitialized) {
        joinToAll(LaneInfo::getUninitialized());
        return success();
      }

      if (a.kind == LaneInfo::Uniform && b.kind == LaneInfo::Uniform) {
        joinToAll(LaneInfo::getUniform(
            UniformLinearExpr::add(a.uniform, negate(b.uniform))));
        return success();
      }

      if (a.kind == LaneInfo::Pointwise && b.kind == LaneInfo::Uniform) {
        joinToAll(LaneInfo::getPointwise(
            a.base, UniformLinearExpr::add(a.uniform, negate(b.uniform)),
            a.laneStride));
        return success();
      }

      // Uniform - Pointwise or Pointwise - Pointwise: not representable in this
      // minimal lattice.
      joinToAll(LaneInfo::getOverdefined());
      return success();
    }

    // Default conservative behavior for other elementwise ops:
    // keep pointwise if possible, else unknown.
    joinToAll(LaneInfo::join(a, b));
    return success();
  }

  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
    LaneInfo base = getOperand(0);
    LaneInfo offs = getOperand(1);
    joinToAll(addElementwiseAdd(base, offs));
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
    joinToAll(LaneInfo::getOverdefined());
    return success();
  }

  // Unknown op kind => unknown
  setAllUnknown(results);
  return success();
}

void LaneMapAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<LaneLattice *> argLattices, unsigned firstIndex) {
#if 0
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
#endif
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
  if (p.kind == LaneInfo::Overdefined || v.kind == LaneInfo::Overdefined)
    return false;

  if (storeOp.getMask()) {
    auto *maskLat = analysis.getLatticeElement(storeOp.getMask());
    if (!maskLat)
      return false;
    const LaneInfo &m = maskLat->getValue();
    LDBG("StoreOp mask lane info: " << m);
    if (m.kind == LaneInfo::Overdefined)
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
  if (p.kind == LaneInfo::Overdefined)
    return false;

  if (!loadOp.getMask())
    return true;
  // check the mask
  auto *mask = analysis.getLatticeElement(loadOp.getMask());

  if (!mask)
    return false;

  const LaneInfo &m = mask->getValue();
  LDBG("LoadOp mask lane info: " << m);
  if (m.kind == LaneInfo::Overdefined)
    return false;

  return true;
}

} // namespace cpu
} // namespace triton
} // namespace mlir
