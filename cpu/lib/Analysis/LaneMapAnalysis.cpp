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

// Updated signature to accept the result Value of the operation being visited
inline LaneInfo addElementwiseAdd(const LaneInfo &a, const LaneInfo &b,
                                  Value resultVal = Value()) {
  // 1. Handle Uninitialized/Overdefined propagation
  if (a.kind == LaneInfo::Uninitialized)
    return b;
  if (b.kind == LaneInfo::Uninitialized)
    return a;
  if (a.kind == LaneInfo::Overdefined || b.kind == LaneInfo::Overdefined)
    return LaneInfo::getOverdefined();

  // 2. Uniform + Uniform -> Uniform
  if (a.kind == LaneInfo::Uniform && b.kind == LaneInfo::Uniform)
    return LaneInfo::getUniform(UniformLinearExpr::add(a.uniform, b.uniform));

  // 3. Handle Mixed Cases (Uniform + Pointwise)
  // We want to preserve the base of the Pointwise operand.
  const LaneInfo *pPointwise = nullptr;
  const LaneInfo *pUniform = nullptr;

  if (a.kind == LaneInfo::Pointwise && b.kind == LaneInfo::Uniform) {
    pPointwise = &a;
    pUniform = &b;
  } else if (b.kind == LaneInfo::Pointwise && a.kind == LaneInfo::Uniform) {
    pPointwise = &b;
    pUniform = &a;
  }

  if (pPointwise) {
    UniformLinearExpr newU =
        UniformLinearExpr::add(pPointwise->uniform, pUniform->uniform);
    // Preserve the existing base and stride
    return LaneInfo::getPointwise(pPointwise->base, newU,
                                  pPointwise->laneStride);
  }

  // 4. Pointwise + Pointwise
  // Both are pointwise. We add their uniform offsets.
  UniformLinearExpr newU = UniformLinearExpr::add(a.uniform, b.uniform);

  // If bases match, preserve the base.
  if (a.base == b.base) {
    return LaneInfo::getPointwise(a.base, newU, a.laneStride);
  }

  // If bases differ (e.g. row_idx + col_idx), we cannot point to just one.
  // We set the NEW base to be the result of this operation.
  // This essentially says: "The variation starts from this add instruction."
  // The uniform parts (coeffs) are still extracted into 'newU'.
  if (resultVal) {
    return LaneInfo::getPointwise(resultVal, newU, 0);
  }

  // Fallback if no resultVal provided (shouldn't happen with correct usage)
  return LaneInfo::getPointwise(Value(), newU, 0);
}

// Helper to detect if a value is effectively constant (even if Pointwise)
bool isConstantValue(Value v) {
  if (!v)
    return false;
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isa<arith::ConstantOp>(op))
    return true;
  // Handle splat of constant
  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    if (auto def = splat.getSrc().getDefiningOp())
      return isa<arith::ConstantOp>(def);
  }
  return false;
}

// Generic merger for arithmetic ops
LaneInfo mergeArithmetic(Operation *op, const LaneInfo &lhs,
                         const LaneInfo &rhs) {
  // 1. Handle Uninitialized/Overdefined
  if (lhs.kind == LaneInfo::Uninitialized)
    return rhs;
  if (rhs.kind == LaneInfo::Uninitialized)
    return lhs;
  if (lhs.kind == LaneInfo::Overdefined || rhs.kind == LaneInfo::Overdefined)
    return LaneInfo::getOverdefined();

  // 2. Uniform + Uniform
  if (lhs.kind == LaneInfo::Uniform && rhs.kind == LaneInfo::Uniform) {
    // For simplicity, we only strictly track ADD for the uniform expr.
    // For others, we might reset 'u' or try to combine if your
    // UniformLinearExpr supports it. Assuming 'add' is the primary concern for
    // pointers:
    if (isa<arith::AddIOp, arith::AddFOp, triton::AddPtrOp>(op)) {
      return LaneInfo::getUniform(
          UniformLinearExpr::add(lhs.uniform, rhs.uniform));
    }
    // For mul/rem/etc between uniforms, you might just return a clean
    // Uniform(0) or calculate it if you have the facility.
    return LaneInfo::getUniform();
  }

  // 3. Identify Base and Offset
  const LaneInfo *pBase = nullptr; // The operand providing the base
  const LaneInfo *pOff =
      nullptr; // The operand providing the offset (Uniform or Constant)

  // Case A: Pointwise + Uniform
  if (lhs.kind == LaneInfo::Pointwise && rhs.kind == LaneInfo::Uniform) {
    pBase = &lhs;
    pOff = &rhs;
  } else if (rhs.kind == LaneInfo::Pointwise && lhs.kind == LaneInfo::Uniform) {
    pBase = &rhs;
    pOff = &lhs;
  }
  // Case B: Pointwise + Constant Pointwise (Fix for the Loop Issue)
  else if (lhs.kind == LaneInfo::Pointwise && rhs.kind == LaneInfo::Pointwise) {
    if (isConstantValue(rhs.base)) {
      pBase = &lhs;
      pOff = &rhs;
    } else if (isConstantValue(lhs.base)) {
      pBase = &rhs;
      pOff = &lhs;
    } else if (lhs.base == rhs.base) {
      // Bases match, standard merge
      pBase = &lhs;
      pOff = &rhs;
    }
  }

  // If we found a valid Base + Offset pattern and the Op is linear
  // (Add/Sub/AddPtr)
  if (pBase && isa<arith::AddIOp, arith::AddFOp, triton::AddPtrOp>(op)) {
    // We can merge them into the existing base
    UniformLinearExpr newU =
        UniformLinearExpr::add(pBase->uniform, pOff->uniform);
    return LaneInfo::getPointwise(pBase->base, newU, pBase->laneStride);
  }

  // 4. Fallback: Create a NEW Base
  // For Mul, Rem, Div, or conflicting bases, we set the result as the new base.
  // This is crucial: Never return NULL base for a valid operation.
  return LaneInfo::getPointwise(op->getResult(0), UniformLinearExpr(), 0);
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

  // Handle ExpandDims / Broadcast (Ensure they set base=result if needed)
  if (isa<triton::ExpandDimsOp, triton::BroadcastOp>(op)) {
    LaneInfo src = getOperand(0);
    if (src.kind == LaneInfo::Pointwise) {
      // If tracking through dims is supported, do so.
      // Otherwise, start a new base to be safe.
      joinToAll(
          LaneInfo::getPointwise(op->getResult(0), UniformLinearExpr(), 0));
    } else {
      joinToAll(src); // Propagate Uniform
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

    // Handle Arithmetic and Pointer Arithmetic
    if (isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
            arith::MulIOp, arith::DivSIOp, arith::RemSIOp>(op)) {
      LaneInfo lhs = getOperand(0);
      LaneInfo rhs = getOperand(1);
      joinToAll(mergeArithmetic(op, lhs, rhs));
      return success();
    }
#if 0
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
#endif
    // Default conservative behavior for other elementwise ops:
    // keep pointwise if possible, else unknown.
    joinToAll(LaneInfo::join(a, b));
    return success();
  }

#if 1
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
    LaneInfo lhs = getOperand(0);
    LaneInfo rhs = getOperand(1);
    joinToAll(mergeArithmetic(op, lhs, rhs));
    return success();
  }
#else
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
    LaneInfo base = getOperand(0);
    LaneInfo offs = getOperand(1);
    joinToAll(addElementwiseAdd(base, offs, op->getResult(0)));
    return success();
  }
#endif

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
