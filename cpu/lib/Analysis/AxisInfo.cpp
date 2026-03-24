#include "cpu/include/Analysis/AxisInfo.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Utility.h"

#define DEBUG_TYPE "axis-info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

namespace {

class MakeDynamicRangeOpAxisInfoVisitor final : public triton::AxisInfoVisitor {
public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    auto makeDynamicRangeOp = cast<cpu::MakeDynamicRangeOp>(op);
    auto end = makeDynamicRangeOp.getEnd();
    return AxisInfo(/*contiguity=*/{end},
                    /*divisibility=*/{highestPowOf2Divisor(0)},
                    /*constancy=*/{1});
  }

  bool match(Operation *op) final { return isa<cpu::MakeDynamicRangeOp>(op); }
};

} // namespace

CpuAxisInfoAnalysis::CpuAxisInfoAnalysis(DataFlowSolver &solver)
    : triton::AxisInfoAnalysis(solver) {
  visitors.append<MakeDynamicRangeOpAxisInfoVisitor>();
}

void CpuAxisInfoAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor & /*successor*/,
    ValueRange /*nonSuccessorInputs*/,
    ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    visitForOpInductionVar(forOp, argLattices);
  } else if (auto genericOp = dyn_cast<cpu::GenericOp>(op)) {
    visitGenericOpArguments(genericOp, argLattices);
  } else {
    setAllToEntryStates(argLattices);
  }
}

void CpuAxisInfoAnalysis::visitGenericOpArguments(
    cpu::GenericOp genericOp,
    ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) {
  ProgramPoint *programPoint = getProgramPointAfter(genericOp);

  auto vectorShape = genericOp.getVectorShape();

  // handle the induction var / tile offset separately
  {
    AxisInfo::DimVectorT knownContiguity(1, 1);
    AxisInfo::DimVectorT knownDivisibility(1, 1);
    AxisInfo::DimVectorT knownConstancy(1, 1);
    knownDivisibility[0] = vectorShape[0]; // TODO: multi-dim support
    auto inductionVar =
        AxisInfo(knownContiguity, knownDivisibility, knownConstancy);
    propagateIfChanged(argLattices[0], argLattices[0]->join(inductionVar));
  }

  for (auto [operand, argLattice] :
       llvm::zip(genericOp->getOperands(),
                 argLattices.drop_front(genericOp.getNumInductionVars()))) {

    const AxisInfo &outer =
        getLatticeElementFor(programPoint, operand)->getValue();
    // If the operand's axis info hasn't been computed yet, skip.
    if (outer.getRank() == 0)
      continue;

    auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorTy) {
      auto elemTy = tensorTy.getElementType();
      if (auto ptrTy = dyn_cast<PointerType>(elemTy)) {
        elemTy = ptrTy.getPointeeType();
      }

      int rank = outer.getRank();
      AxisInfo::DimVectorT contiguity(rank);
      AxisInfo::DimVectorT divisibility(rank);
      AxisInfo::DimVectorT constancy(rank);

      for (int d = 0; d < rank; ++d) {
        int64_t tileSize = (d < (int)vectorShape.size()) ? vectorShape[d] : 1;
        int64_t tileSizeBytes = tileSize * elemTy.getIntOrFloatBitWidth() / 8;
        contiguity[d] = std::min(outer.getContiguity(d), tileSize);
        constancy[d] = std::min(outer.getConstancy(d), tileSize);
        divisibility[d] = std::gcd(outer.getDivisibility(d), tileSizeBytes);
      }

      auto axisInfo = AxisInfo(contiguity, divisibility, constancy,
                               outer.getConstantValue());
      LDBG("visitGenericOpArguments tensor " << operand
                                             << " updating axis info ");
      LLVM_DEBUG({ axisInfo.print(DBGS()); });
      propagateIfChanged(argLattice, argLattice->join(axisInfo));
    } else {
      LDBG("visitGenericOpArguments scalar " << operand
                                             << " updating axis info ");
      LLVM_DEBUG({
        outer.print(DBGS());
        DBGS() << "\n";
      });
      // propagate scalar op axis info from the operand to the body
      propagateIfChanged(argLattice, argLattice->join(outer));
    }
  }
}

AxisInfoAnalysis *CpuAxisInfoAnalysis::loadAnalysis(DataFlowSolver *solver) {
  return solver->load<CpuAxisInfoAnalysis>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
