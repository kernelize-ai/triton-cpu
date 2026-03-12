#include "cpu/include/Analysis/AxisInfo.h"
#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include <numeric>

namespace mlir {
namespace triton {
namespace cpu {

namespace {

class GenericOpNCFAVisitor final : public triton::AxisInfoNCFAVisitor {
public:
  bool match(Operation *op) final { return isa<cpu::GenericOp>(op); }

  void
  visit(Operation *op, ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices,
        function_ref<const dataflow::Lattice<AxisInfo> *(Value)> lookupLattice,
        function_ref<void(dataflow::Lattice<AxisInfo> *, AxisInfo)> propagate)
      final {
    auto genericOp = cast<cpu::GenericOp>(op);
    auto vectorShape = genericOp.getVectorShape();

    for (auto [operand, argLattice] :
         llvm::zip(op->getOperands(), argLattices)) {
      const AxisInfo &outer = lookupLattice(operand)->getValue();
      // If the operand's axis info hasn't been computed yet, skip.
      if (outer.getRank() == 0)
        continue;

      auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorTy)
        continue;

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

      propagate(argLattice, AxisInfo(contiguity, divisibility, constancy,
                                     outer.getConstantValue()));
    }
  }
};

} // namespace

void AxisInfoCpu::addVisitors(triton::AxisInfoVisitorList &visitors) {}

void AxisInfoCpu::addNCFAVisitors(triton::AxisInfoNCFAVisitorList &visitors) {
  visitors.append<GenericOpNCFAVisitor>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
