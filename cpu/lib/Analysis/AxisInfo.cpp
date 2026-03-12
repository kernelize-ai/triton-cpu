#include "cpu/include/Analysis/AxisInfo.h"
#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {

namespace {

class GenericOpAxisInfoVisitor final : public triton::AxisInfoVisitor {
public:
  using triton::AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    llvm::errs() << "got here?\n";
    assert(false && "TODO");
  }

  bool match(Operation *op) final { return isa<cpu::GenericOp>(op); }
};

} // namespace

void AxisInfoCpu::addVisitors(triton::AxisInfoVisitorList &visitors) {
  visitors.append<GenericOpAxisInfoVisitor>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
