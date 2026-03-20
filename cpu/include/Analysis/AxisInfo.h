#ifndef TRITON_CPU_ANALYSIS_AXISINFO_H
#define TRITON_CPU_ANALYSIS_AXISINFO_H

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
namespace cpu {

class CpuAxisInfoAnalysis : public triton::AxisInfoAnalysis {
public:
  CpuAxisInfoAnalysis(DataFlowSolver &solver)
      : triton::AxisInfoAnalysis(solver) {}

  static AxisInfoAnalysis *loadAnalysis(DataFlowSolver *solver);

protected:
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor & /*successor*/,
      ValueRange /*nonSuccessorInputs*/,
      ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) override;

  void
  visitGenericOpArguments(cpu::GenericOp genericOp,
                          ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices);
};

class ModuleAxisInfoAnalysis : public triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : triton::ModuleAxisInfoAnalysis(moduleOp,
                                       CpuAxisInfoAnalysis::loadAnalysis) {}
};

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
