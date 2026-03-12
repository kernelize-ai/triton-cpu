#ifndef TRITON_CPU_ANALYSIS_AXISINFO_H
#define TRITON_CPU_ANALYSIS_AXISINFO_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
namespace cpu {

struct AxisInfoCpu {
  static void addVisitors(triton::AxisInfoVisitorList &visitors);
};

class ModuleAxisInfoAnalysis : public triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : triton::ModuleAxisInfoAnalysis(moduleOp, AxisInfoCpu::addVisitors) {}
};

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
