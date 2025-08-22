#include "npu/include/TritonNPUToLLVM/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

#include "TargetInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYNPU
#include "npu/include/TritonNPUToLLVM/Passes.h.inc"
} // namespace npu
} // namespace triton
} // namespace mlir

namespace mlir::triton::npu {

std::function<unsigned(Operation *)>
getNPUAllocationAnalysisScratchSize(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    return 0; // TODO
  };
  return allocation;
}

} // namespace mlir::triton::npu

namespace {

// TODO: copy from proton
struct AllocateSharedMemoryNPU
    : public mlir::triton::npu::impl::AllocateSharedMemoryNPUBase<
          AllocateSharedMemoryNPU> {
  using AllocateSharedMemoryNPUBase::AllocateSharedMemoryNPUBase;

  AllocateSharedMemoryNPU() : AllocateSharedMemoryNPUBase() {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::triton::npu::TargetInfo targetInfo;
    ModuleAllocation allocation(
        mod,
        mlir::triton::npu::getNPUAllocationAnalysisScratchSize(targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace

namespace mlir::triton::npu {
std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass() {
  return std::make_unique<AllocateSharedMemoryNPU>();
}
} // namespace mlir::triton::npu
