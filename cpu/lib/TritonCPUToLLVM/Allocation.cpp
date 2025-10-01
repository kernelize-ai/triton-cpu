#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

#include "TargetInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYCPU
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace mlir::triton::cpu {

std::function<unsigned(Operation *)>
getCPUAllocationAnalysisScratchSize(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    return mlir::triton::defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}

} // namespace mlir::triton::cpu

namespace {

struct AllocateSharedMemoryCPU
    : public mlir::triton::cpu::impl::AllocateSharedMemoryCPUBase<
          AllocateSharedMemoryCPU> {
  using AllocateSharedMemoryCPUBase::AllocateSharedMemoryCPUBase;

  AllocateSharedMemoryCPU() : AllocateSharedMemoryCPUBase() {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::triton::cpu::TargetInfo targetInfo;
    ModuleAllocation allocation(
        mod,
        mlir::triton::cpu::getCPUAllocationAnalysisScratchSize(targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace

namespace mlir::triton::cpu {
std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass() {
  return std::make_unique<AllocateSharedMemoryCPU>();
}
} // namespace mlir::triton::cpu
