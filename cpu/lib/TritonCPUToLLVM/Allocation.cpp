#include "Allocation.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Alignment.h"

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

namespace {

bool reductionNeedsLaneExchange(RankedTensorType srcTy, unsigned axis) {
  auto *ctx = srcTy.getContext();
  auto kLane = StringAttr::get(ctx, "lane");

  auto reduced = gpu::toLinearLayout(srcTy);
  auto it = reduced.getBases().find(kLane);
  if (it == reduced.getBases().end())
    return false;

  for (auto &base : it->second) {
    if (base[axis] != 0)
      return true;
  }
  return false;
}

unsigned getReductionScratchSizeForLaneExchange(RankedTensorType srcTy) {
  auto *ctx = srcTy.getContext();
  auto kLane = StringAttr::get(ctx, "lane");

  auto reduced = gpu::toLinearLayout(srcTy);
  unsigned elemSizeInBits = srcTy.getElementType().getIntOrFloatBitWidth();
  return reduced.getInDimSize(kLane) * elemSizeInBits / 8;
}

} // namespace

std::function<unsigned(Operation *)>
getCPUAllocationAnalysisScratchSize(TargetInfo &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    // NOTE: these allocations are 128-bit aligned by default
    // see AllocationAnalysis::getScratchValueSize

    // pad all per-op shared memory allocations to 64-byte alignment so the
    // barrier synchronization buffers are properly aligned

    if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      ReduceOpHelper helper(reduceOp);
      unsigned scratchSizeBytes = helper.getScratchSizeInBytes();
      if (reductionNeedsLaneExchange(helper.getSrcTy(), reduceOp.getAxis())) {
        // Handle lane exchange case
        scratchSizeBytes +=
            getReductionScratchSizeForLaneExchange(helper.getSrcTy());
      }
      return llvm::alignTo(scratchSizeBytes, 64);
    }

    return llvm::alignTo(
        mlir::triton::defaultAllocationAnalysisScratchSizeFn(op), 64);
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
