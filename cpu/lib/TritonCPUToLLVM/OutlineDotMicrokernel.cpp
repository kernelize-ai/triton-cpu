#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritoncpu-outline-dot-microkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_OUTLINEDOTMICROKERNEL
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

struct OutlineDotMicrokernelPass
    : public impl::OutlineDotMicrokernelBase<OutlineDotMicrokernelPass> {
  using OutlineDotMicrokernelBase::OutlineDotMicrokernelBase;

  void runOnOperation() override { assert(false && "got here"); }
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineDotMicrokernelPass() {
  return std::make_unique<OutlineDotMicrokernelPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
