#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritoncpu-lower-dot-microkernel-to-sme"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_LOWERDOTMICROKERNELTOSME
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

struct LowerDotMicrokernelToSMEPass
    : public impl::LowerDotMicrokernelToSMEBase<LowerDotMicrokernelToSMEPass> {
  using LowerDotMicrokernelToSMEBase::LowerDotMicrokernelToSMEBase;

  void runOnOperation() override { assert(false && "TODO"); }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerDotMicrokernelToSMEPass() {
  return std::make_unique<LowerDotMicrokernelToSMEPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
