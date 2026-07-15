#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritoncpu-lower-dot-microkernel-to-sme"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_LOWERDOTMICROKERNELTOSME
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

struct SmeGemmInfo {
  ttc::GenericOp generic; // anchor: read geometry, delete in phase 3
  tt::DotOp dot;

  Type elemTy;            // f32
  int64_t blockM, blockN; // 64, 64   <- generic.blocks[1..2]
  int64_t blockK;         // 32       <- tileShape[0]; slab depth + pack height
  Value kPad;             //          <- generic.blocks[0]; slab loop bound

  Value aTile, bTile, acc; // dot.getA()/getB()/getC() — one hop, no chain walk

  static std::optional<SmeGemmInfo> tryMatch(cpu::GenericOp) {
    // TODO
    return std::nullopt;
  }
};

struct LowerDotMicrokernelToSMEPass
    : public impl::LowerDotMicrokernelToSMEBase<LowerDotMicrokernelToSMEPass> {
  using LowerDotMicrokernelToSMEBase::LowerDotMicrokernelToSMEBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mod.walk([&](triton::FuncOp funcOp) {
      if (!funcOp.getName().ends_with("matmul_kernel_dot_microkernel"))
        return;

      // TODO: lower this microkernel

      // 1. Match → SmeGemmInfo::tryMatch(generic).

      // 2. Get-or-create the leaf at module scope, keyed on (elemTy, blockM,
      // blockN). blockK is a runtime arg (%KC), so it doesn't key the function
      // — one leaf serves all slab depths.

      // 3. In-body rewrite, at the generic's location: entry-block allocas →
      // zero C → slab loop { pack A, pack B, call } → drain.

      // 4. Delete the generic.

      llvm::errs() << "lower microkernel " << funcOp.getName() << "\n";
    });

    assert(false && "TODO");
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerDotMicrokernelToSMEPass() {
  return std::make_unique<LowerDotMicrokernelToSMEPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
