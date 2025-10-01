#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "TargetInfo.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_SHAREDMEMORYGLOBALCONVERSIONCPU
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

struct SharedMemoryGlobalConversionCPU
    : public mlir::triton::cpu::impl::SharedMemoryGlobalConversionCPUBase<
          SharedMemoryGlobalConversionCPU> {
  using SharedMemoryGlobalConversionCPUBase::
      SharedMemoryGlobalConversionCPUBase;

  SharedMemoryGlobalConversionCPU() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::SymbolTable symTable(mod);

    auto g = symTable.lookup<mlir::LLVM::GlobalOp>("global_smem");
    if (!g)
      return signalPassFailure();

    mlir::SymbolTableCollection symbolTables;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.empty())
        continue;

      func.walk([&](mlir::LLVM::AddressOfOp addressOf) {
        if (addressOf.getGlobal(symbolTables) == g) {
          // TODO: we can hoist this when intermediate functions support shared
          // memory args
          assert(triton::isKernel(func) &&
                 "shared memory for non-kernel functions not yet supported on "
                 "CPU");
          auto smemFuncArg = func.getArgument(func.getNumArguments() +
                                              cpu::kSharedMemoryOffset);
          assert(isa<LLVM::LLVMPointerType>(smemFuncArg.getType()) &&
                 "expecting shared memory argument to be a pointer");

          addressOf.replaceAllUsesWith(smemFuncArg);
          addressOf.erase();
        }
      });
    }

    // delete the global
    g.erase();
  }
};

} // namespace

namespace mlir::triton::cpu {
std::unique_ptr<OperationPass<ModuleOp>>
createSharedMemoryGlobalConversionPass() {
  return std::make_unique<SharedMemoryGlobalConversionCPU>();
}
} // namespace mlir::triton::cpu
