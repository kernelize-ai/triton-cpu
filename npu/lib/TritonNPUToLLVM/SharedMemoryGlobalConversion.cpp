#include "npu/include/TritonNPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "TargetInfo.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {
#define GEN_PASS_DEF_SHAREDMEMORYGLOBALCONVERSIONNPU
#include "npu/include/TritonNPUToLLVM/Passes.h.inc"
} // namespace npu
} // namespace triton
} // namespace mlir

namespace {

struct SharedMemoryGlobalConversionNPU
    : public mlir::triton::npu::impl::SharedMemoryGlobalConversionNPUBase<
          SharedMemoryGlobalConversionNPU> {
  using SharedMemoryGlobalConversionNPUBase::
      SharedMemoryGlobalConversionNPUBase;

  SharedMemoryGlobalConversionNPU() = default;

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
                                              npu::kSharedMemoryOffset);
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

namespace mlir::triton::npu {
std::unique_ptr<OperationPass<ModuleOp>>
createSharedMemoryGlobalConversionPass() {
  return std::make_unique<SharedMemoryGlobalConversionNPU>();
}
} // namespace mlir::triton::npu
