#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToD2M/Passes.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // ttc.GenericOp

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "TTCGenericPlan.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTTCGENERICTOD2M
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "convert-ttc-generic-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

struct ConvertTTCGenericToD2MPass
    : public impl::ConvertTTCGenericToD2MBase<ConvertTTCGenericToD2MPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    SmallVector<int64_t> gridShape = llvm::to_vector(gridAttr.getShape());

    mod.walk([&](triton::FuncOp func) {
      // TODO: if there are already generic plans from a previous function, bail
      // - we can't cross function boundaries yet
      func.walk([&](cpu::GenericOp generic) {
        auto planResult = buildPlan(generic);
        if (failed(planResult)) {
          signalPassFailure();
          return;
        }

        llvm::errs() << "generic = " << generic.getHeader() << "\n";
      });
    });

    llvm::errs() << "TODO\n";
    signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
