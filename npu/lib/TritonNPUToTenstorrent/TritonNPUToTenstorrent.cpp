#include "mlir/Pass/Pass.h"

#include "npu/include/TritonNPUToTenstorrent/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONNPUTOTENSTORRENT
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

struct ConvertTritonNPUToTenstorrent
    : public triton::impl::ConvertTritonNPUToTenstorrentBase<
          ConvertTritonNPUToTenstorrent> {
  using ConvertTritonNPUToTenstorrentBase::ConvertTritonNPUToTenstorrentBase;

  ConvertTritonNPUToTenstorrent() : ConvertTritonNPUToTenstorrentBase() {}

  void runOnOperation() override { assert(false); }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonNPUToTenstorrentPass() {
  return std::make_unique<ConvertTritonNPUToTenstorrent>();
}

} // namespace triton
} // namespace mlir
