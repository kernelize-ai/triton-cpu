#include "mlir/Pass/Pass.h"

#include "npu/include/TritonNPUToTenstorrent/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DECL_CONVERTTRITONNPUTOTENSTORRENT
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonNPUToTenstorrentPass() {
  return nullptr;
}

} // namespace triton
} // namespace mlir
