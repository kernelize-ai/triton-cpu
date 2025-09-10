#ifndef TRITONNPU_CONVERSION_TRITONNPUTOTENSTORRENT_PASSES_H
#define TRITONNPU_CONVERSION_TRITONNPUTOTENSTORRENT_PASSES_H

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace npu {

#define GEN_PASS_DECL
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonNPUToTenstorrentPass();

#define GEN_PASS_REGISTRATION
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

} // namespace npu
} // namespace triton

} // namespace mlir

#endif
