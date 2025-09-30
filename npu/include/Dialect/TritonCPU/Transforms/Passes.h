#ifndef TRITON_DIALECT_TRITONCPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONCPU_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "npu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "npu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
