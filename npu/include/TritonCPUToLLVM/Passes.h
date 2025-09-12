#ifndef TRITONNPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H
#define TRITONNPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace npu {

#define GEN_PASS_DECL
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonCPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>>
createSharedMemoryGlobalConversionPass();

#define GEN_PASS_REGISTRATION
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"

} // namespace npu
} // namespace triton

} // namespace mlir

#endif
