#ifndef TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H
#define TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonCPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>>
createSharedMemoryGlobalConversionPass();

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
