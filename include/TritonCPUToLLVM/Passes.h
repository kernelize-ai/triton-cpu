#ifndef TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H
#define TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"

}
}
}



#endif
