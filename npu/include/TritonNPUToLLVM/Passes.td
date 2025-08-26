#ifndef TRITONNPU_CONVERSION_PASSES
#define TRITONNPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToLLVM : Pass<"convert-triton-npu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to LLVM";
    let description = [{
        The `convert-triton-npu-to-llvm` pass converts Triton NPU operations
        into LLVM operations, enabling further optimizations and code generation.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"
                             ];
}

def AllocateSharedMemoryNPU : Pass<"allocate-shared-memory-npu", "mlir::ModuleOp"> {
    let summary = "Add metadata for NPU shared memory allocation";

    let description = [{
        coming soon
    }];
}

#endif
