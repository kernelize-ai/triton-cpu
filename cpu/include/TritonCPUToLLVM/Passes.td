#ifndef TRITONCPU_CONVERSION_PASSES
#define TRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonCPUToLLVM : Pass<"convert-triton-cpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonCPU to LLVM";
    let description = [{
        The `convert-triton-cpu-to-llvm` pass converts Triton CPU operations
        into LLVM operations.
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

def ConvertMaskedOpsToLLVM : Pass<"convert-masked-ops-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert masked operations to LLVM";

    let description = [{
        The `convert-masked-ops-to-llvm` pass transforms TritonCPU masked operations
        into equivalent LLVM instructions. This is done after ConvertTritonCPUToLLVM
        for better debugging and observability.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::cpu::TritonCPUDialect"
                            ];
}

def AllocateSharedMemoryCPU : Pass<"allocate-shared-memory-cpu", "mlir::ModuleOp"> {
    let summary = "Add metadata for CPU shared memory allocation";

    let description = [{
        See `allocate-shared-memory` for more details.
    }];
}

def SharedMemoryGlobalConversionCPU : Pass<"shared-memory-global-conversion-cpu", "mlir::ModuleOp"> {
    let summary = "Convert CPU shared memory ptrs from global arguments to function args";

    let description = [{
        Triton GPU backends expect shared memory addresses to be available as globals. However,
        in the CPU backend we pass the shared memory address in as a kernel argument. This pass
        converts the global shared memory ptr accesses to retrieve shared memory addresses
        from the function arguments.
    }];
}

#endif
