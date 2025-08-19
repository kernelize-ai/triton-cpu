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

def MathToUKernel : Pass<"convert-npu-math-to-ukernel", "mlir::ModuleOp"> {
    let summary = "Convert NPU math operations to micro-kernels";
    let description = [{
        The `convert-npu-math-to-ukernel` pass transforms math operations
        into calls to NPU-specific uKernel library implementations.
    }];
    let constructor = "mlir::triton::npu::createMathToUKernelPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::triton::TritonDialect"];
}

#endif
