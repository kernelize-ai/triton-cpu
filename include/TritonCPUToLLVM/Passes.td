#ifndef TRITONCPU_CONVERSION_PASSES
#define TRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def MemoryOpToLLVM : Pass<"triton-cpu-memory-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory operations to LLVM for CPU.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createMemoryOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

#endif
