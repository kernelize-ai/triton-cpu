#ifndef TRITONNPU_TENSTORRENT_CONVERSION_PASSES
#define TRITONNPU_TENSTORRENT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToTenstorrent : Pass<"convert-triton-npu-to-tenstorrent", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to Tenstorrent";
    let description = [{
        The `convert-triton-npu-to-tenstorrent` pass converts Triton NPU operations
        into Tenstorrent operations, enabling further optimizations and code generation.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "mlir::func::FuncDialect",
                             "mlir::tt::ttkernel::TTKernelDialect"
                            ];
}

#endif
