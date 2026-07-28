#ifndef TRITONNPU_TENSORRNT_D2M_CONVERSION_PASSES
#define TRITONNPU_TENSORRNT_D2M_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToD2M : Pass<"convert-triton-npu-to-d2m", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to D2M";
    let description = [{
        The `convert-triton-npu-to-d2m` pass transforms Triton NPU operations
        into D2M operations, preparing the IR for D2M kernel generation.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::d2m::D2MDialect",
                             "mlir::tt::ttir::TTIRDialect",
                             "mlir::tt::ttnn::TTNNDialect",
                             "mlir::tt::ttkernel::TTKernelDialect",
                             "mlir::arith::ArithDialect",
                             "func::FuncDialect",
                             "mlir::triton::npu::tt::TritonTenstorrentDialect"
                            ];
}

def ConvertTTCGenericToD2M : Pass<"convert-ttc-generic-to-d2m", "mlir::ModuleOp"> {
    let summary = "Convert TritonCPU Generic Ops to D2M";
    let description = [{
        // TODO
    }];

    let dependentDialects = ["mlir::d2m::D2MDialect"];
}

def TensorizeScalars : Pass<"tensorize-scalars", "mlir::ModuleOp"> {
    let summary = "Convert scalar arguments to a function into 0D tensors";
    let description = [{
        The `tensorize-scalars` pass transforms scalar arguments of functions into 0D tensors. This
        prepares the IR for injection into tt-mlir. As we understand things, scalars to
        `{d2m,ttnn}.generic` are only ever allowed to be ui32; `emitc.literal
        "get_compile_time_arg_val"` only returns a ui32.
    }];
    let dependentDialects = ["func::FuncDialect", "tensor::TensorDialect", "mlir::tt::ttnn::TTNNDialect"];
}

#endif
