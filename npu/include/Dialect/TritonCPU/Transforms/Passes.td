#ifndef TRITONCPU_PASSES
#define TRITONCPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonCPUCoalesce: Pass<"tritoncpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    The pass analyses loads/stores with type `tensor<tt.ptr<>>` or
    `tt.ptr<tensor<>>` and replaces the layouts of these operations with
    coalesced layouts, i.e. cache friendly access patterns.
    Layout conversions are inserted before and after the load/store op
    to maintain consistency with the rest of the program.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

#endif
