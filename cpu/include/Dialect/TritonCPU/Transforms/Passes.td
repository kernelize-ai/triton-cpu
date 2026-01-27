#ifndef TRITONCPU_PASSES
#define TRITONCPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonCPUAccelerateMatmul : Pass<"tritoncpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., Nvidia tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

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

def MakePersistentKernelPass : Pass<"make-persistent-kernel", "mlir::ModuleOp"> {
    let summary = "Rewrite the kernel to process multiple blocks sequentially.";

    let description = [{
        This pass rewrites kernels to process multiple blocks sequentially by creating a
        wrapper function that loops over a range of block IDs. Block ID arguments are marked
        with sentinel ops `ttc.block_start`, `ttc.block_end`, and `ttc.current_block` which
        are later replaced with function arguments during LLVM lowering. Other backends
        may choose to replace these ops with function calls or instructions as appropriate.
        The ops are introduced to differentiate the persistent kernel loop from other loops
        in the kernel.
    }];

    let dependentDialects = [
      "mlir::triton::gpu::TritonGPUDialect",
      "mlir::scf::SCFDialect",
      "mlir::triton::cpu::TritonCPUDialect"
    ];
}

def TestLaneMapAnalysisPass : Pass<"test-lane-map-analysis", "mlir::ModuleOp"> {
  let summary = "Test lane map analysis";

  let description = [{
    This pass tests the lane map analysis on the module.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

#endif
