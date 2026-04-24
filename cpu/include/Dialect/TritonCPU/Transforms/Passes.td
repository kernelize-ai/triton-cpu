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

def TritonCPUTileAndFuse : Pass<"tritoncpu-tile-and-fuse", "mlir::ModuleOp"> {
  let summary = "Wrap elementwise op chains in ttc.generic for tiled lowering";

  let description = [{
    Identifies closed chains of tensor operations ending in a `tt.store` and
    wraps them in `ttc.generic` regions to enable tiled, iterative lowering.

    A chain is a set of ops reachable by walking the def-use graph back from a
    `tt.store` through `tt.load` and elementwise `arith` ops. The chain must be
    "closed": every result produced within the chain must be consumed within it
    (i.e. no values escape to users outside the chain).

    When a valid chain is found, a `ttc.generic` op is created whose:
    - `ins` operands are the pointer, mask, and other arguments of any `tt.load`
      and `tt.store` ops in the chain.
    - `blockShape` is taken from the tensor shape of the common tensor type.
    - `tileShape` is taken from the `sizePerThread` field of the
      `BlockedEncodingAttr` on the common tensor type.

    The ops in the chain are cloned into the generic region with tensor types
    updated to the tile shape, and load/store pointer arguments replaced with
    tiled block arguments. The original ops are erased.

    The resulting `ttc.generic` is later lowered to LLVM by unrolling the region
    body once per tile (blockShape / tileShape iterations), extracting and
    inserting sub-vectors from the operands for each tile.
  }];

  let dependentDialects = [
    "mlir::triton::cpu::TritonCPUDialect"
  ];
}

#endif
