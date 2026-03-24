#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "TritonCPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTC_BlockIdOp : TTC_Op<"block_id", []> {
  let summary = "Get the block ID.";

  let description = [{
    Get the block ID along the specified axis.
  }];

  let arguments = (ins TT_ProgramDim:$axis);

  let results = (outs I32:$blockId);

  let builders = [
    OpBuilder<(ins "int":$axis), [{
      build($_builder, $_state, $_builder.getI32Type(), ProgramIDDimAttr::get($_builder.getContext(), ProgramIDDim(axis)));
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t getAxisAsInt() {
      return static_cast<int32_t>(getAxis());
    }
  }];
}

def TTC_CurrentBlockOp : TTC_Op<"current_block", [SameOperandsAndResultType]> {
  let summary = "The current block ID for this kernel invocation.";

  let description = [{
    Get the current block ID. We expect it to be used predominantly by
    persistent kernels (see `MakePersistentKernel`). In that context, this
    operation serves only as a marker for the SSA value it points to: the sole
    input is expected to be the induction variable for the persistent loop.
  }];

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
}

def TTC_BlockStartOp : TTC_Op<"block_start", [Pure]> {
  let summary = "Get the starting block ID for this kernel invocation.";

  let arguments = (ins);

  let results = (outs I32:$blockId);

  let description = [{
    Get the starting block ID. We expect it to be used predominantly by
    persistent kernels (see `MakePersistentKernel`): that pass synthesizes a
    loop over a range of blocks. Then, we use runtime information to populate
    the starting and ending block IDs for each kernel invocation, which allows
    the kernel to process multiple blocks sequentially. The `block_start` and
    `block_end` ops are used to query this information within the kernel.
  }];

  let assemblyFormat = "attr-dict";
}

def TTC_BlockEndOp : TTC_Op<"block_end", [Pure]> {
  let summary = "Get the ending block ID for this kernel invocation.";

  let arguments = (ins);

  let results = (outs I32:$blockId);

  let description = [{
    Get the ending block ID. See `BlockStartOp` for more details.
  }];

  let assemblyFormat = "attr-dict";
}

def TTC_MaskedLoadOp : TTC_Op<"masked_load", [
  TypesMatchWith<"mask size matches false val size", "falseVal", "mask", "mlir::getI1SameShape($_self)">,
  TypesMatchWith<"result type matches false val type", "result", "falseVal", "$_self">,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Masked load operation";

  let description = [{
    Load operation which lowers to a predicated load, scalar load, or masked vector load intrinsic.
  }];

  let arguments = (
    ins
    LLVM_AnyPointer:$ptr,
    AnyTypeOf<[LLVM_VectorOf<I1>, I1]>:$mask,
    AnyTypeOf<[LLVM_VectorOf<LLVM_Type>, LLVM_Type]>:$falseVal,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs LLVM_Type:$result);

  let builders = [
    OpBuilder<(ins "Type":$retType, "Value":$ptr, "Value":$mask, "Value":$falseVal), [{
      build($_builder, $_state, retType, ptr, mask, falseVal, IntegerAttr());
    }]>
  ];

  let assemblyFormat = [{
    $ptr `,` $mask `,` $falseVal
    attr-dict `:` functional-type(operands, results)
  }];
}

def TTC_MaskedStoreOp : TTC_Op<"masked_store", [
  TypesMatchWith<"mask type matches value type", "value", "mask",
                 "mlir::getI1SameShape($_self)">
]> {
  let summary = "Masked store operation";

  let description = [{
    Store operation which conditionally lowers to the llvm.masked.store intrinsic, a predicated store, or a LLVM store if the mask is provably always true.
  }];

  let arguments = (
    ins LLVM_AnyPointer:$ptr,
    LLVM_Type:$value,
    AnyTypeOf<[LLVM_VectorOf<I1>, I1]>:$mask,
    OptionalAttr<I32Attr>:$alignment
    );

  let results = (outs);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$value, "Value":$mask), [{
      build($_builder, $_state, {}, ptr, value, mask, IntegerAttr());
    }]>
  ];

  let assemblyFormat = [{
    $ptr `,` $value `,` $mask
    attr-dict `:` functional-type(operands, results)
  }];
}

def TTC_YieldOp : TTC_Op<"yield", [Terminator, ReturnLike, HasParent<"GenericOp">]> {
  let summary = "Terminator for ttc.generic";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

def TTC_GenericOp : TTC_Op<"generic", [AttrSizedOperandSegments]> {
  let summary = "Tiled iteration over a tensor block with fused loads and stores";

  let description = [{
    Wraps a fused chain of tensor operations (loads, elementwise math, stores)
    and owns the schedule for iterating over the block in tiles.

    The operation logically iterates over `blockShape` in steps of `vectorShape`,
    executing the body region once per tile. This models the pattern of splitting
    a full-block tensor computation into a loop over sub-vectors, enabling
    register-pressure-friendly lowering without materializing the entire block
    at once.

    Operands:
    - `ins`: tensor-typed inputs (e.g. pointer tensors for loads and stores).
      Each operand is sliced to `vectorShape` for each tile iteration and passed
      as the corresponding block argument to the body region.
    - `params`: scalar or tensor-typed pass-through values that are forwarded
      into the body region unchanged (not tiled).

    The `body` region receives block arguments corresponding to (in order) the
    tiled slices of `ins` followed by the untiled `params`. It is executed once
    per tile and terminated by `ttc.yield`. When the op has scalar results,
    `ttc.yield` returns one partial value per result (e.g. the output of a
    `tt.reduce` over the tile).

    The `combiners` region contains one block per scalar result. Each block takes
    two arguments `(acc, partial)` of the result type and yields the updated
    accumulator. The lowering bootstraps the accumulator from the first tile's
    partial result and applies the combiner for each subsequent tile.

    Constraints:
    - `blockShape` and `vectorShape` must have the same rank.
    - Each `blockShape[i]` must be a positive multiple of `vectorShape[i]`.
    - `combiners` must have exactly as many blocks as there are scalar results,
      each with two arguments and one yield of the matching type.

    Example — elementwise chain (no scalar results, empty combiners):
    ```
    ttc.generic %src_ptrs, %dst_ptrs {blockShape = [128], vectorShape = [16]}
        : tensor<128x!tt.ptr<f32>>, tensor<128x!tt.ptr<f32>>
    body {
    ^bb0(%src: tensor<16x!tt.ptr<f32>>, %dst: tensor<16x!tt.ptr<f32>>):
      %v = tt.load %src : tensor<16xf32>
      %w = math.exp %v  : tensor<16xf32>
      tt.store %dst, %w : tensor<16xf32>
      ttc.yield
    } combiners {}
    ```

    Example — reduction (one scalar result):
    ```
    %max = ttc.generic %ptrs, %mask {blockShape = [128], vectorShape = [16]}
        : tensor<128x!tt.ptr<f32>>, tensor<128xi1> -> f32
    body {
    ^bb0(%tptrs: tensor<16x!tt.ptr<f32>>, %tmask: tensor<16xi1>):
      %tile    = tt.load %tptrs, %tmask : tensor<16xf32>
      %partial = tt.reduce %tile {max}  : tensor<16xf32> -> f32
      ttc.yield %partial : f32
    } combiners {
    ^bb0(%acc: f32, %partial: f32):
      %new = arith.maxnumf %acc, %partial : f32
      ttc.yield %new : f32
    }
    ```

    During lowering to LLVM, the body is unrolled `blockShape / vectorShape`
    times. For each iteration, sub-vectors are extracted from the LLVM struct
    representation of each `ins` operand, cast back to the tile tensor type, and
    substituted for the corresponding block arguments before cloning the body.
    Scalar results are accumulated across tiles using the combiner blocks.
  }];

  let arguments = (ins
    Variadic<TT_Type>:$ins,
    Variadic<TT_Type>:$params,

    DenseI32ArrayAttr:$blockShape,
    DenseI32ArrayAttr:$vectorShape
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$body, AnyRegion:$combiners);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ins (` ` `(` $params^ `)`)? attr-dict-with-keyword `body` $body `combiners` $combiners `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    Value getChunkOffset() { return getBody().front().getArgument(0); }
    unsigned getNumInductionVars() { return 1; }
    unsigned getNumRegionIterArgs() {
      return getBody().front().getNumArguments() - getNumInductionVars();
    }
  }];
}

def TTC_MakeDynamicRangeOp : TTC_Op<"make_dynamic_range", [Pure]> {
    let summary = "make range starting at start";

    let description = [{
        Returns a 1ID int32 tensor starting at `$start` and ending at `$start` + `$result`.getShape()[0].

        Step = 1
    }];

    let arguments = (ins I32:$start);
    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "$start attr-dict `:` type($result)";
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        int32_t getEnd() {
            auto resultTy = cast<RankedTensorType>(getResult().getType());
            return resultTy.getShape()[0];
        }
    }];
}

#endif
