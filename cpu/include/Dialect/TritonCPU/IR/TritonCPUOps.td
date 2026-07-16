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

def TTC_GenericOp : TTC_Op<"generic", [AttrSizedOperandSegments, RecursiveMemoryEffects, RecursivelySpeculatable, IsolatedFromAbove]> {
  let summary = "Tiled iteration over a tensor block with fused tensor operations";

  let description = [{
    Tiled iteration over a block of tensor operations with optional loop-carried
    iter args for reductions.

    The operation logically iterates over `blockShape` in steps of `tileShape`,
    executing the `body` region once per tile. This splits a full-block tensor
    computation into a loop over sub-tiles, enabling register-pressure-friendly
    lowering without materializing the entire block at once.

    Operands:
    - `init_vals`: initial accumulator values, one per entry in `reductionDims`.
      Each value seeds the corresponding iter arg before the first tile.
    - `ins`: inputs visible to each tile iteration (pointer tensors, scalars,
      etc.). Each tensor operand is sliced to `tileShape` for each tile.

    Attributes:
    - `blockShape`: the shape of the full block being iterated (runtime values).
    - `tileShape`: the shape of each tile (compile-time constants).
    - `reductionDims`: which blockShape dimensions are reduction dimensions.
      One iter arg is associated with each reduction dimension.

    Body region block argument ordering: `[ivs..., iter_args..., ins_args...]`
    - `ivs`: one i32 per tile dimension, giving the per-dimension tile offset.
    - `iter_args`: one value per entry in `reductionDims`, carrying the running
      accumulator. The i-th iter arg starts at `init_vals[i]` and is updated
      each tile via the leading values of `ttc.yield`.
    - `ins_args`: one tiled slice per `ins` operand.

    `ttc.yield` convention:
    - The first `|reductionDims|` values update the iter args (carried to the
      next tile iteration).
    - Any remaining values are tensor tiles that are scattered into the op's
      tensor results.

    Results:
    - The first `|reductionDims|` results hold the final iter arg values after
      all tiles have been processed.
    - Any additional results are full-block tensors assembled from the
      per-tile scatter values.

    Constraints:
    - `blockShape` and `tileShape` must have the same rank.
    - Each `blockShape[i]` must be a positive multiple of `tileShape[i]`.
    - `|init_vals| == |reductionDims|`.
    - Body block must have `|ivs| + |reductionDims| + |ins|` arguments.

    Example — elementwise chain (no reductions):
    ```
    ttc.generic ins (%src_ptrs, %dst_ptrs) blocks [%c128 : i32]
        {tileShape = array<i32: 16>, reductionDims = array<i32>}
        body {
    ^bb0(%iv: i32, %src: tensor<16x!tt.ptr<f32>>, %dst: tensor<16x!tt.ptr<f32>>):
      %v = tt.load %src : tensor<16xf32>
      %w = math.exp %v  : tensor<16xf32>
      tt.store %dst, %w : tensor<16xf32>
      ttc.yield
    } : (tensor<128x!tt.ptr<f32>>, tensor<128x!tt.ptr<f32>>) -> ()
    ```

    Example — scalar reduction (softmax max pass):
    ```
    %max = ttc.generic ins (%ptrs, %n_cols) init (%neg_inf : f32)
        blocks [%c128 : i32]
        {tileShape = array<i32: 16>, reductionDims = array<i32: 0>}
        body {
    ^bb0(%iv: i32, %acc: f32, %tptrs: tensor<16x!tt.ptr<f32>>, %n: i32):
      %tile    = tt.load %tptrs : tensor<16xf32>
      %partial = tt.reduce(%tile) {max} : tensor<16xf32> -> f32
      %new_acc = arith.maxnumf %partial, %acc : f32
      ttc.yield %new_acc : f32
    } : (!tt.ptr<f32>, i32) -> f32
    ```

    Example — mixed scalar + tensor result (softmax sum + exp):
    ```
    %sum, %exp_out = ttc.generic ins (%ptrs, %n_cols) init (%zero : f32)
        blocks [%c128 : i32]
        {tileShape = array<i32: 16>, reductionDims = array<i32: 0>}
        body {
    ^bb0(%iv: i32, %acc: f32, %tptrs: tensor<16x!tt.ptr<f32>>, %n: i32):
      %tile    = tt.load %tptrs : tensor<16xf32>
      %exp     = math.exp %tile : tensor<16xf32>
      %partial = tt.reduce(%exp) {add} : tensor<16xf32> -> f32
      %new_acc = arith.addf %partial, %acc : f32
      ttc.yield %new_acc, %exp : f32, tensor<16xf32>
    } : (!tt.ptr<f32>, i32) -> (f32, tensor<128xf32>)
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$init_vals,
    Variadic<TT_Type>:$ins,

    Variadic<I32>:$blockShape,
    DenseI32ArrayAttr:$tileShape,
    DenseI32ArrayAttr:$reductionDims
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$body);

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "ValueRange":$ins, "ValueRange":$blockShape, "ArrayRef<int32_t>":$tileShape), [{
      build($_builder, $_state, results, ValueRange(), ins, blockShape, tileShape, SmallVector<int32_t>{});
    }]>
  ];

  let assemblyFormat = [{
    (`init` `(` $init_vals^ `:` type($init_vals) `)`)? (`ins` `(` $ins^ `:` type($ins) `)`)? `blocks` `[` $blockShape `:` type($blockShape) `]` attr-dict-with-keyword `body` $body (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    Value getTileOffset(unsigned dim) { return getBody().front().getArgument(dim); }
    unsigned getNumInductionVars() { return getBlockShape().size(); }
    unsigned getNumIterArgs() { return getReductionDims().size(); }
    unsigned getNumRegionIterArgs() {
      return getBody().front().getNumArguments() - getNumInductionVars();
    }
    BlockArgument getIterArg(unsigned i) {
      return getBody().front().getArgument(getNumInductionVars() + i);
    }
    // Index of the first ins block argument: ivs + iter_args precede ins.
    unsigned getInsArgOffset() {
      return getNumInductionVars() + getNumIterArgs();
    }

    std::string getHeader();
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

def TTC_LocalAllocOp : TTC_Op<"local_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor on the stack";

  let description = [{
    Allocates a tensor on the stack. Serves as a Triton-native standin for LLVM::AllocaOp.
  }];

  let arguments = (
    ins
    I32Attr:$size,
    Optional<TT_Tensor>:$src,
    OptionalAttr<I32Attr>:$alignment
  );
  let results = (outs TT_Ptr:$result);

  let assemblyFormat = [{
    ($src^)? attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "PointerType":$type, "uint32_t":$size),
      [{ build($_builder, $_state, type, size, Value(), IntegerAttr()); }]>
  ];

  // TODO: verifier, folder, etc
}

#endif
