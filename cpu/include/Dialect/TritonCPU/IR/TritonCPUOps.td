#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "TritonCPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTC_BlockIdOp : TTC_Op<"block_id", [Pure]> {
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
    Get the current block ID within the loop over blocks for this persistent kernel.
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
    Get the starting block ID (typically retrieved from a function argument).
  }];

  let assemblyFormat = "attr-dict";
}

def TTC_BlockEndOp : TTC_Op<"block_end", [Pure]> {
  let summary = "Get the ending block ID for this kernel invocation.";

  let arguments = (ins);

  let results = (outs I32:$blockId);

  let description = [{
    Get the ending block ID (typically retrieved from a function argument).
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

#endif
