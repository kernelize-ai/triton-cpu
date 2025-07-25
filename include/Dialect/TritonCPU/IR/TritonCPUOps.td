#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "TritonCPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"

class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTC_BlockIdOp : TTC_Op<"block_id", [Pure]> {
  let summary = "Get the block ID.";

  let arguments = (ins TT_ProgramDim:$axis);

  let results = (outs Index:$blockId);

  let description = [{
    Get the ID of the current block.
  }];

  let builders = [
    OpBuilder<(ins "int":$axis), [{
      build($_builder, $_state, $_builder.getIndexType(), ProgramIDDimAttr::get($_builder.getContext(), ProgramIDDim(axis)));
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t getAxisAsInt() {
      return static_cast<int32_t>(getAxis());
    }
  }];
}

#endif
