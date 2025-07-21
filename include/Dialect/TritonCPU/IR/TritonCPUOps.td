#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "TritonCPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // NoMemoryEffect
include "triton/Dialect/Triton/IR/TritonTypes.td"

class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTC_PtrToMemRefOp : TTC_Op<"ptr_to_memref", [NoMemoryEffect]> {
  let summary = "Build a memref for a pointer.";

  let description = [{
    Build memref with static shape, offset, strides, and specified base pointer.
  }];

  let arguments = (ins TT_Ptr:$src);

  let results = (outs AnyStaticShapeMemRef:$result);

  let hasCanonicalizer = 0;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif
