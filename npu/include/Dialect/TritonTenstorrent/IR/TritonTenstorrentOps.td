#ifndef TRITONTENSTORRENT_OPS
#define TRITONTENSTORRENT_OPS

include "TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"

class TTTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonTenstorrent_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTTS_KernelArgOp : TTTS_Op<"kernel_arg"> {
  let summary = "Placeholder for translation of a runtime kernel arg to compile time constant in the Tenstorrent compute kernel.";

  let arguments = (ins IndexAttr:$arg_index);

  let results = (outs AnyTypeOf<[TT_PtrLike, TT_FloatLike, TT_IntLike]>:$result);

  let description = [{
    Coming soon...
  }];

}

#endif
