#ifndef TRITONCPU_DIALECT
#define TRITONCPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonCPU_Dialect : Dialect {
  let name = "ttc";

  let cppNamespace = "::mlir::triton::cpu";

  let description = [{
    Triton CPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
  ];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
