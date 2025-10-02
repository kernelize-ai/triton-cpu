#include "triton/Dialect/Triton/IR/Dialect.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

/// Parse an attribute registered to this dialect.
::mlir::Attribute
TritonCPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                 ::mlir::Type type) const {
  llvm_unreachable("parse stub called");
}

/// Print an attribute registered to this dialect.
void TritonCPUDialect::printAttribute(::mlir::Attribute attr,
                                      ::mlir::DialectAsmPrinter &os) const {
  llvm_unreachable("print stub called");
}

void TritonCPUDialect::initialize() {
  //   registerTypes();

  addOperations<
#define GET_OP_LIST
#include "cpu/include/Dialect/TritonCPU/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface>();
}
