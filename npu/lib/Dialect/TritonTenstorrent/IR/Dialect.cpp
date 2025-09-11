#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::npu;

namespace mlir::triton::npu::tt {

/// Parse an attribute registered to this dialect.
::mlir::Attribute
TritonTenstorrentDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                         ::mlir::Type type) const {
  llvm_unreachable("parse stub called");
}

/// Print an attribute registered to this dialect.
void TritonTenstorrentDialect::printAttribute(
    ::mlir::Attribute attr, ::mlir::DialectAsmPrinter &os) const {
  llvm_unreachable("print stub called");
}

void TritonTenstorrentDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/include/Dialect/TritonTenstorrent/IR/Ops.cpp.inc"
      >();
}

} // namespace mlir::triton::npu::tt
