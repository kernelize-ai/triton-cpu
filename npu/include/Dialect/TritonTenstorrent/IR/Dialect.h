#ifndef TRITON_DIALECT_TRITONTENSTORRENT_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONTENSTORRENT_IR_DIALECT_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonTenstorrent depends on Triton types
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GET_OP_CLASSES
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h.inc"
#include "npu/include/Dialect/TritonTenstorrent/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONTENSTORRENT_IR_DIALECT_H_
