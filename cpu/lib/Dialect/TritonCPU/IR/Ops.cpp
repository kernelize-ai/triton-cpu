#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"

namespace mlir {

static Type getI1SameShape(Type type) {
  Type i1Type = IntegerType::get(type.getContext(), 1);
  if (LLVM::isCompatibleVectorType(type))
    return LLVM::getVectorType(i1Type, LLVM::getVectorNumElements(type));
  return i1Type;
}

namespace triton {
namespace cpu {

void MaskedLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable(),
                       GlobalMemory::get());
}

LogicalResult GenericOp::verify() {
#if 1
  auto blockShape = getBlockShape();
  auto vectorShape = getVectorShape();

  if (blockShape.size() != vectorShape.size()) {
    return emitOpError(
               "expects blockShape and vectorShape to have the same rank, got ")
           << blockShape.size() << " vs " << vectorShape.size();
  }
  for (unsigned i = 0; i < blockShape.size(); i++) {
    if (blockShape[i] % vectorShape[i] != 0) {
      return emitOpError("expects blockShape[")
             << i << "] % vectorShape[" << i << "] == 0, got " << blockShape[i]
             << " vs " << vectorShape[i];
    }
  }
#else
  RankedTensorType blockType = getBlockType();
  auto encoding = blockType.getEncoding();
  if (!encoding) {
    return emitOpError("expects blockType to have a valid encoding");
  }
  auto blockedEncoding = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding);
  if (!blockedEncoding) {
    return emitOpError("expects blockType to have a BlockedEncodingAttr");
  }

  const auto blockDims = blockType.getShape();
  const auto sizePerThread = blockedEncoding.getSizePerThread();
  if (blockDims.size() != sizePerThread()) {
    return emitOpError("expects blockDims.size() == sizePerThread(), got ")
           << blockDims.size() << " vs " << sizePerThread();
  }

  for (unsigned i = 0; i < blockDims.size(); i++) {
    if (blockDims[i] % sizePerThread[i] != 0) {
      return emitOpError("expects blockDims[")
             << i << "] % sizePerThread[" << i << "] == 0, got " << blockDims[i]
             << " vs " << sizePerThread[i];
    }
  }
#endif
  // TODO: other verification checks?

  return success();
}

} // namespace cpu
} // namespace triton

} // namespace mlir

#define GET_OP_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Ops.cpp.inc"
