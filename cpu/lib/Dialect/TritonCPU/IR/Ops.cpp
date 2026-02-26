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
  const int64_t blockSize = getBlockSize();
  const int64_t vectorSize = getVectorSize();
  if (blockSize <= 0)
    return emitOpError("expects blockSize > 0, got ") << blockSize;
  if (vectorSize <= 0)
    return emitOpError("expects vectorSize > 0, got ") << vectorSize;
  if (blockSize % vectorSize != 0)
    return emitOpError("expects blockSize % vectorSize == 0, got blockSize=")
           << blockSize << " vectorSize=" << vectorSize;

  // TODO: other verification checks?

  return success();
}

} // namespace cpu
} // namespace triton

} // namespace mlir

#define GET_OP_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Ops.cpp.inc"
