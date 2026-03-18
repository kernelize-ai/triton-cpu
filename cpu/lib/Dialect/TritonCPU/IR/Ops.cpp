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
  auto blockShape = getBlockShape();
  auto vectorShape = getVectorShape();

  if (blockShape.size() != 1) {
    return emitOpError(
               "only rank-1 generic ops are currently supported, got rank ")
           << blockShape.size();
  }

  if (blockShape.size() != vectorShape.size()) {
    return emitOpError(
               "expects blockShape and vectorShape to have the same rank, got ")
           << blockShape.size() << " vs " << vectorShape.size();
  }
  for (unsigned i = 0; i < blockShape.size(); i++) {
    if (blockShape[i] <= 0) {
      return emitOpError("expects blockShape[")
             << i << "] to be positive, got " << blockShape[i];
    }
    if (vectorShape[i] <= 0) {
      return emitOpError("expects vectorShape[")
             << i << "] to be positive, got " << vectorShape[i];
    }
    if (blockShape[i] < vectorShape[i]) {
      return emitOpError("expects blockShape[")
             << i << "] >= vectorShape[" << i << "], got " << blockShape[i]
             << " vs " << vectorShape[i];
    }
    if (blockShape[i] % vectorShape[i] != 0) {
      return emitOpError("expects blockShape[")
             << i << "] % vectorShape[" << i << "] == 0, got " << blockShape[i]
             << " vs " << vectorShape[i];
    }
  }

  // Body must exist and have the implicit induction variable arguments.
  Region &body = getBody();
  if (body.empty())
    return emitOpError("expects a non-empty body region");
  Block &bodyBlock = body.front();
  unsigned numInductionVars = getNumInductionVars();
  if (bodyBlock.getNumArguments() < numInductionVars)
    return emitOpError("body block must have at least ")
           << numInductionVars << " argument(s) for induction variable(s)";
  for (unsigned i = 0; i < numInductionVars; ++i) {
    if (!bodyBlock.getArgument(i).getType().isInteger(32))
      return emitOpError("body induction variable ") << i << " must be i32";
  }

  // Combiners region must have one block per scalar result.
  Region &combiners = getCombiners();
  unsigned numResults = getNumResults();
  if (combiners.getBlocks().size() != numResults) {
    return emitOpError("expects combiners region to have ")
           << numResults << " block(s), got " << combiners.getBlocks().size();
  }
  for (auto [i, block] : llvm::enumerate(combiners.getBlocks())) {
    Type resultTy = getResultTypes()[i];
    if (block.getNumArguments() != 2 ||
        block.getArgument(0).getType() != resultTy ||
        block.getArgument(1).getType() != resultTy) {
      return emitOpError("combiner block ")
             << i << " expects two arguments of type " << resultTy;
    }
  }

  return success();
}

} // namespace cpu
} // namespace triton

} // namespace mlir

#define GET_OP_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Ops.cpp.inc"
