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
  auto tileShape = getTileShape();

  if (blockShape.empty())
    return emitOpError("must provide a non-empty block/tile shape");

  if (blockShape.size() != tileShape.size()) {
    return emitOpError(
               "expects blockShape and tileShape to have the same rank, got ")
           << blockShape.size() << " vs " << tileShape.size();
  }
  for (unsigned i = 0; i < tileShape.size(); i++) {
    if (tileShape[i] <= 0) {
      return emitOpError("expects tileShape[")
             << i << "] to be positive, got " << tileShape[i];
    }

    // blockShape[i] is a runtime value; check constraints only when it folds
    // to a constant (e.g. when constructed from arith.constant).
    APInt blockVal;
    if (matchPattern(blockShape[i], m_ConstantInt(&blockVal))) {
      int64_t bv = blockVal.getSExtValue();
      if (bv <= 0)
        return emitOpError("expects blockShape[")
               << i << "] to be positive, got " << bv;
      if (bv < tileShape[i])
        return emitOpError("expects blockShape[")
               << i << "] >= tileShape[" << i << "], got " << bv << " vs "
               << tileShape[i];
      if (bv % tileShape[i] != 0)
        return emitOpError("expects blockShape[")
               << i << "] % tileShape[" << i << "] == 0, got " << bv << " vs "
               << tileShape[i];
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

  auto reductionDims = getReductionDims();
  auto initVals = getInitVals();

  // reductionDims entries must be valid indices into blockShape.
  for (auto [i, dim] : llvm::enumerate(reductionDims)) {
    if (dim < 0 || (unsigned)dim >= blockShape.size())
      return emitOpError("reductionDims[")
             << i << "] = " << dim << " is out of range for blockShape of size "
             << blockShape.size();
  }

  // reductionDims must not contain duplicates.
  SmallVector<int32_t> sortedDims(reductionDims.begin(), reductionDims.end());
  llvm::sort(sortedDims);
  if (llvm::adjacent_find(sortedDims) != sortedDims.end())
    return emitOpError("reductionDims must not contain duplicate entries");

  // init_vals count must match reductionDims count.
  if (initVals.size() != reductionDims.size())
    return emitOpError("init_vals has ")
           << initVals.size() << " value(s) but reductionDims has "
           << reductionDims.size() << " entry(ies)";

  // Body block must have numIVs + numIterArgs + numIns arguments.
  unsigned numIns = getIns().size();
  unsigned numIterArgs = reductionDims.size();
  unsigned expectedArgs = numInductionVars + numIterArgs + numIns;
  if (bodyBlock.getNumArguments() != expectedArgs)
    return emitOpError("body block has ")
           << bodyBlock.getNumArguments() << " argument(s) but expects "
           << expectedArgs << " (numIVs=" << numInductionVars
           << " + numIns=" << numIns << " + numIterArgs=" << numIterArgs << ")";

  // ttc.yield leading values must match iter arg types.
  // The body may have multiple blocks after SCF-to-CF lowering (e.g. a K-loop
  // inside the body). Scan all blocks to find the unique ttc.yield.
  cpu::YieldOp yieldOp;
  for (Block &block : body) {
    auto y = dyn_cast<cpu::YieldOp>(block.getTerminator());
    if (!y)
      continue;
    if (yieldOp)
      return emitOpError("body region must have exactly one ttc.yield");
    yieldOp = y;
  }
  if (!yieldOp)
    return emitOpError("body region must contain a ttc.yield terminator");
  if (yieldOp.getValues().size() < numIterArgs)
    return emitOpError("ttc.yield must return at least ")
           << numIterArgs << " value(s) for iter args, got "
           << yieldOp.getValues().size();
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Type yieldTy = yieldOp.getValues()[i].getType();
    Type iterArgTy = bodyBlock.getArgument(numInductionVars + i).getType();
    if (yieldTy != iterArgTy)
      return emitOpError("ttc.yield value ")
             << i << " has type " << yieldTy << " but iter arg " << i
             << " expects type " << iterArgTy;
  }

  return success();
}

std::string GenericOp::getHeader() {
  std::string s;
  llvm::raw_string_ostream os(s);
  print(os, OpPrintingFlags().skipRegions());
  return s;
}

LogicalResult MakeDynamicRangeOp::verify() {
  auto resultTensorTy = cast<RankedTensorType>(getResult().getType());
  if (resultTensorTy.getShape().size() != 1)
    return emitOpError("expects rank-1 result tensor type, got ")
           << resultTensorTy;
  return success();
}

} // namespace cpu
} // namespace triton

} // namespace mlir

#define GET_OP_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Ops.cpp.inc"
