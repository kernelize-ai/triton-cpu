#ifndef TTRITON_CONVERSION_TRITONNPU_TO_LLVM_TARGETINFONPU_H
#define TTRITON_CONVERSION_TRITONNPU_TO_LLVM_TARGETINFONPU_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::NPU {

class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo() {}

  bool supportMaximumMinimum() const override {
    // TODO
    return true;
  }

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  bool canUseStMatrix(RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> order,
                      int swizzleByteSize) const override;

  bool supportLdMatrix() const override { return false; }
  bool supportStMatrix() const override { return false; }

  void storeMatrixShared(RewriterBase &rewriter, Location loc, Value ptr,
                         Value val) const override;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  int axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override;

  int getAddressSpace(Attribute addressSpace) const override;

  bool supportVectorizedAtomics() const override;
};

} // namespace mlir::triton::NPU

#endif
