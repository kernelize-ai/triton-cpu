#include "TargetInfo.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {

// Adapted from `TritonGPUToLLVM/DotOpToLLVM/FMA.cpp`.
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    Type tgtTy = accum.getType();
    for (auto it = llvm::zip(a, b).begin(); it != llvm::zip(a, b).end(); ++it) {
      const auto &aElem = std::get<0>(*it);
      const auto &bElem = std::get<1>(*it);
      const auto ty = aElem.getType();
      Value mul = LLVM::FMulOp::create(builder, loc, ty, aElem, bElem);
      if (ty != tgtTy) {
        mul = LLVM::FPExtOp::create(builder, loc, tgtTy, mul);
      }
      accum = LLVM::FAddOp::create(builder, loc, tgtTy, accum, mul);
    }
    return accum;
  }
};

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter,
                  const cpu::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    GenericFMAVectorMultiplier multiplier(rewriter, loc);
    return parametricConvertFMADot(op, adaptor, getTypeConverter(), rewriter,
                                   multiplier);
  }

private:
  const cpu::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const cpu::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, targetInfo, benefit);
}
