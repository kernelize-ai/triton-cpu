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

namespace {

// Adapted from `TritonGPUToLLVM/DotOpToLLVM/FMA.cpp`.
class GenericFMAVectorMultiplier : public triton::gpu::FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    assert(a.size() == b.size() &&
           "operands to `tt.dot` have mismatched sizes");
    Value accum = c;
    Type tgtTy = accum.getType();
    for (auto it = llvm::zip(a, b).begin(); it != llvm::zip(a, b).end(); ++it) {
      Value aElem = std::get<0>(*it);
      Value bElem = std::get<1>(*it);

      // Extend to the target type if needed.
      const auto ty = aElem.getType();
      assert(ty == bElem.getType() &&
             "operands to `tt.dot` have mismatched types");
      if (ty != tgtTy) {
        assert(
            ty.isFloat() && tgtTy.isFloat() &&
            "only float point type casting is currently supported in `tt.dot`");
        assert(
            ty.getIntOrFloatBitWidth() < tgtTy.getIntOrFloatBitWidth() &&
            "operands to `tt.dot` must be smaller than the accumulator type");
        aElem = LLVM::FPExtOp::create(builder, loc, tgtTy, aElem);
        bElem = LLVM::FPExtOp::create(builder, loc, tgtTy, bElem);
      }

      // Multiply and accumulate.
      auto flags = LLVM::FastmathFlagsAttr::get(builder.getContext(),
                                                LLVM::FastmathFlags::fast);
      accum =
          LLVM::FMAOp::create(builder, loc, tgtTy, aElem, bElem, accum, flags);
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
