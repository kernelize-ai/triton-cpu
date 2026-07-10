#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"

#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir::triton::gpu;

namespace {

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {

    return {LLVM::FDivOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }
};

static Value checkIsNan(TritonLLVMOpBuilder &builder, Value v) {
  Location loc = builder.loc;
  OpBuilder &rewriter = *builder.builder;

  // bits 0 and 1 indicate signaling Nan and quiet Nan, respectively
  IntegerAttr controlBits = rewriter.getIntegerAttr(i32_ty, 0b11);
  return LLVM::IsFPClass::create(rewriter, loc, i1_ty, v, controlBits);
}

static Value convertFp32ToBf16(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v, const RoundingMode rounding) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int32 = b.bitcast(v, i32_ty);
  if (rounding == RoundingMode::RTZ) {
    auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
    auto truncated = b.trunc(i16_ty, shifted);
    return b.bitcast(truncated, bf16_ty);
  }

  // This implementation is a faster version for fp32 to bf16 type conversion
  // It is from CK:
  // https://github.com/cgmillette/composable_kernel/commit/24e75bef6aa5
  // It uses less VGPR and less number of instructions compared to the
  // previous implementation
  Value isNan = checkIsNan(b, v);
  Value v16 = b.i32_val(16);
  Value tmp = b.and_(i32_ty, b.lshr(i32_ty, as_int32, v16), b.i32_val(1));

  Value v7FFF = b.i32_val(0x7FFF);
  Value s1 = b.add(as_int32, tmp);
  Value s2 = b.add(s1, v7FFF);

  Value vNan = b.i32_val(0x7FFF0000);
  Value res = b.select(isNan, vNan, s2);

  Value shifted = b.lshr(i32_ty, res, v16);
  Value truncated = b.trunc(i16_ty, shifted);
  return b.bitcast(truncated, bf16_ty);
}

Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty);
  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return b.bitcast(shifted, f32_ty);
}

static SmallVector<Value> S8_to_Bf16(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> inValues = {v[0], v[1], v[2], v[3]};
  SmallVector<Value> outValues = {};
  for (Value inVal : inValues) {
    Value bf16Val = LLVM::SIToFPOp::create(rewriter, loc, bf16_ty, inVal);
    outValues.push_back(bf16Val);
  }
  return outValues;
}

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementTypeOrSelf(op.getIn());
    Type outElemTy = getElementTypeOrSelf(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = S8_to_Bf16(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value =
          LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
      return {convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    } else {
      return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementTypeOrSelf(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, value)};
    } else {
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

} // namespace

void mlir::triton::cpu::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const cpu::TargetInfo &targetInfo,
    PatternBenefit benefit) {

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

#define POPULATE_OP(SRC_OP, DST_OP)                                            \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit)

  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);

  POPULATE_OP(arith::ExtFOp, LLVM::FPExtOp);
  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);

  POPULATE_OP(triton::PreciseDivFOp, LLVM::FDivOp);
  POPULATE_OP(triton::PreciseSqrtOp, LLVM::SqrtOp);

#undef POPULATE_OP
}
