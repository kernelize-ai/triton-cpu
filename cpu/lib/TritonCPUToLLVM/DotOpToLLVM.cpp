#include "TargetInfo.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter,
                  const cpu::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    CDBG("Converting `tt.dot`: " << *op);

    // Extract the dimensions of matrix A from `tensor<MxKxT>`
    Type aOriginalType = op.getA().getType();
    CDBG("Original input A type: " << aOriginalType);
    assert(isa<RankedTensorType>(aOriginalType) &&
           "Expected `tt.dot` input A to be a ranked tensor type");
    auto aOriginalRankedType = dyn_cast<RankedTensorType>(aOriginalType);
    assert(aOriginalRankedType.getRank() == 2 &&
           "Expected a matrix (rank 2 tensor) for `tt.dot` input A");
    auto M = aOriginalRankedType.getShape()[0];
    auto K = aOriginalRankedType.getShape()[1];

    // Extract the dimensions of matrix B from `tensor<KxNxT>`
    Type bOriginalType = op.getB().getType();
    CDBG("Original input B type: " << bOriginalType);
    assert(isa<RankedTensorType>(bOriginalType) &&
           "Expected `tt.dot` input B to be a ranked tensor type");
    auto bOriginalRankedType = dyn_cast<RankedTensorType>(bOriginalType);
    assert(aOriginalRankedType.getElementType() ==
               bOriginalRankedType.getElementType() &&
           "Expected `tt.dot` inputs A and B to have the same element type");
    assert(bOriginalRankedType.getRank() == 2 &&
           "Expected a matrix (rank 2 tensor) for `tt.dot` input B");
    assert(bOriginalRankedType.getShape()[0] == K &&
           "Inner dimension of `tt.dot` inputs A and B must match");
    auto N = bOriginalRankedType.getShape()[1];

    // Collect the actual values to operate on. We expect they have been
    // converted into structs by the `ConvertLayoutOpConversion` pass
    // (`ttg.convert_layout`).
    auto A = adaptor.getA();
    CDBG("Adapted input A: " << A);
    assert(isa<LLVM::LLVMStructType>(A.getType()) &&
           "Expected `tt.dot` input A to be converted to an LLVM struct type");
    auto B = adaptor.getB();
    CDBG("Adapted input B: " << B);
    assert(isa<LLVM::LLVMStructType>(B.getType()) &&
           "Expected `tt.dot` input B to be converted to an LLVM struct type");
    auto C = adaptor.getC();
    CDBG("Adapted input C: " << C);
    assert(isa<LLVM::LLVMStructType>(C.getType()) &&
           "Expected `tt.dot` input C to be converted to an LLVM struct type");
    auto ty = cast<LLVM::LLVMStructType>(A.getType()).getBody()[0];
    assert(ty.isF16() &&
           "Only f16 `tt.dot` is supported so far"); // TODO: assert same as B.
    auto ty2 = rewriter.getF32Type(); // TODO: calculate from `ty`.

    // Iterate over each element to compute the dot product. This is quite naive
    // (TODO) but a good initial implementation to check correctness.
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          auto a_ = b.extract_val(ty, A, k + m * K);
          auto b_ = b.extract_val(ty, B, n + k * N);
          auto c_ = b.extract_val(ty2, C, n + m * N);
          auto mul = b.fmul(ty, a_, b_);
          auto mul_ = b.fpext(ty2, mul);
          auto fma = b.fadd(ty2, c_, mul_);
          C = b.insert_val(C.getType(), C, fma, n + m * N);
        }
      }
    }

    rewriter.replaceOp(op, {C});
    return success();
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
