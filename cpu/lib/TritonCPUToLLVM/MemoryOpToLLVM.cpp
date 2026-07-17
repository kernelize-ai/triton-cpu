#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::cpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::cpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::cpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);

    auto type = cast<PointerType>(op.getType());
    Type elemTy = type.getPointeeType();
    unsigned allocBytes = op.getSize() * elemTy.getIntOrFloatBitWidth() / 8;

    auto allocaOp =
        LLVM::AllocaOp::create(rewriter, loc, ptr_ty(context), elemTy,
                               b.i32_val(op.getSize()), /*alignment=*/64);

    auto isZeroSplat = [](SplatElementsAttr splat) -> bool {
      return splat.getSplatValue<APFloat>().isZero() ||
             splat.getSplatValue<APInt>().isZero();
    };

    if (auto src = op.getSrc()) {
      if (auto constantOp =
              dyn_cast_or_null<arith::ConstantOp>(src.getDefiningOp())) {
        assert(isa<RankedTensorType>(constantOp.getType()) &&
               "expected constant op to have tensor type");
        auto splatAttr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
        if (splatAttr && isZeroSplat(splatAttr)) {
          Value zero = b.i8_val(0);
          LLVM::MemsetInlineOp::create(
              rewriter, loc, allocaOp.getResult(), zero,
              rewriter.getI64IntegerAttr(allocBytes),
              /*isVolatile=*/rewriter.getBoolAttr(false));
        } else {
          op.emitError("non-zero src not yet supported");
          return failure();
        }
      } else {
        // TODO: do we need to support non-constant/non-zero src?
        op.emitError("non-constant or non-zero src not yet supported");
        return failure();
      }
    }

    rewriter.replaceOp(op, allocaOp);
    return success();
  }

  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
}
