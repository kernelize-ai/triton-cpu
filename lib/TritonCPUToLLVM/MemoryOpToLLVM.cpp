#include "TypeConverter.h"

#include "npu/include/TritonCPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"


namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MEMORYOPTOLLVM
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;


namespace {

    class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct PtrToMemRefOpConversion : public OpConversionPattern<PtrToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PtrToMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = rewriter.getRemappedValue(op.getSrc());
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());

    Value res = b.undef(memRefStructTy);
    res =
        rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, res, ptr, 1);
    rewriter.replaceOp(op, res);

    return success();
  }
};

struct MemoryOpToLLVM
    : public triton::impl::MemoryOpToLLVMBase<MemoryOpToLLVM> {
  using MemoryOpToLLVMBase::MemoryOpToLLVMBase;

  MemoryOpToLLVM() : MemoryOpToLLVMBase() {}

    void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<PtrToMemRefOpConversion>(typeConverter, context);


    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace


namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass() {
  return std::make_unique<MemoryOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
