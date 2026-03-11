#include "TargetInfo.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct GenericOpConversion : public ConvertOpToLLVMPattern<cpu::GenericOp> {
  using ConvertOpToLLVMPattern<cpu::GenericOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cpu::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    llvm::errs() << "rewrite generic op " << op << "\n";

    auto blockShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("blockShape");
    auto vectorShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("vectorShape");

    ArrayRef<int32_t> blockShape = blockShapeAttr.asArrayRef();
    ArrayRef<int32_t> vectorShape = vectorShapeAttr.asArrayRef();
    assert(blockShape.size() == vectorShape.size() && !blockShape.empty() &&
           "blockShape and vectorShape must be non-empty and of the same size");

    // TODO: assuming 1D shapes
    assert(blockShape.size() == 1);
    int64_t blockSize = blockShape[0];
    int64_t vectorSize = vectorShape[0];
    unsigned numChunks = blockSize / vectorSize;

    Block *body = &op.getBody().front();

    // TODO: currently we unroll the generic op during lowering because
    // extractvalue cannot take a dynamic index. To reduce code size, we will
    // want to keep some sort of loop here for larger blocks
    for (unsigned i = 0; i < numChunks; ++i) {
      SmallVector<Value> chunkedArgs;

      for (auto [origArg, llvmArg] :
           llvm::zip(body->getArguments(), adaptor.getOperands())) {

        if (!isa<RankedTensorType>(origArg.getType())) {
          // forward constants and scalars without chunking
          chunkedArgs.push_back(origArg);
        } else {

          Type convertedBodyType =
              getTypeConverter()->convertType(origArg.getType());

          Value chunk = LLVM::UndefOp::create(rewriter, loc, convertedBodyType);

          for (unsigned j = 0; j < vectorSize; ++j) {
            int64_t srcIndex = i * vectorSize + j;

            Value extractedElement = LLVM::ExtractValueOp::create(
                rewriter, loc, llvmArg, {srcIndex});
            chunk = LLVM::InsertValueOp::create(rewriter, loc, chunk,
                                                extractedElement, {j});
          }

          Value castedChunk = UnrealizedConversionCastOp::create(
                                  rewriter, loc, origArg.getType(), chunk)
                                  .getResult(0);
          chunkedArgs.push_back(castedChunk);
        }
      }

      // clone the body of the generic op for this chunk only
      IRMapping mapping;
      for (unsigned j = 0; j < chunkedArgs.size(); j++) {
        mapping.map(body->getArgument(j), chunkedArgs[j]);
      }

      for (Operation &bOp : body->without_terminator()) {
        rewriter.clone(bOp, mapping);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateGenericOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GenericOpConversion>(typeConverter, benefit);
}
