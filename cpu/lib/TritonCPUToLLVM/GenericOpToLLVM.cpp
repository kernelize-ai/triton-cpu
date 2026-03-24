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

    Value result;
    const bool hasReductions = !op.getCombiners().empty();

    Block *body = &op.getBody().front();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // TODO: currently we unroll the generic op during lowering because
    // extractvalue cannot take a dynamic index. To reduce code size, we will
    // want to keep some sort of loop here for larger blocks
    for (unsigned i = 0; i < numChunks; ++i) {
      Value chunkOffset = b.i32_val(i * vectorSize);

      SmallVector<Value> chunkedArgs;

      for (auto [opIdx, origArg, llvmArg] : llvm::enumerate(
               body->getArguments().drop_front(), adaptor.getOperands())) {

        if (!isa<RankedTensorType>(origArg.getType())) {
          // forward constants and scalars without chunking
          assert(isa<PointerType>(origArg.getType()) ||
                 origArg.getType() == llvmArg.getType() &&
                     "expected non-tensor arguments to be unchanged by type "
                     "conversion");
          chunkedArgs.push_back(op.getOperand(opIdx));
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
      mapping.map(body->getArgument(0), chunkOffset);
      for (auto [bodyArg, chunkedArg] :
           llvm::zip(body->getArguments().drop_front(), chunkedArgs))
        mapping.map(bodyArg, chunkedArg);

      for (Operation &bOp : *body) {
        if (auto yieldOp = dyn_cast<cpu::YieldOp>(bOp)) {
          if (yieldOp.getValues().size() == 0)
            continue;

          assert(hasReductions &&
                 "unexpected yield op result in generic without reductions");
          auto yieldOpValues = llvm::to_vector(llvm::map_range(
              yieldOp.getValues(), [&](Value v) { return mapping.lookup(v); }));
          if (i == 0) {
            result = yieldOpValues[0];
          } else {
            // combine with the previous reduction result using the same
            // combiner region
            auto *combinerBlock = &op.getCombiners().front();
            IRMapping combMapping;
            combMapping.map(combinerBlock->getArgument(0), result);
            combMapping.map(combinerBlock->getArgument(1), yieldOpValues[0]);

            auto terminator =
                cast<cpu::YieldOp>(combinerBlock->getTerminator());
            auto yieldVals = terminator.getValues();
            assert(
                yieldVals.size() == 1 &&
                "expected exactly one value yielded from the combiner block");

            Operation *combinerOp = yieldVals.front().getDefiningOp();
            assert(combinerOp && "expected yielded value to be defined by an "
                                 "op in the combiner block");
            auto newCombiner = rewriter.clone(*combinerOp, combMapping);
            result = newCombiner->getResult(0);
          }
        } else {
          auto newOp = rewriter.clone(bOp, mapping);
        }
      }
    }

    if (result) {
      rewriter.replaceOp(op, result);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateGenericOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GenericOpConversion>(typeConverter, benefit);
}
