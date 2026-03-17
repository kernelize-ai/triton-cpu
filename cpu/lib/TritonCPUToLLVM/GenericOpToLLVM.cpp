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

  // Builds chunked args for chunk index i by statically slicing each tensor
  // operand using extractvalue/insertvalue. Non-tensor operands are forwarded
  // unchanged. Only valid for compile-time-known chunk indices.
  SmallVector<Value> buildStaticChunkedArgs(cpu::GenericOp op,
                                            OpAdaptor adaptor,
                                            ConversionPatternRewriter &rewriter,
                                            unsigned i,
                                            unsigned vectorSize) const {
    Location loc = op.getLoc();
    Block *body = &op.getBody().front();
    SmallVector<Value> chunkedArgs;

    for (auto [opIdx, origArg, llvmArg] :
         llvm::enumerate(body->getArguments(), adaptor.getOperands())) {

      if (!isa<RankedTensorType>(origArg.getType())) {
        // forward constants and scalars without chunking
        assert(origArg.getType() == llvmArg.getType() &&
               "expected non-tensor arguments to be unchanged by type "
               "conversion");
        chunkedArgs.push_back(op.getOperand(opIdx));
      } else {
        Type convertedBodyType =
            getTypeConverter()->convertType(origArg.getType());

        Value chunk = LLVM::UndefOp::create(rewriter, loc, convertedBodyType);

        for (unsigned j = 0; j < vectorSize; ++j) {
          int64_t srcIndex = i * vectorSize + j;

          Value extractedElement =
              LLVM::ExtractValueOp::create(rewriter, loc, llvmArg, {srcIndex});
          chunk = LLVM::InsertValueOp::create(rewriter, loc, chunk,
                                              extractedElement, {j});
        }

        Value castedChunk = UnrealizedConversionCastOp::create(
                                rewriter, loc, origArg.getType(), chunk)
                                .getResult(0);
        chunkedArgs.push_back(castedChunk);
      }
    }

    return chunkedArgs;
  }

  // Clones the generic body for a given set of chunked args and folds the
  // yielded result into `result` via the combiner. On the first call `result`
  // must be a null Value; subsequent calls accumulate via the combiner.
  void emitChunkBody(cpu::GenericOp op, ConversionPatternRewriter &rewriter,
                     ArrayRef<Value> chunkedArgs, Value &result) const {
    Block *body = &op.getBody().front();
    const bool hasReductions = !op.getCombiners().empty();

    IRMapping mapping;
    for (auto [bodyArg, chunkedArg] :
         llvm::zip(body->getArguments(), chunkedArgs))
      mapping.map(bodyArg, chunkedArg);

    for (Operation &bOp : *body) {
      if (auto yieldOp = dyn_cast<cpu::YieldOp>(bOp)) {
        if (yieldOp.getValues().empty())
          continue;

        assert(hasReductions &&
               "unexpected yield op result in generic without reductions");
        auto yieldOpValues = llvm::to_vector(llvm::map_range(
            yieldOp.getValues(), [&](Value v) { return mapping.lookup(v); }));

        if (!result) {
          result = yieldOpValues[0];
        } else {
          // combine with the previous reduction result using the combiner
          // region
          auto *combinerBlock = &op.getCombiners().front();
          IRMapping combMapping;
          combMapping.map(combinerBlock->getArgument(0), result);
          combMapping.map(combinerBlock->getArgument(1), yieldOpValues[0]);

          auto terminator = cast<cpu::YieldOp>(combinerBlock->getTerminator());
          auto yieldVals = terminator.getValues();
          assert(yieldVals.size() == 1 &&
                 "expected exactly one value yielded from the combiner block");

          Operation *combinerOp = yieldVals.front().getDefiningOp();
          assert(combinerOp && "expected yielded value to be defined by an "
                               "op in the combiner block");
          auto newCombiner = rewriter.clone(*combinerOp, combMapping);
          result = newCombiner->getResult(0);
        }
      } else {
        rewriter.clone(bOp, mapping);
      }
    }
  }

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

    if (false && numChunks > 10) {
      // TODO: for large numbers of chunks the generated IR can become quite
      // big. Generate a dynamic loop and use alloca to store/load the tensor.
    } else {
      for (unsigned i = 0; i < numChunks; ++i) {
        auto chunkedArgs =
            buildStaticChunkedArgs(op, adaptor, rewriter, i, vectorSize);
        emitChunkBody(op, rewriter, chunkedArgs, result);
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
