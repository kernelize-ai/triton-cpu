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

  // TODO: rename everything from chunk -> tile? is tile the nomenclature we
  // want to adopt?
  SmallVector<Value> buildStaticChunkedArgs(cpu::GenericOp op,
                                            OpAdaptor adaptor,
                                            ConversionPatternRewriter &rewriter,
                                            unsigned i,
                                            unsigned vectorSize) const {
    Location loc = op.getLoc();
    Block *body = &op.getBody().front();
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

  void emitTileBody(cpu::GenericOp op, ConversionPatternRewriter &rewriter,
                    ArrayRef<Value> chunkedArgs, Value tileOffset,
                    Value &result) const {
    Block *body = &op.getBody().front();
    const bool hasReductions = !op.getCombiners().empty();

    // clone the body of the generic op for this chunk only
    IRMapping mapping;
    mapping.map(body->getArgument(0), tileOffset);
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
        if (!result) {
          result = yieldOpValues[0];
        } else {
          // combine with the previous reduction result using the same
          // combiner region
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
    const bool hasReductions = !op.getCombiners().empty();

    Block *body = &op.getBody().front();

    const bool hasTensorArgs =
        llvm::any_of(body->getArguments(), [](BlockArgument arg) {
          return isa<RankedTensorType>(arg.getType());
        });

    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // if the generic op has tensor args we materialize the input tensors then
    // unroll the loop over tiles, slicing each tensor with the current tile
    // index. llvm.extractvalue only supports attr indices, so we need to know
    // the individual tile indices at compile time. However, a generic with no
    // tensor args can be lowered as a loop which dramatically reduces code
    // size. Note that generics without tensor args can still materialize
    // tensors within the body of the generic. If we only have 1 chunk then
    // there's no need to generate the runtime loop and we "unroll" regardless
    // of the input type
    if (hasTensorArgs || numChunks == 1) {
      for (unsigned i = 0; i < numChunks; ++i) {

        SmallVector<Value> chunkedArgs =
            buildStaticChunkedArgs(op, adaptor, rewriter, i, vectorSize);

        Value chunkOffset = b.i32_val(i * vectorSize);

        emitTileBody(op, rewriter, chunkedArgs, chunkOffset, result);
      }
    } else {
      Value one = b.i32_val(1);
      Value numChunksVal = b.i32_val(numChunks);
      // peel the first chunk to establish an initial "result" for reductions
      // TODO  we could only do this if we have reductions but it probably won't
      // hurt anything
      auto firstArgs =
          buildStaticChunkedArgs(op, adaptor, rewriter, 0, vectorSize);
      emitTileBody(op, rewriter, firstArgs, /*tileOffset=*/b.i32_val(0),
                   result);

      // build the loop infrastructure for chunks 1...numChunks-1
      Block *currentBlock = rewriter.getInsertionBlock();
      Block *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

      Value afterResult;
      if (hasReductions)
        afterResult = afterBlock->addArgument(result.getType(), loc);

      SmallVector<Type> headerArgTypes = {i32_ty};
      SmallVector<Location> headerArgLocs = {loc};
      if (hasReductions) {
        headerArgTypes.push_back(result.getType());
        headerArgLocs.push_back(loc);
      }
      Block *loopHeader =
          rewriter.createBlock(afterBlock, headerArgTypes, headerArgLocs);
      Block *loopBody = rewriter.createBlock(afterBlock);

      // currentBlock -> loopHeader(1, result)
      rewriter.setInsertionPointToEnd(currentBlock);
      {
        SmallVector<Value> headerInitArgs = {b.i32_val(1)};
        if (hasReductions)
          headerInitArgs.push_back(result);
        LLVM::BrOp::create(rewriter, loc, headerInitArgs, loopHeader);
      }

      // loopHeader: check bound, branch to body or exit
      rewriter.setInsertionPointToEnd(loopHeader);
      Value loopI = loopHeader->getArgument(0);
      Value loopAcc = hasReductions ? loopHeader->getArgument(1) : Value{};
      Value cond = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                        loopI, numChunksVal);
      SmallVector<Value> exitArgs;
      if (hasReductions)
        exitArgs.push_back(loopAcc);
      LLVM::CondBrOp::create(rewriter, loc, cond, loopBody, {}, afterBlock,
                             exitArgs);

      // populate the loop body
      rewriter.setInsertionPointToEnd(loopBody);
      {
        SmallVector<Value> tileArgs;
        for (auto [opIdx, origArg, llvmArg] : llvm::enumerate(
                 body->getArguments().drop_front(), adaptor.getOperands())) {
          assert(!isa<RankedTensorType>(origArg.getType()) &&
                 "tensor types are not allowed in compile-time generated "
                 "generic tile loops");
          // forward the type from the generic body to the loop body
          assert(isa<PointerType>(origArg.getType()) ||
                 origArg.getType() == llvmArg.getType() &&
                     "expected non-tensor arguments to be unchanged by type "
                     "conversion");
          tileArgs.push_back(op.getOperand(opIdx));
        }
        Value tileResult = loopAcc;
        Value tileOffset =
            LLVM::MulOp::create(rewriter, loc, loopI, b.i32_val(vectorSize));
        emitTileBody(op, rewriter, tileArgs, tileOffset, tileResult);

        Value nextI = LLVM::AddOp::create(rewriter, loc, loopI, one);
        SmallVector<Value> backArgs = {nextI};
        if (hasReductions)
          backArgs.push_back(tileResult);
        LLVM::BrOp::create(rewriter, loc, backArgs, loopHeader);
      }

      // forward the final result through the after block and to the next op
      rewriter.setInsertionPointToStart(afterBlock);
      if (hasReductions)
        result = afterResult;
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
