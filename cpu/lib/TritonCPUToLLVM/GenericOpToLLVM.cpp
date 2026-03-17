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
    const bool hasReductions = !op.getCombiners().empty();

    if (numChunks > 1) {
      Block *body = &op.getBody().front();
      auto i64Type = rewriter.getI64Type();
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

      Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                           rewriter.getI64IntegerAttr(1));
      Value numChunksVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(numChunks));

      // Step 1: hoist one alloca per tensor operand to the function entry
      // block so it is not re-executed on each call when the generic is inside
      // a loop.
      SmallVector<Value> tensorAllocas(body->getNumArguments(), Value{});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
        assert(funcOp && "expected generic op inside an LLVM function");
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());

        for (auto [idx, origArg, llvmArg] :
             llvm::enumerate(body->getArguments(), adaptor.getOperands())) {
          auto tensorTy = dyn_cast<RankedTensorType>(origArg.getType());
          if (!tensorTy)
            continue;
          auto structType = cast<LLVM::LLVMStructType>(llvmArg.getType());
          // should we get this from the structType or the typeconverter on
          // tensorTy element type?
          Type elemType = structType.getBody().front();
          unsigned tensorSize = blockSize;
          Value arraySizeVal = LLVM::ConstantOp::create(
              rewriter, loc, i64Type, rewriter.getI64IntegerAttr(tensorSize));
          tensorAllocas[idx] = LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                                      elemType, arraySizeVal);
        }
      }

      // Step 2: store each struct operand into its alloca (compile-time
      // unrolled over blockSize — only runs once per generic invocation).
      for (auto [idx, origArg, llvmArg] :
           llvm::enumerate(body->getArguments(), adaptor.getOperands())) {
        auto tensorTy = dyn_cast<RankedTensorType>(origArg.getType());
        if (!tensorTy)
          continue;
        auto structType = cast<LLVM::LLVMStructType>(llvmArg.getType());
        Type elemType = structType.getBody().front();
        // TODO: we should make sure we cannot get multiple sizes of tensors to
        // ttc.generic
        unsigned tensorSize = blockSize;
        for (int64_t j = 0; j < tensorSize; ++j) {
          Value elem =
              LLVM::ExtractValueOp::create(rewriter, loc, llvmArg, {j});
          Value jVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                rewriter.getI64IntegerAttr(j));
          Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrType, elemType,
                                              tensorAllocas[idx],
                                              ArrayRef<LLVM::GEPArg>{jVal});
          LLVM::StoreOp::create(rewriter, loc, elem, elemPtr);
        }
      }

      // Step 3: peel the first chunk to establish the initial `result`.
      // emitChunkBody initializes result (null → first yield value) on the
      // first call, so we can reuse it here by passing a null result.
      {
        auto firstArgs =
            buildStaticChunkedArgs(op, adaptor, rewriter, 0, vectorSize);
        emitChunkBody(op, rewriter, firstArgs, result);
      }

      // Step 4: build the loop for chunks 1..numChunks-1.
      //
      // Block layout (in source order):
      //   currentBlock  →  loopHeader(i, [acc])  ↔  loopBody
      //                                           ↘  afterBlock([result])
      //
      // loopHeader block args:
      //   %i   : i64          — chunk index, starts at 1
      //   %acc : <resultType> — running accumulator (only when hasReductions)
      //
      // afterBlock block args:
      //   %finalResult : <resultType>  (only when hasReductions)

      Block *currentBlock = rewriter.getInsertionBlock();
      Block *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

      // Add the result arg to afterBlock before wiring branches to it.
      Value afterResult;
      if (hasReductions)
        afterResult = afterBlock->addArgument(result.getType(), loc);

      SmallVector<Type> headerArgTypes = {i64Type};
      SmallVector<Location> headerArgLocs = {loc};
      if (hasReductions) {
        headerArgTypes.push_back(result.getType());
        headerArgLocs.push_back(loc);
      }
      Block *loopHeader =
          rewriter.createBlock(afterBlock, headerArgTypes, headerArgLocs);
      Block *loopBody = rewriter.createBlock(afterBlock);

      // currentBlock → loopHeader(1, result)
      rewriter.setInsertionPointToEnd(currentBlock);
      SmallVector<Value> initArgs = {one};
      if (hasReductions)
        initArgs.push_back(result);
      LLVM::BrOp::create(rewriter, loc, initArgs, loopHeader);

      // loopHeader: if i < numChunks goto loopBody else goto afterBlock(acc)
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

      // loopBody: load chunk from allocas, run body, combine with loopAcc,
      // then branch back to loopHeader.
      rewriter.setInsertionPointToEnd(loopBody);
      {
        SmallVector<Value> chunkArgs;
        for (auto [idx, origArg] : llvm::enumerate(body->getArguments())) {
          auto tensorTy = dyn_cast<RankedTensorType>(origArg.getType());
          if (!tensorTy) {
            // forward constants and scalars without chunking
            chunkArgs.push_back(op.getOperand(idx));
          } else {
            // Load vectorSize elements from tensorAllocas[idx] at offset
            // loopI * vectorSize + j using a dynamic GEP, build the chunk
            // struct, and cast it back to the original tensor type —
            // mirroring buildStaticChunkedArgs but with dynamic GEP.
            auto tensorAlloca = tensorAllocas[idx];
            Type chunkStructTy = getTypeConverter()->convertType(tensorTy);
            Type elemType =
                getTypeConverter()->convertType(tensorTy.getElementType());
            Value chunk = LLVM::UndefOp::create(rewriter, loc, chunkStructTy);

            Value vectorSizeVal = LLVM::ConstantOp::create(
                rewriter, loc, i64Type, rewriter.getI64IntegerAttr(vectorSize));
            Value baseOffset =
                LLVM::MulOp::create(rewriter, loc, loopI, vectorSizeVal);

            for (unsigned j = 0; j < vectorSize; ++j) {
              Value jVal = LLVM::ConstantOp::create(
                  rewriter, loc, i64Type, rewriter.getI64IntegerAttr(j));
              Value offset =
                  LLVM::AddOp::create(rewriter, loc, baseOffset, jVal);
              Value elemPtr = LLVM::GEPOp::create(
                  rewriter, loc, ptrType, elemType, tensorAlloca,
                  ArrayRef<LLVM::GEPArg>{offset});
              Value elem =
                  LLVM::LoadOp::create(rewriter, loc, elemType, elemPtr);
              chunk =
                  LLVM::InsertValueOp::create(rewriter, loc, chunk, elem, {j});
            }

            Value castedChunk = UnrealizedConversionCastOp::create(
                                    rewriter, loc, tensorTy, chunk)
                                    .getResult(0);
            chunkArgs.push_back(castedChunk);
          }
        }

        // emitChunkBody combines with `result` when result is non-null.
        // Seed it with loopAcc so that the combine path is always taken.
        Value chunkResult = loopAcc;
        emitChunkBody(op, rewriter, chunkArgs, chunkResult);
        // After the call, chunkResult holds the newly combined accumulator.

        Value nextI = LLVM::AddOp::create(rewriter, loc, loopI, one);
        SmallVector<Value> backArgs = {nextI};
        if (hasReductions)
          backArgs.push_back(chunkResult);
        LLVM::BrOp::create(rewriter, loc, backArgs, loopHeader);
      }

      // Continue emission in afterBlock; the final result comes from the
      // block argument we added above.
      rewriter.setInsertionPointToStart(afterBlock);
      if (hasReductions)
        result = afterResult;
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
