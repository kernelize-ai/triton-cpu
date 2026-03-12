#include "TargetInfo.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    MLIRContext *ctx = rewriter.getContext();

    auto blockShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("blockShape");
    auto vectorShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("vectorShape");

    ArrayRef<int32_t> blockShape = blockShapeAttr.asArrayRef();
    ArrayRef<int32_t> vectorShape = vectorShapeAttr.asArrayRef();
    assert(blockShape.size() == vectorShape.size() && !blockShape.empty() &&
           "blockShape and vectorShape must be non-empty and of the same size");

    // TODO: support multi-dimensional shapes
    assert(blockShape.size() == 1);
    int64_t blockSize = blockShape[0];
    int64_t vectorSize = vectorShape[0];
    int64_t numChunks = blockSize / vectorSize;

    const bool hasReductions = !op.getCombiners().empty();
    Block *body = &op.getBody().front();

    Type i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Split the current block at the op's position:
    //   preheaderBlock: alloca/populate + peeled iteration 0
    //   continuationBlock: starts with op (erased by rewriter), then the rest
    Block *preheaderBlock = rewriter.getInsertionBlock();
    Block *continuationBlock =
        rewriter.splitBlock(preheaderBlock, rewriter.getInsertionPoint());

    // If there's a reduction result, add a block arg to continuationBlock
    // to carry the final accumulated value out of the loop.
    Value finalResult;
    if (hasReductions) {
      finalResult =
          continuationBlock->addArgument(op.getResult(0).getType(), loc);
    }

    // Create loop header and body blocks (only needed for numChunks > 1).
    // Both are inserted before continuationBlock so the block order is:
    //   preheaderBlock -> loopHeader -> loopBody -> continuationBlock
    Block *loopHeader = nullptr;
    Block *loopBody = nullptr;
    if (numChunks > 1) {
      // loopHeader carries the loop index and (if reducing) the accumulator.
      SmallVector<Type> headerArgTypes = {i64Ty};
      SmallVector<Location> headerArgLocs = {loc};
      if (hasReductions) {
        headerArgTypes.push_back(op.getResult(0).getType());
        headerArgLocs.push_back(loc);
      }
      loopHeader = rewriter.createBlock(continuationBlock, headerArgTypes,
                                        headerArgLocs);
      loopBody = rewriter.createBlock(continuationBlock);
    }

    // ── Preheader: spill tensor operands to allocas ──────────────────────────
    // Spilling avoids the ExtractValueOp constant-index requirement inside the
    // loop: we populate each alloca once here (static unroll, O(blockSize)),
    // then use GEP with a dynamic loop index inside the loop body.
    rewriter.setInsertionPointToEnd(preheaderBlock);

    // Locate the alloca insertion point in the function entry block so that
    // alloca ops are hoisted out of any loops, preventing unbounded stack
    // growth (stack overflows).
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(funcOp && "expected GenericOp to be inside an LLVMFuncOp");
    Block &entryBlock = funcOp.getBody().front();
    Block::iterator allocaInsertPt = entryBlock.begin();
    while (allocaInsertPt != entryBlock.end() &&
           isa<LLVM::AllocaOp>(*allocaInsertPt))
      ++allocaInsertPt;

    struct TensorInfo {
      Value allocaPtr;
      Type elemTy;
    };
    SmallVector<std::optional<TensorInfo>> tensorInfos(
        body->getNumArguments());
    SmallVector<Value> scalarVals(body->getNumArguments());

    Value blockSizeConst =
        LLVM::ConstantOp::create(rewriter, loc, i64Ty, blockSize);
    for (auto [idx, pair] : llvm::enumerate(
             llvm::zip(body->getArguments(), adaptor.getOperands()))) {
      auto [origArg, llvmArg] = pair;
      if (isa<RankedTensorType>(origArg.getType())) {
        auto tensorTy = cast<RankedTensorType>(origArg.getType());
        Type elemTy =
            getTypeConverter()->convertType(tensorTy.getElementType());

        // Hoist alloca to the function entry block so it executes at most once
        // per function call regardless of how many times this loop iterates.
        Value allocaPtr;
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(&entryBlock, allocaInsertPt);
          Value allocaSizeConst =
              LLVM::ConstantOp::create(rewriter, loc, i64Ty, blockSize);
          allocaPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemTy,
                                             allocaSizeConst,
                                             /*alignment=*/0);
          allocaInsertPt = rewriter.getInsertionPoint();
        }

        // Populate the alloca from the struct (static unroll).
        for (int k = 0; k < blockSize; ++k) {
          Value elem =
              LLVM::ExtractValueOp::create(rewriter, loc, llvmArg, {k});
          Value kConst =
              LLVM::ConstantOp::create(rewriter, loc, i64Ty, (int64_t)k);
          Value ptr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy,
                                          allocaPtr, ValueRange{kConst});
          LLVM::StoreOp::create(rewriter, loc, elem, ptr);
        }
        tensorInfos[idx] = TensorInfo{allocaPtr, elemTy};
      } else {
        assert(origArg.getType() == llvmArg.getType() &&
               "expected non-tensor arguments to be unchanged by type "
               "conversion");
        scalarVals[idx] = llvmArg;
      }
    }

    // vecSizeConst is used by buildChunkArgs in both preheader and loop body.
    // Both blocks are dominated by preheaderBlock, so defining it here is fine.
    Value vecSizeConst =
        LLVM::ConstantOp::create(rewriter, loc, i64Ty, vectorSize);

    // Build chunk args for a given runtime chunk index (i64 value).
    // Loads vectorSize elements from each tensor alloca at offset
    // chunkIdx*vectorSize, assembles them into a vectorSize-element struct,
    // and casts back to the original tile type expected by the body.
    auto buildChunkArgs = [&](Value chunkIdxVal) -> SmallVector<Value> {
      Value base = LLVM::MulOp::create(rewriter, loc, chunkIdxVal, vecSizeConst);
      SmallVector<Value> chunkedArgs;
      for (auto [idx, origArg] : llvm::enumerate(body->getArguments())) {
        if (tensorInfos[idx]) {
          auto &info = *tensorInfos[idx];
          Type chunkStructTy =
              getTypeConverter()->convertType(origArg.getType());
          Value chunk = LLVM::UndefOp::create(rewriter, loc, chunkStructTy);
          for (int j = 0; j < vectorSize; ++j) {
            Value jConst =
                LLVM::ConstantOp::create(rewriter, loc, i64Ty, (int64_t)j);
            Value offset = LLVM::AddOp::create(rewriter, loc, base, jConst);
            Value ptr = LLVM::GEPOp::create(rewriter, loc, ptrTy, info.elemTy,
                                            info.allocaPtr, ValueRange{offset});
            Value elem = LLVM::LoadOp::create(rewriter, loc, info.elemTy, ptr);
            chunk = LLVM::InsertValueOp::create(rewriter, loc, chunk, elem,
                                                {(int64_t)j});
          }
          Value castedChunk =
              UnrealizedConversionCastOp::create(rewriter, loc,
                                                 origArg.getType(), chunk)
                  .getResult(0);
          chunkedArgs.push_back(castedChunk);
        } else {
          chunkedArgs.push_back(scalarVals[idx]);
        }
      }
      return chunkedArgs;
    };

    // Clone body ops with the given chunk args; return the partial reduction
    // result (if any). Emits into the current insertion block.
    auto cloneBodyOps = [&](SmallVector<Value> chunkedArgs) -> Value {
      IRMapping mapping;
      for (auto [j, arg] : llvm::enumerate(body->getArguments()))
        mapping.map(arg, chunkedArgs[j]);
      Value partial;
      for (Operation &bOp : body->without_terminator()) {
        auto *newOp = rewriter.clone(bOp, mapping);
        if (isa<triton::ReduceOp>(bOp)) {
          assert(hasReductions &&
                 "unexpected reduce op in generic without reductions");
          partial = newOp->getResult(0);
        }
      }
      return partial;
    };

    // Apply the combiner region to (acc, partial) to produce a new accumulator.
    auto applyCombiner = [&](Value acc, Value partial) -> Value {
      auto *combinerBlock = &op.getCombiners().front();
      IRMapping combMapping;
      combMapping.map(combinerBlock->getArgument(0), acc);
      combMapping.map(combinerBlock->getArgument(1), partial);
      auto terminator = cast<cpu::YieldOp>(combinerBlock->getTerminator());
      Operation *combinerOp =
          terminator.getValues().front().getDefiningOp();
      assert(combinerOp && "expected yielded value to be defined by an op");
      return rewriter.clone(*combinerOp, combMapping)->getResult(0);
    };

    // ── Peel iteration 0 (still in preheaderBlock) ───────────────────────────
    // Bootstrap the accumulator from the first chunk's partial result so that
    // the loop body can always apply the combiner without needing an identity.
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, (int64_t)0);
    Value partial0 = cloneBodyOps(buildChunkArgs(zero));

    if (numChunks == 1) {
      // No loop needed; fall straight through to the continuation.
      SmallVector<Value> contArgs;
      if (hasReductions)
        contArgs.push_back(partial0);
      LLVM::BrOp::create(rewriter, loc, contArgs, continuationBlock);
    } else {
      // Branch to loopHeader with initial values: i=1, acc=partial0.
      Value one = LLVM::ConstantOp::create(rewriter, loc, i64Ty, (int64_t)1);
      SmallVector<Value> headerInitArgs = {one};
      if (hasReductions)
        headerInitArgs.push_back(partial0);
      LLVM::BrOp::create(rewriter, loc, headerInitArgs, loopHeader);

      // ── Loop header: check i < numChunks ──────────────────────────────────
      rewriter.setInsertionPointToEnd(loopHeader);
      Value loopIdx = loopHeader->getArgument(0);
      Value acc = hasReductions ? loopHeader->getArgument(1) : nullptr;
      Value numChunksConst =
          LLVM::ConstantOp::create(rewriter, loc, i64Ty, numChunks);
      Value cond = LLVM::ICmpOp::create(rewriter, loc,
                                        LLVM::ICmpPredicate::ult,
                                        loopIdx, numChunksConst);
      SmallVector<Value> exitArgs;
      if (hasReductions)
        exitArgs.push_back(acc);
      LLVM::CondBrOp::create(rewriter, loc, cond, loopBody, ValueRange{},
                              continuationBlock, exitArgs);

      // ── Loop body: process chunk, combine, increment ───────────────────────
      rewriter.setInsertionPointToEnd(loopBody);
      Value partial = cloneBodyOps(buildChunkArgs(loopIdx));
      Value newAcc = hasReductions ? applyCombiner(acc, partial) : nullptr;
      Value nextIdx = LLVM::AddOp::create(rewriter, loc, loopIdx, one);
      SmallVector<Value> backEdgeArgs = {nextIdx};
      if (hasReductions)
        backEdgeArgs.push_back(newAcc);
      LLVM::BrOp::create(rewriter, loc, backEdgeArgs, loopHeader);
    }

    if (hasReductions) {
      rewriter.replaceOp(op, finalResult);
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
