#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Analysis/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

std::pair<unsigned, unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();
  assert(!shouldUseDistSmem(srcLayout, dstLayout));

  auto srcLinAttr = triton::gpu::toLinearEncoding(srcTy);
  auto dstLinAttr = triton::gpu::toLinearEncoding(dstTy);
  auto inOrd = srcLinAttr.getOrder();
  auto outOrd = dstLinAttr.getOrder();

  unsigned rank = srcTy.getRank();

  unsigned srcContigPerThread = srcLinAttr.getContigPerThread()[inOrd[0]];
  unsigned dstContigPerThread = dstLinAttr.getContigPerThread()[outOrd[0]];
  unsigned innerDim = rank - 1;
  unsigned inVec = outOrd[0] != innerDim  ? 1
                   : inOrd[0] != innerDim ? 1
                                          : srcContigPerThread;
  unsigned outVec = outOrd[0] != innerDim ? 1 : dstContigPerThread;

  return {inVec, outVec};
}

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  explicit ConvertLayoutOpConversion(LLVMTypeConverter &typeConverter,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout dstLayout = triton::gpu::toLinearLayout(dstTy);

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (llvm::is_contained(dims, kBlock)) {
      // Case 1: Transfer between values in different CTAs.
      //          This requires moving values through distributed shared memory.
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different CTAs");
    } else if (llvm::is_contained(dims, kWarp)) {
      // Case 2: Transfer between values in the same CTA, in which case we move
      //         values through shared memory.
      auto result = transferWithinBlock(op, adaptor.getSrc(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    } else if (llvm::is_contained(dims, kLane)) {
      // Case 3. Transfer between values in the same warp, in which case we try
      //         to move values using warp shuffles, though if the pattern is
      //         expensive enough we fall back to using shared memory
      return rewriter.notifyMatchFailure(
          op, "Triton CPU warp size should always be 1 - no transfer within "
              "warp needed.");
    } else if (llvm::is_contained(dims, kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(op, conversion, adaptor, rewriter);
    } else {
      // Cast 5. The two layouts are equivalent. We should probably remove
      // these in RemoveLayoutConversion.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
  }

  SmallVector<unsigned> getShapePerCTATile(RankedTensorType tensorTy) const {
    auto llEnc = triton::gpu::toLinearEncoding(tensorTy);
    auto sizePerThread = llEnc.getSizePerThread();
    auto threadsPerWarp = llEnc.getThreadsPerWarp();
    auto warpsPerCTA = llEnc.getWarpsPerCTA();
    llvm::errs() << "warpsPerCTA: ";
    for (size_t i = 0; i < warpsPerCTA.size(); i++)
      llvm::errs() << warpsPerCTA[i] << " ";
    llvm::errs() << "\n";
    SmallVector<unsigned> shape;
    for (auto [size, thread, warp] :
         llvm::zip(sizePerThread, threadsPerWarp, warpsPerCTA)) {
      shape.push_back(size * thread * warp);
    }
    return shape;
  }

  Value transferWithinBlock(triton::gpu::ConvertLayoutOp op, Value src,
                            ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    llvm::errs() << "srcTy: " << srcTy << "\n";
    llvm::errs() << "dstTy: " << dstTy << "\n";

    auto [inVec, outVec] = getScratchCvtInOutVecLengths(srcTy, dstTy);
    // TODO: these might be too big, but the problem right now is they're too small?
    llvm::errs() << "inVec: " << inVec << ", outVec: " << outVec << "\n";

    auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
    llvm::errs() << "srcShapePerCTA: ";
    for (auto s : srcShapePerCTA)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";
    auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    llvm::errs() << "dstShapePerCTA: ";
    for (auto s : dstShapePerCTA)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";

    auto srcShapePerCTATile = getShapePerCTATile(srcTy);
    llvm::errs() << "srcShapePerCTATile: ";
    for (auto s : srcShapePerCTATile)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";
    auto dstShapePerCTATile = getShapePerCTATile(dstTy);
    llvm::errs() << "dstShapePerCTATile: ";
    for (auto s : dstShapePerCTATile)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";

    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> repShape(rank);
    for (unsigned d = 0; d < rank; ++d) {
      repShape[d] = std::max(
          std::min<unsigned>(srcShapePerCTA[d], srcShapePerCTATile[d]),
          std::min<unsigned>(dstShapePerCTA[d], dstShapePerCTATile[d]));
    }
    llvm::errs() << "repShape: ";
    for (auto s : repShape)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";
    auto order = triton::gpu::getOrder(dstTy);

    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto dstLayout = triton::gpu::toLinearLayout(dstTy);

    // Pretty sure this is the identity function ATM
    // It'd be better to simply call `quotient({kBlock})` and
    // remove kBlock from transferWithinBlockImpl
    auto srcLayoutWithinBlock = triton::gpu::getLayoutWithinBlock(srcLayout);
    auto dstLayoutWithinBlock = triton::gpu::getLayoutWithinBlock(dstLayout);

    SmallVector<Value> inVals = unpackLLElements(loc, src, rewriter);
    assert(!inVals.empty());

    SmallVector<Value> outVals = transferWithinBlockImpl(
        inVals, op, srcLayoutWithinBlock, dstLayoutWithinBlock, repShape, order,
        inVec, outVec, rewriter);

    Value result =
        packLLElements(loc, typeConverter, outVals, rewriter, op.getType());
    return result;
  }

  // Determine which registers are read/written in which iteration of the shmem
  // transfer specified by `layout`.
  SmallVector<SmallVector<int> /*registers*/>
  collectRegsForIter(MLIRContext *ctx, const LinearLayout &layout) const {
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kIteration = str_attr("iteration");

    // The choice of iteration should be determined only by the register.  That
    // is, it should be correct to split the register dimension into iterations.
    assert(layout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    LinearLayout sublayout = layout.sublayout({kRegister}, {kIteration});
    SmallVector<SmallVector<int>> ret(sublayout.getOutDimSize(kIteration));
    for (int reg = 0; reg < sublayout.getInDimSize(kRegister); reg++) {
      auto idx = sublayout.apply({{kRegister, reg}});
      ret[idx.begin()->second].push_back(reg);
    }
    return ret;
  }

  SmallVector<Value> transferWithinBlockImpl(
      ArrayRef<Value> inVals, triton::gpu::ConvertLayoutOp op,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      const SmallVector<unsigned> &repShape, const SmallVector<unsigned> &order,
      const unsigned inVec, const unsigned outVec,
      RewriterBase &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kOffset = str_attr("offset");
    StringAttr kIteration = str_attr("iteration");

    llvm::errs() << "repShape: ";
    for (auto s : repShape)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";
    llvm::errs() << "order: ";
    for (auto s : order)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";

    auto tensorShapePerCTA =
        convertType<unsigned, int64_t>(triton::gpu::getShapePerCTA(
            op.getSrc().getType().getEncoding(), op.getType().getShape()));
    llvm::errs() << "tensorShapePerCTA: ";
    for (auto s : tensorShapePerCTA)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";

    LinearLayout sharedLayout =
        triton::gpu::chooseShemLayoutForRegToRegConversion(
            ctx, tensorShapePerCTA, repShape, order);
    llvm::errs() << "sharedLayout: " << sharedLayout << "\n";

    // Layout for the store from registers to shared memory.
    //
    // Note: If two threads in the same warp write to the same shmem offset, the
    // hardware resolves that without a stall or a bank conflict.  Therefore we
    // don't need to avoid duplicate writes.
    // Input dims: [reg, lane, warp]
    // Output dims: [offset, iteration]
    LinearLayout shmemStoreLayout = srcLayout.invertAndCompose(sharedLayout);
    llvm::errs() << "shmemStoreLayout: " << shmemStoreLayout << "\n";

    const int shmemAllocatedNumElems = getNumScratchElements(repShape);
    assert(shmemStoreLayout.getOutDimSize(kOffset) <= shmemAllocatedNumElems);

    /// Layout for the load from shmem to registers.
    LinearLayout shmemLoadLayout = dstLayout.invertAndCompose(sharedLayout);
    llvm::errs() << "shmemLoadLayout: " << shmemLoadLayout << "\n";

    // Check that the `register` fully determines the `iteration`.  That is,
    // each thread does exactly the same reads and writes to shmem on each
    // iteration, just with different input/output registers.
    assert(
        shmemStoreLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));
    assert(
        shmemLoadLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

#if 1
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
#else
    Value laneId = b.i32_val(0);
    Value warpId = getThreadId(rewriter, loc);
#endif 

    // iteration -> registers
    SmallVector<SmallVector<int>> inRegsForIter =
        collectRegsForIter(ctx, shmemStoreLayout);
    SmallVector<SmallVector<int>> outRegsForIter =
        collectRegsForIter(ctx, shmemLoadLayout);

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto sharedPtrTy = smemBase.getType();
    Type elemTy = inVals[0].getType();
    llvm::errs() << "elemTy: " << elemTy << "\n";
    auto outSize = shmemLoadLayout.getInDimSize(kRegister);
    auto iterations = sharedLayout.getInDimSize(kIteration);
    assert(inVec * iterations <= inVals.size());
    assert(outVec * iterations <= outSize);

    llvm::errs() << "outSize = " << outSize << ", outVec = " << outVec << ", iterations = " << iterations
                 << ", inVals.size() = " << inVals.size() << "\n";

    auto getVecAddr = [&](LinearLayout &layout, Value &regBase, int regSlice,
                          Value _laneId, Value _warpId) -> Value {
      Value offset = applyLinearLayout(loc, rewriter, layout,
                                       {{kRegister, b.i32_val(regSlice)},
                                        {kLane, _laneId},
                                        {kWarp, _warpId},
                                        { kBlock,
                                          b.i32_val(0) }})[0]
                         .second;
      auto vecAddr = b.gep(sharedPtrTy, elemTy, smemBase, offset);
      return vecAddr;
    };

    // register idx -> Value
    llvm::MapVector<int, Value> outVals;
    for (int i = 0; i < iterations; i++) {
      llvm::errs() << "Iteration " << i << " of " << iterations << "\n";
      if (i != 0)
        b.barrier();

      auto &inRegs = inRegsForIter[i];
      auto &outRegs = outRegsForIter[i];

      for (int j = 0; j < inVals.size() / iterations; j += inVec) {
        auto inRegSlice = inRegs[j];
        llvm::errs() << "Storing slice " << inRegSlice << " at location " << j
                     << " with inVec = " << inVec << ".\n";

        Value vecAddr =
            getVecAddr(shmemStoreLayout, smemBase, inRegSlice, laneId, warpId);
        SmallVector<Value> inValsVec;
        for (int k = 0; k < inVec; k++)
          inValsVec.push_back(inVals[inRegSlice + k]);
        Value valsVec = packLLVector(loc, inValsVec, rewriter);
        targetInfo.storeDShared(rewriter, loc, vecAddr, std::nullopt, valsVec,
                                /*pred=*/b.true_val());
      }

      b.barrier();

      for (int j = 0; j < outSize / iterations; j += outVec) {
        auto outRegSlice = outRegs[j];
        llvm::errs() << "loading slice " << outRegSlice << " at location " << j
                     << " with outVec = " << outVec << ".\n";
        auto vecAddr =
            getVecAddr(shmemLoadLayout, smemBase, outRegSlice, laneId, warpId);
        Value valsVec = targetInfo.loadDShared(
            rewriter, loc, vecAddr, std::nullopt, vec_ty(elemTy, outVec),
            /*pred=*/b.true_val());
        for (Value v : unpackLLVector(loc, valsVec, rewriter))
          outVals[outRegSlice++] = v;
      }
    }

    SmallVector<Value> outValsVec;
    for (size_t i = 0; i < outVals.size(); i++)
      outValsVec.push_back(outVals[i]);
    return outValsVec;
  }

  LogicalResult
  transferWithinThread(triton::gpu::ConvertLayoutOp op,
                       const LinearLayout &conversion, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getInDimSize(kRegister));
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = conversion.apply({{kRegister, i}}).begin()->second;
      outVals[i] = inVals[srcIdx];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo, benefit);
}
