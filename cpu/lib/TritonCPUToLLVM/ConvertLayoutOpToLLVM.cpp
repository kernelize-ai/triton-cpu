#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

std::pair<unsigned, unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

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

  Value transferWithinBlock(triton::gpu::ConvertLayoutOp op, Value src,
                            ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto [inVec, outVec] = getScratchCvtInOutVecLengths(srcTy, dstTy);
    llvm::errs() << "inVec: " << inVec << ", outVec: " << outVec << "\n";

    auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
    auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> repShape(rank);
    for (unsigned d = 0; d < rank; ++d) {
      repShape[d] = std::max(srcShapePerCTA[d], dstShapePerCTA[d]);
    }

    auto tensorShapePerCTA =
        convertType<unsigned, int64_t>(triton::gpu::getShapePerCTA(
            op.getSrc().getType().getEncoding(), op.getType().getShape()));

    auto order = triton::gpu::getOrder(dstTy);
    LinearLayout sharedLayout =
        triton::gpu::chooseShemLayoutForRegToRegConversion(
            ctx, tensorShapePerCTA, repShape, order);
    llvm::errs() << "sharedLayout: " << sharedLayout << "\n";

    // TODO: move this code into the impl function
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto dstLayout = triton::gpu::toLinearLayout(dstTy);

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

    // Check that the `register` fully determines the `iteration`.  That is,
    // each thread does exactly the same reads and writes to shmem on each
    // iteration, just with different input/output registers.
    assert(
        shmemStoreLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));
    assert(
        shmemLoadLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    SmallVector<Value> inVals = unpackLLElements(loc, src, rewriter);
    assert(!inVals.empty());

    // Pretty sure this is the identity function ATM
    // It'd be better to simply call `quotient({kBlock})` and
    // remove kBlock from transferWithinBlockImpl
    auto srcLayoutWithinBlock = triton::gpu::getLayoutWithinBlock(srcLayout);
    auto dstLayoutWithinBlock = triton::gpu::getLayoutWithinBlock(dstLayout);

    SmallVector<Value> outVals = transferWithinBlockImpl(
        inVals, op, srcLayoutWithinBlock, dstLayoutWithinBlock, rewriter);

    Value result =
        packLLElements(loc, typeConverter, outVals, rewriter, op.getType());
    return result;
  }

  SmallVector<Value> transferWithinBlockImpl(ArrayRef<Value> inVals,
                                             triton::gpu::ConvertLayoutOp op,
                                             const LinearLayout &srcLayout,
                                             const LinearLayout &dstLayout,
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

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    assert(false);

    return {};
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
