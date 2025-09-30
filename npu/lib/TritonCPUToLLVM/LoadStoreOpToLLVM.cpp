#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (a && b) {
    return tb.and_(a, b);
  }
  return a ? a : b;
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask. If no predicate is required, return true.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const npu::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, dimPred);
    }
  }
  return pred;
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const npu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = builder.create<LLVM::ConstantOp>(loc, vecTy, denseValue);
    return zeroVal;
  }

  // Create a LLVM vector of type `vecTy` containing all true values
  Value createTrueVector(OpBuilder &builder, Location loc,
                         VectorType inVecTy) const {
    mlir::Attribute oneAttr = builder.getBoolAttr(true);
    VectorType vecTy = VectorType::get(inVecTy.getShape(), builder.getI1Type());

    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), oneAttr);
    Value trueVal = builder.create<LLVM::ConstantOp>(loc, vecTy, denseValue);
    return trueVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(ConversionPatternRewriter &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 256 bits on the avg CPU (TODO: be more
    // specific?)
    return std::min<unsigned>(256 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const npu::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const npu::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    LDBG("Load value LLVM Type: " << valueElemTy);
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    unsigned vecOrig = vec;
    if (llMask) {
      LDBG("vec = " << vec << " mask_alignment = " << getMaskAlignment(mask));
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      LDBG(" vec (post mask alignment adjustment) = " << vec);
    }
    if (vec == 1 && numElems > 1) {
      int maskValue = !llMask ? -1 : getMaskAlignment(mask);
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " origin vec = " << vecOrig
                       << " numElems = " << numElems << " mask is " << maskValue
                       << "\n";
    }
    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;

    // Load redundantly in all dims except reg
    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    LDBG("LoadOp numElems = " << numElems << " vec = " << vec
                              << " valueElemNBits = " << valueElemNBits
                              << ", vecTy = " << vecTy);

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (auto canonicalVecStart = getCanonicalIndex(vecStart, regMask);
          vecStart != canonicalVecStart) {
        // For redundant registers, refer back to the canonical load
        for (auto iVec = 0; iVec < vec; ++iVec) {
          loadedVals.push_back(loadedVals[canonicalVecStart + iVec]);
        }
        continue;
      }

      Value pred =
          mask ? packElementRangeIntoVector(
                     rewriter, getTypeConverter(), loc,
                     cast<VectorType>(LLVM::getVectorType(i1_ty, vec)),
                     maskElems, vecStart)
               : createTrueVector(rewriter, loc, cast<VectorType>(vecTy));
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(rewriter, getTypeConverter(), loc,
                                              cast<VectorType>(vecTy),
                                              otherElems, vecStart);

      const uint32_t alignment = vec * valueElemNBytes;
      Value loadVec =
          npu::llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, alignment);

      for (size_t ii = 0; ii < vec; ii++) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loadedVal = b.extract_element(valueElemTy, loadVec, vecIdx);
        loadedVals.push_back(loadedVal);
      }
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(op.getType());
    LDBG("loadedVals Size = " << loadedVals.size());
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const npu::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();
    LDBG("Lower StoreOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert store with a tensor pointer into LLVM; "
           "this case should be transformed to normal store before lowering");
    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getValue();
    Value llMask = adaptor.getMask();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    unsigned vecOrig = vec;
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    auto vecTy = LLVM::getVectorType(valueElemTy, vec);

    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    LDBG("StoreOp numElems = " << elemsPerThread << " vec = " << vec
                               << " valueElemNBits = " << valueElemNBits << " "
                               << valueTy);
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred = packLLVector(
          loc, ValueRange{llvm::SmallVector<Value>(vec, threadPred)}, rewriter);

      if (maskElems.size()) {
        Value maskVector = packElementRangeIntoVector(
            rewriter, getTypeConverter(), loc,
            cast<VectorType>(LLVM::getVectorType(i1_ty, vec)), maskElems,
            vecStart);
        pred = b.and_(pred, maskVector);
      }

      // predicated store
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);

      const uint32_t alignment = vec * valueElemNBytes;
      npu::llStore(rewriter, loc, b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 0)),
                   storeVal, pred, alignment);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(const LLVMTypeConverter &converter,
                                     const npu::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>(converter,
                                                                      benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#if 1

    auto ctx = getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // %src
    // note that srcElems come from a global memory space allocation, so we need
    // to bitcast back to address space 0
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // We assume other = 0, see XXX(Keren) below
    // %other
    // SmallVector<Value> otherElems;
    // if (llOther) {
    //   otherElems = unpackLLElements(loc, llOther, rewriter);
    //   assert(srcElems.size() == otherElems.size());
    // }

    // zip(src, mask)
    SmallVector<Value> vals;
    auto ptrTy = ptr_ty(
        ctx, 0); // srcElems[0].getType(); don't use the struct ptr type because
                 // it is the wrong address space (seems fishy but ok for now)
    auto structTy =
        LLVM::LLVMStructType::getLiteral(ctx, ArrayRef<Type>{ptrTy, i1_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = rewriter.create<LLVM::UndefOp>(loc, structTy);
      packedArr = b.insert_val(packedArr,
                               b.addrspacecast(ptr_ty(ctx, 0), srcElems[i]), 0);
      auto maskElem = llMask ? maskElems[i] : b.false_val();
      packedArr = b.insert_val(packedArr, maskElem, 1);
      vals.push_back(packedArr);
    }

    // Remove broadcasted registers
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    vals = removeBroadcastSrc.apply(vals);

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned maxVec = getContiguity(op.getSrc());
    if (mask) {
      maxVec = std::min(maxVec, getMaskAlignment(mask));
    }
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    // maxVec = std::min(maxVec, 128 / resElemTy.getIntOrFloatBitWidth());

    int vecBytes = maxVec * resElemTy.getIntOrFloatBitWidth() / 8;
    auto freeVarMasks = getFreeVariableMasks(srcTy);
    // NOTE(@peterbell10): We load redundant data on different CTAs, so the data
    // is available in each CTAs respective shared memory. Otherwise, we would
    // need an additional broadcast step to copy the data between CTAs.
    freeVarMasks[str_attr("block")] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    auto emitCpAsync = [&b, threadPred, ptrTy, hasMask = bool(llMask)](
                           RewriterBase &rewriter, Location loc,
                           ArrayRef<Value> vals, Value shmemAddr, int startIdx,
                           VectorType vecTy) -> SmallVector<Value> {
      assert(isa<VectorType>(vecTy));
#if 1
      auto ctx = rewriter.getContext();
      auto i1Ty = rewriter.getI1Type();
      auto elemTy = vecTy.getElementType();
      const int64_t nElts = vecTy.getNumElements();
      // auto elemPtrTy = ptr_ty(ctx, 0);

      // Unpack {srcPtr, mask}
      auto structElem = vals[startIdx];
      // cast ptrs from address space 3 (default shared on GPU) to address space
      // 0
      Value srcPtr = b.extract_val(ptrTy, structElem, 0);
      Value maskVal = hasMask ? b.extract_val(i1Ty, structElem, 1) : Value();

      // Cast base pointers to element pointers if needed.
      // (Adjust addrspace/types for your IR: this assumes flat pointers.)

      Value srcBase = srcPtr;   // b.bitcast(ptrTy, srcPtr);
      Value shBase = shmemAddr; // b.bitcast(ptrTy, shmemAddr);

      // Constants
      Value c0 = rewriter.create<LLVM::ConstantOp>(
          loc, elemTy, rewriter.getZeroAttr(elemTy));
      Value truePred = b.true_val();
      Value laneMaskSplat = hasMask ? maskVal : truePred;
      Value threadPredVal = threadPred ? threadPred : truePred;

      // Combined predicate: threadPred && laneMask (scalar path uses same pred
      // for each lane)
      Value combinedPred = b.and_(threadPredVal, laneMaskSplat);

      // Scalarize: GEP each lane, load (predicated, zero-fill), store
      // (predicated)
      for (int64_t i = 0; i < nElts; ++i) {
        Value idx = b.i64_val(i);

        // src_i = &srcBase[i], sh_i = &shBase[i]
        Value src_i = b.gep(ptrTy, elemTy, srcBase, idx);
        Value sh_i = b.gep(ptrTy, elemTy, shBase, idx);

        // If you have a vector mask (e.g., vector<nxi1>), replace combinedPred
        // with mask[i]:
        //   Value laneMask = b.extract_element(i1Ty, maskVec, idx);
        //   Value lanePred = b.and_(threadPredVal, laneMask);
        Value lanePred = combinedPred;

        // Note: alignment is not currently used in llLoad - it is computed when
        // lowering the generated cpu masked load op to LLVM. If that changes,
        // we need to change this value here as it is likely too conservative.
        // load with zero-fill on miss (pred=false)
        Value val_i =
            npu::llLoad(rewriter, loc, src_i, elemTy, /*pred=*/lanePred,
                        /*falseVal=*/c0, /*alignment=*/1);

        // store under the same predicate
        npu::llStore(rewriter, loc, sh_i, val_i, /*pred=*/lanePred,
                     /*alignment=*/1);
      }
#else
      auto *ctx = rewriter.getContext();
      auto elemTy = vecTy.getElementType();
      auto nBytes = vecTy.getNumElements() * elemTy.getIntOrFloatBitWidth() / 8;
      // assert(nBytes == 16 || nBytes == 8 || nBytes == 4);

      auto structElem = vals[startIdx];
      auto srcElem = b.extract_val(ptrTy, structElem, 0);
      auto maskElem = b.extract_val(i1_ty, structElem, 1);

      // Value loadVec =
      // npu::llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, alignment);

      PTXBuilder ptxBuilder;
      auto &copyAsyncOp =
          *ptxBuilder.create<PTXCpAsyncLoadInstr>(srcCacheModifier);
      auto *dstOperand = ptxBuilder.newAddrOperand(shmemAddr, "r");
      auto *srcOperand = ptxBuilder.newAddrOperand(srcElem, "l");
      auto *copySize = ptxBuilder.newConstantOperand(nBytes);
      auto *srcSize = copySize;
      if (hasMask) {
        // We don't use predicate in this case, setting src-size to 0
        // if there's any mask. cp.async will automatically fill the
        // remaining slots with 0 if cp-size > src-size.
        // XXX(Keren): Always assume other = 0 for now.
        // When 'other != 0' is supported, we will need to fold the
        // op.getMask() and redundantDataMask() into the same predicate, the
        // way it is done for LoadOp.
        auto selectOp = b.select(maskElem, b.i32_val(nBytes), b.i32_val(0));
        srcSize = ptxBuilder.newOperand(selectOp, "r");
      }
      copyAsyncOp(dstOperand, srcOperand, copySize, srcSize)
          .maybePredicate(threadPred);
      ptxBuilder.launch(rewriter, loc, void_ty(ctx));
#endif
      return {};
    };

    // %dst
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto smemLayout = triton::gpu::toLinearLayout(dstTy);
    auto cvt = srcLayout.invertAndCompose(smemLayout);
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(loc,
                       "cp.async does not support non-trivial block dimension");
    }
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset = SharedMemoryObject::getMaskSpanOffsets(dstTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    lowerLdSt(
        loc, ctx, cvt, vals, resElemTy, smemObj.getBase(),
        [](Value v) { return v; }, affineOffset, maskSpanAffineOffset, laneId,
        warpId, rewriter, targetInfo, maxVec, emitCpAsync);

    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);

    return success();
#else
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstEnc = dstTy.getEncoding();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value llDst = adaptor.getResult();

    unsigned vec = getVectorSize(op.getSrc());
    Value mask = op.getMask();
    Value llMask = adaptor.getMask();

    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }

    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> otherElems;
    if (op.getOther())
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);

    auto maybeSwizzledEnc =
        dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(dstEnc);
    bool hasSwizzling = maybeSwizzledEnc && maybeSwizzledEnc.getMaxPhase() != 1;
    assert(!hasSwizzling && "Swizzling not supported on CPU");

    Type srcPtrTy = srcElems[0].getType();
    bool hasOther = !otherElems.empty();
    Type otherTy = hasOther ? otherElems[0].getType() : i1_ty;

    SmallVector<Value> swizzledLaneOffsets;
    SmallVector<Value> loadVals =
        zipLoadValues(rewriter, loc, vec, srcElems, srcPtrTy, maskElements,
                      otherElems, otherTy, swizzledLaneOffsets);
    Value threadPred = emitRedundantThreadPredicate(getFreeVariableMasks(srcTy),
                                                    rewriter, loc, targetInfo);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    llvm::errs() << "srcLayout = " << srcLayout << "\n";
    llvm::errs() << "removeBroadcastSrc = " << removeBroadcastSrc << "\n";

    return failure();
#endif
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // same deal, copy is synchronous
    // auto num = op->getAttrOfType<IntegerAttr>("num");
    // rewriter.create<NVVM::CpAsyncWaitGroupOp>(loc, num);

    // Drop the result token.
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // our load/store isn't actually async so we probably don't care
    // rewriter.create<NVVM::CpAsyncCommitGroupOp>(loc);

    // Drop the result token.
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

} // namespace

void mlir::triton::npu::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion,
               AsyncCopyGlobalToLocalOpConversion>(typeConverter, targetInfo,
                                                   axisInfoAnalysis, benefit);
  patterns.add<AsyncCommitGroupOpConversion, AsyncWaitOpConversion>(
      typeConverter, benefit);
}
