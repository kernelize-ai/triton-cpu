#include "TargetInfo.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "PatternTritonGPUOpToLLVM.h"
#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

namespace {

struct GenericOpConversion : public ConvertOpToLLVMPattern<cpu::GenericOp> {
  using ConvertOpToLLVMPattern<cpu::GenericOp>::ConvertOpToLLVMPattern;

  // NOTE: Tensors materialized by generic ops and used by other generic ops
  // are lowered to alloca'd arrays to avoid loop-carried vector phi nodes
  // (which would overflow the stack for large block sizes). The alloca is
  // defined once before the tile loop and elements are written/read via GEP.
  // We need to look through any unrealized conversion cast ops inserted by the
  // MLIR Conversion framework to find the underlying alloca pointer.
  Value getGenericOutputTensorAsPtr(cpu::GenericOp op, unsigned opIdx,
                                    Value llvmArg) const {
    // find the generic op producing this tensor
    auto result = dyn_cast<OpResult>(op.getIns()[opIdx]);
    if (!result)
      return {};
    auto defGeneric = dyn_cast<cpu::GenericOp>(result.getOwner());
    if (!defGeneric)
      return {};
    // TODO: overly defensive?
    if (result.getResultNumber() < defGeneric.getNumIterArgs())
      return {};

    assert(isa<LLVM::LLVMStructType>(llvmArg.getType()) &&
           "expected tensor value adaptor type to be a struct type");
    auto castOp = llvmArg.getDefiningOp<UnrealizedConversionCastOp>();
    assert(castOp && "expected unrealized conversion cast defining op for "
                     "generic op tensor input");
    assert(castOp.getInputs().size() == 1 &&
           isa<LLVM::LLVMPointerType>(castOp.getInputs()[0].getType()) &&
           "expected materialized tensor from generic op to have LLVM pointer "
           "type (alloca)");
    return castOp.getInputs()[0];
  }

  SmallVector<Value> buildStaticChunkedArgs(cpu::GenericOp op,
                                            OpAdaptor adaptor,
                                            ConversionPatternRewriter &rewriter,
                                            unsigned i,
                                            unsigned vectorSize) const {
    Location loc = op.getLoc();
    Block *body = &op.getBody().front();
    SmallVector<Value> chunkedArgs;

    for (auto [opIdx, origArg, llvmArg] :
         llvm::enumerate(body->getArguments().drop_front(op.getInsArgOffset()),
                         adaptor.getIns())) {
      LDBG("Chunk operand " << opIdx << " = " << origArg << " --> " << llvmArg);
      if (!isa<RankedTensorType>(origArg.getType())) {
        // forward constants and scalars without chunking
        assert(isa<PointerType>(origArg.getType()) ||
               origArg.getType() == llvmArg.getType() &&
                   "expected non-tensor arguments to be unchanged by type "
                   "conversion");
        chunkedArgs.push_back(op.getIns()[opIdx]);
      } else if (Value ptrArg =
                     getGenericOutputTensorAsPtr(op, opIdx, llvmArg)) {
        // Load this tile's elements from the alloca produced by the prior
        // generic, then pack them into the tile struct expected by the body.
        Type tileStructTy = getTypeConverter()->convertType(origArg.getType());
        auto structTy = cast<LLVM::LLVMStructType>(tileStructTy);
        Type elemTy = structTy.getBody()[0];
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        Value tileStruct = LLVM::UndefOp::create(rewriter, loc, tileStructTy);
        for (unsigned j = 0; j < vectorSize; ++j) {
          Value idx = b.i32_val(i * vectorSize + j);
          Value gep = LLVM::GEPOp::create(
              rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
              elemTy, ptrArg, ValueRange{idx});
          Value elem = LLVM::LoadOp::create(rewriter, loc, elemTy, gep);
          tileStruct =
              LLVM::InsertValueOp::create(rewriter, loc, tileStruct, elem, {j});
        }
        chunkedArgs.push_back(UnrealizedConversionCastOp::create(
                                  rewriter, loc, origArg.getType(), tileStruct)
                                  .getResult(0));
      } else {
        Type convertedBodyType =
            getTypeConverter()->convertType(origArg.getType());

        Value chunk = LLVM::UndefOp::create(rewriter, loc, convertedBodyType);

        for (unsigned j = 0; j < vectorSize; ++j) {
          int64_t srcIndex = i * vectorSize + j;
          LDBG("Read " << srcIndex);
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

  // Returns {newIterArgVals, tensorTiles} from the tile's ttc.yield.
  // iterArgVals provides the current accumulator values for iter args.
  std::pair<SmallVector<Value>, SmallVector<Value>>
  cloneTileBody(cpu::GenericOp op, ConversionPatternRewriter &rewriter,
                ArrayRef<Value> iterArgVals, ArrayRef<Value> chunkedArgs,
                ArrayRef<Value> tileOffsets) const {
    assert(op.getBody().getBlocks().size() == 1 &&
           "expected generic op body to have one block");
    Block *body = &op.getBody().front();

    IRMapping mapping;
    // IVs
    for (auto [blockArg, offset] : llvm::zip(
             body->getArguments().take_front(tileOffsets.size()), tileOffsets))
      mapping.map(blockArg, offset);
    // iter args: positions numIVs..numIVs+numIterArgs-1
    for (auto [i, iterArgVal] : llvm::enumerate(iterArgVals))
      mapping.map(body->getArgument(op.getNumInductionVars() + i), iterArgVal);
    // ins args: positions insArgOffset..
    for (auto [bodyArg, chunkedArg] :
         llvm::zip(body->getArguments().drop_front(op.getInsArgOffset()),
                   chunkedArgs))
      mapping.map(bodyArg, chunkedArg);

    unsigned numIterArgs = op.getNumIterArgs();
    SmallVector<Value> newIterArgVals, tensorTiles;

    for (Operation &bOp : *body) {
      if (auto yieldOp = dyn_cast<cpu::YieldOp>(bOp)) {
        if (yieldOp.getValues().empty())
          continue;
        auto yieldVals = llvm::to_vector(llvm::map_range(
            yieldOp.getValues(), [&](Value v) { return mapping.lookup(v); }));
        newIterArgVals.assign(yieldVals.begin(),
                              yieldVals.begin() + numIterArgs);
        tensorTiles.assign(yieldVals.begin() + numIterArgs, yieldVals.end());
      } else {
        rewriter.clone(bOp, mapping);
      }
    }

    return {newIterArgVals, tensorTiles};
  }

  void scatterTiles(ConversionPatternRewriter &rewriter, Location loc,
                    ArrayRef<Value> tiles, Value tileOffset,
                    unsigned vectorSize, ArrayRef<Value> tensorAccPtrs,
                    ArrayRef<Type> tensorElemTys) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (auto [tile, accPtr, elemTy] :
         llvm::zip(tiles, tensorAccPtrs, tensorElemTys)) {
      Type tileStructTy = getTypeConverter()->convertType(tile.getType());
      Value llvmTile =
          UnrealizedConversionCastOp::create(rewriter, loc, tileStructTy, tile)
              .getResult(0);
      for (unsigned j = 0; j < vectorSize; ++j) {
        Value elem = LLVM::ExtractValueOp::create(rewriter, loc, llvmTile, {j});
        Value idx =
            LLVM::AddOp::create(rewriter, loc, tileOffset, b.i32_val(j));
        Value gep = LLVM::GEPOp::create(
            rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
            elemTy, accPtr, ValueRange{idx});
        LLVM::StoreOp::create(rewriter, loc, elem, gep);
      }
    }
  }

  // Builds tile args for a dynamic tile at runtime offset tileOffset.
  // Mirrors buildStaticChunkedArgs but uses a runtime Value for the offset
  // instead of a compile-time index. Only valid for temporary storage backed
  // tensor args and non-tensor args — statically-indexed tensor args must use
  // the unrolled path.
  SmallVector<Value>
  buildDynamicChunkedArgs(cpu::GenericOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter, Value tileOffset,
                          unsigned vectorSize) const {
    Location loc = op.getLoc();
    Block *body = &op.getBody().front();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> tileArgs;

    for (auto [opIdx, origArg, llvmArg] :
         llvm::enumerate(body->getArguments().drop_front(op.getInsArgOffset()),
                         adaptor.getIns())) {
      if (Value ptrArg = getGenericOutputTensorAsPtr(op, opIdx, llvmArg)) {
        // Load this tile's elements from the alloca produced by the prior
        // generic and pack them into the tile struct.
        Type tileStructTy = getTypeConverter()->convertType(origArg.getType());
        auto structTy = cast<LLVM::LLVMStructType>(tileStructTy);
        Type elemTy = structTy.getBody()[0];
        Value tileStruct = LLVM::UndefOp::create(rewriter, loc, tileStructTy);
        for (unsigned j = 0; j < vectorSize; ++j) {
          Value globalIdx =
              LLVM::AddOp::create(rewriter, loc, tileOffset, b.i32_val(j));
          Value gep = LLVM::GEPOp::create(
              rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
              elemTy, ptrArg, ValueRange{globalIdx});
          Value elem = LLVM::LoadOp::create(rewriter, loc, elemTy, gep);
          tileStruct =
              LLVM::InsertValueOp::create(rewriter, loc, tileStruct, elem, {j});
        }
        tileArgs.push_back(tileStruct);
      } else if (isa<PointerType>(origArg.getType())) {
        // we might have a tt.ptr hiding in an unrealized conversion cast due to
        // late conversion of generic op block arguments. Use unrealized
        // conversion cast which will be folded away later
        if (origArg.getType() != llvmArg.getType()) {
          tileArgs.push_back(
              UnrealizedConversionCastOp::create(
                  rewriter, loc, llvmArg.getType(), op.getIns()[opIdx])
                  .getResult(0));
        } else {
          tileArgs.push_back(op.getIns()[opIdx]);
        }
      } else {
        assert(!isa<RankedTensorType>(origArg.getType()) &&
               "tensor types are not allowed in compile-time generated "
               "generic tile loops");
        assert(origArg.getType() == llvmArg.getType() &&
               "expected non-tensor arguments to be unchanged by type "
               "conversion");
        tileArgs.push_back(op.getIns()[opIdx]);
      }
    }
    return tileArgs;
  }

  // statically unroll all tiles. Required when any tensor input uses
  // llvm.extractvalue (which requires compile-time indices), or when there is
  // only one tile and loop overhead is unnecessary.
  // iterArgVals is updated in-place to the final accumulated values.
  void emitUnrolled(cpu::GenericOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, unsigned numChunks,
                    unsigned vectorSize, ArrayRef<int32_t> blockShape,
                    ArrayRef<int32_t> tileShape,
                    SmallVectorImpl<Value> &iterArgVals,
                    ArrayRef<Value> tensorAccPtrs,
                    ArrayRef<Type> tensorElemTys) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    LDBG("Unrolling generic, cloning body for " << numChunks
                                                << " loop iterations");

    for (unsigned i = 0; i < numChunks; ++i) {
      SmallVector<Value> chunkedArgs =
          buildStaticChunkedArgs(op, adaptor, rewriter, i, vectorSize);

      unsigned rank = blockShape.size();
      SmallVector<Value> perDimOffsets(rank);
      unsigned remaining = i;
      LDBG("i = " << i);
      for (int d = rank - 1; d >= 0; --d) {
        unsigned nc = blockShape[d] / tileShape[d];
        unsigned chunkIdx = remaining % nc;
        LDBG("perDimOffsets[" << d << "] = " << chunkIdx << " * "
                              << tileShape[d] << " = "
                              << (chunkIdx * tileShape[d]));
        perDimOffsets[d] = b.i32_val(chunkIdx * tileShape[d]);
        remaining /= nc;
      }

      Value flatOffset = b.i32_val(i * vectorSize);
      LDBG("flatOffset = " << (i * vectorSize));
      auto [newIterArgVals, tiles] =
          cloneTileBody(op, rewriter, iterArgVals, chunkedArgs, perDimOffsets);
      iterArgVals.assign(newIterArgVals.begin(), newIterArgVals.end());
      scatterTiles(rewriter, loc, tiles, flatOffset, vectorSize, tensorAccPtrs,
                   tensorElemTys);
    }
  }

  // Emits a loop over numChunks tiles, carrying initCarried values as
  // loop-carried state alongside the loop counter. bodyFn receives the current
  // carried values and returns the updated carried values for the back edge.
  // Returns the final carried values available after the loop.
  SmallVector<Value>
  emitSingleLoop(ConversionPatternRewriter &rewriter, Location loc,
                 int32_t numChunks, int32_t tileSize,
                 ArrayRef<Value> initCarried,
                 llvm::function_ref<SmallVector<Value>(
                     Value /*loopI*/, Value /*dimTileOffset*/,
                     ArrayRef<Value> /*carried*/, Block * /*afterBlock*/)>
                     bodyFn) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // afterBlock receives the final carried values
    SmallVector<Type> carriedTys;
    for (Value v : initCarried)
      carriedTys.push_back(v.getType());
    for (Type ty : carriedTys)
      afterBlock->addArgument(ty, loc);

    // loop header: (i: i32, carried...)
    SmallVector<Type> headerTys = {i32_ty};
    headerTys.append(carriedTys);
    SmallVector<Location> headerLocs(headerTys.size(), loc);
    Block *loopHeader = rewriter.createBlock(afterBlock, headerTys, headerLocs);
    Block *loopBody = rewriter.createBlock(afterBlock);

    // currentBlock -> loopHeader(0, initCarried...)
    rewriter.setInsertionPointToEnd(currentBlock);
    SmallVector<Value> initArgs = {b.i32_val(0)};
    initArgs.append(initCarried.begin(), initCarried.end());
    LLVM::BrOp::create(rewriter, loc, initArgs, loopHeader);

    // loopHeader: check i < numChunks, branch to body or exit with carried
    rewriter.setInsertionPointToEnd(loopHeader);
    Value loopI = loopHeader->getArgument(0);
    SmallVector<Value> currentCarried;
    for (unsigned i = 0; i < carriedTys.size(); ++i)
      currentCarried.push_back(loopHeader->getArgument(1 + i));
    Value cond = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                      loopI, b.i32_val(numChunks));
    LLVM::CondBrOp::create(rewriter, loc, cond, loopBody, {}, afterBlock,
                           currentCarried);

    // loop body: emit tile, increment counter, branch back with new carried
    rewriter.setInsertionPointToEnd(loopBody);
    Value tileOffset = b.mul(loopI, b.i32_val(tileSize));
    SmallVector<Value> newCarried =
        bodyFn(loopI, tileOffset, currentCarried, afterBlock);

    Value nextI = LLVM::AddOp::create(rewriter, loc, loopI, b.i32_val(1));
    SmallVector<Value> nextArgs = {nextI};
    nextArgs.append(newCarried.begin(), newCarried.end());
    LLVM::BrOp::create(rewriter, loc, nextArgs, loopHeader);

    rewriter.setInsertionPointToStart(afterBlock);

    SmallVector<Value> finalCarried;
    for (unsigned i = 0; i < carriedTys.size(); ++i)
      finalCarried.push_back(afterBlock->getArgument(i));
    return finalCarried;
  }

  // Emits nested loops over tiles, threading iter arg values as loop-carried
  // state. iterArgVals is updated in-place to the final accumulated values.
  void emitNestedLoops(cpu::GenericOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter, unsigned dim,
                       SmallVectorImpl<Value> &accOffsets, Value flatElemOffset,
                       ArrayRef<int32_t> blockShape,
                       ArrayRef<int32_t> tileShape, unsigned vectorSize,
                       SmallVectorImpl<Value> &iterArgVals,
                       ArrayRef<Value> tensorAccPtrs,
                       ArrayRef<Type> tensorElemTys,
                       Block *innerAfterBlock = nullptr) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    unsigned rank = blockShape.size();

    if (dim == rank) {
      // innermost: inline the body region
      assert(innerAfterBlock &&
             "expected inner after block for innermost loop");

      Region &bodyRegion = op.getBody();
      Block *bodyEntry = &bodyRegion.front();

      // build tile args before inlining (references block arguments)
      auto tileArgs = buildDynamicChunkedArgs(op, adaptor, rewriter,
                                              flatElemOffset, vectorSize);

      rewriter.inlineRegionBefore(bodyRegion, *innerAfterBlock->getParent(),
                                  innerAfterBlock->getIterator());

      TypeConverter::SignatureConversion sigConv(bodyEntry->getNumArguments());
      for (auto [idx, arg] : llvm::enumerate(bodyEntry->getArguments()))
        sigConv.addInputs(idx, getTypeConverter()->convertType(arg.getType()));
      bodyEntry = rewriter.applySignatureConversion(bodyEntry, sigConv,
                                                    getTypeConverter());

      // entryArgs: [IVs..., iterArgs..., insArgs...]
      SmallVector<Value> entryArgs(accOffsets.begin(), accOffsets.end());
      for (auto arg : iterArgVals)
        LDBG("iterArgVals: " << arg);
      entryArgs.append(iterArgVals.begin(), iterArgVals.end());
      for (auto arg : tileArgs)
        LDBG("tileArgs: " << arg);
      entryArgs.append(tileArgs.begin(), tileArgs.end());
      LLVM::BrOp::create(rewriter, loc, entryArgs, bodyEntry);

      unsigned numIterArgs = op.getNumIterArgs();
      for (Block &block : llvm::make_range(bodyEntry->getIterator(),
                                           innerAfterBlock->getIterator())) {
        auto yieldOp = dyn_cast<cpu::YieldOp>(block.getTerminator());
        if (!yieldOp)
          continue;

        rewriter.setInsertionPoint(yieldOp);
        // extract new iter arg values from the leading yield operands
        iterArgVals.clear();
        for (unsigned i = 0; i < numIterArgs; ++i)
          iterArgVals.push_back(yieldOp.getValues()[i]);

        // scatter non-iter-arg tensor tiles
        SmallVector<Value> loopTiles(yieldOp.getValues().begin() + numIterArgs,
                                     yieldOp.getValues().end());
        scatterTiles(rewriter, loc, loopTiles, flatElemOffset, vectorSize,
                     tensorAccPtrs, tensorElemTys);

        rewriter.eraseOp(yieldOp);
        rewriter.setInsertionPointToEnd(&block);
        break;
      }

      return;
    }

    // emit single loop, recurse to next dimension
    int32_t numChunks = blockShape[dim] / tileShape[dim];

    int32_t innerStride = vectorSize;
    for (unsigned d = dim + 1; d < rank; ++d)
      innerStride *= blockShape[d] / tileShape[d];

    auto finalCarried = emitSingleLoop(
        rewriter, loc, numChunks, tileShape[dim], iterArgVals,
        [&](Value loopI, Value dimTileOffset, ArrayRef<Value> currentCarried,
            Block *afterBlock) -> SmallVector<Value> {
          accOffsets.push_back(dimTileOffset);
          Value newFlat =
              LLVM::AddOp::create(rewriter, loc, flatElemOffset,
                                  LLVM::MulOp::create(rewriter, loc, loopI,
                                                      b.i32_val(innerStride)));
          SmallVector<Value> innerIterArgVals(currentCarried.begin(),
                                              currentCarried.end());
          emitNestedLoops(op, adaptor, rewriter, dim + 1, accOffsets, newFlat,
                          blockShape, tileShape, vectorSize, innerIterArgVals,
                          tensorAccPtrs, tensorElemTys, afterBlock);
          accOffsets.pop_back();
          return innerIterArgVals;
        });

    iterArgVals.assign(finalCarried.begin(), finalCarried.end());
  }

  LogicalResult
  matchAndRewrite(cpu::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto tileShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("tileShape");
    assert(tileShapeAttr && "expected generic op to have tileShape attribute");
    ArrayRef<int32_t> tileShape = tileShapeAttr.asArrayRef();

    SmallVector<int32_t> blockShape;
    for (Value dim : op.getBlockShape()) {
      APInt val;
      assert(matchPattern(dim, m_ConstantInt(&val)) &&
             "expected blockShape operand to fold to a constant integer");
      blockShape.push_back(static_cast<int32_t>(val.getSExtValue()));
    }

    assert(!blockShape.empty() && blockShape.size() == tileShape.size() &&
           "blockShape and tileShape must be non-empty and of the same size");

    unsigned vectorSize = 1, numChunks = 1;
    for (unsigned d = 0; d < blockShape.size(); ++d) {
      vectorSize *= tileShape[d];
      numChunks *= blockShape[d] / tileShape[d];
    }

    LDBG("Lowering genericOp with vectorSize = "
         << vectorSize << ", numChunks = " << numChunks);

    // Tensor results are materialized as thread-local global arrays rather than
    // LLVM vectors. This avoids loop-carried <blockSize x elemTy> phi nodes
    // which would allocate tens of kilobytes on the stack and cause stack
    // overflows for large block sizes (e.g. blockSize=4096).
    //
    // Globals are addressed via AddressOf in the function entry block so the
    // pointer dominates all uses. Each thread gets its own copy since the
    // globals are thread-local.
    //
    // TODO: we should probably check that generic results are only used by
    // other generics or we will run into conversion problems
    SmallVector<Value> tensorAccPtrs;
    SmallVector<Type> tensorElemTys;
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(func && "expected generic op to be inside an LLVM function");
    auto module = op->getParentOfType<ModuleOp>();
    assert(module && "expected generic op to be inside a module");
    for (Type resultTy :
         llvm::drop_begin(op.getResultTypes(), op.getNumIterArgs())) {
      auto tensorTy = cast<RankedTensorType>(resultTy);
      int64_t blockSize = std::accumulate(blockShape.begin(), blockShape.end(),
                                          int64_t{1}, std::multiplies<>());
      Type elemTy = getTypeConverter()->convertType(tensorTy.getElementType());
      tensorElemTys.push_back(elemTy);

      auto globalArrayTy = LLVM::LLVMArrayType::get(elemTy, blockSize);
      std::string globalName;
      unsigned nameIdx = tensorAccPtrs.size();
      do {
        globalName =
            (func.getName() + "_tac_" + std::to_string(nameIdx++)).str();
      } while (module.lookupSymbol(globalName));
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        LLVM::GlobalOp::create(rewriter, loc, globalArrayTy,
                               /*isConstant=*/false, LLVM::Linkage::Internal,
                               globalName, Attribute{},
                               /*alignment=*/4, /*addrSpace=*/0,
                               /*dsoLocal=*/false, /*threadLocal=*/true);
      }
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&func.getBody().front());
      LDBG("Creating thread local global allocation `"
           << globalName << "` for tensor of size " << blockSize);
      Value globalPtr = LLVM::AddressOfOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          globalName);
      tensorAccPtrs.push_back(globalPtr);
    }

    Block *body = &op.getBody().front();

    // Tensor args passed as LLVM structs require static tile indices
    // (llvm.extractvalue only accepts attribute indices). Alloca-backed tensors
    // from a prior generic are GEP-indexed and support dynamic offsets.
    const bool requiresTensorArgMaterialization =
        llvm::any_of(llvm::enumerate(llvm::zip(
                         body->getArguments().drop_front(op.getInsArgOffset()),
                         adaptor.getIns())),
                     [this, &op](auto pair) {
                       auto [opIdx, argPair] = pair;
                       auto [origArg, llvmArg] = argPair;
                       if (!isa<RankedTensorType>(origArg.getType()))
                         return false;
                       if (getGenericOutputTensorAsPtr(op, opIdx, llvmArg))
                         return false;
                       return true;
                     });

    // Iter arg values start from init_vals and are updated tile-by-tile.
    SmallVector<Value> iterArgVals(adaptor.getInitVals().begin(),
                                   adaptor.getInitVals().end());

    if (requiresTensorArgMaterialization || numChunks == 1) {
      emitUnrolled(op, adaptor, rewriter, numChunks, vectorSize, blockShape,
                   tileShape, iterArgVals, tensorAccPtrs, tensorElemTys);
    } else {
      const bool genericHasTensorOutputs =
          llvm::any_of(op->getResults(), [](Value result) {
            return isa<RankedTensorType>(result.getType());
          });
      const bool allUsersAreGeneric =
          genericHasTensorOutputs
              ? llvm::all_of(
                    op->getUsers(),
                    [](Operation *user) { return isa<cpu::GenericOp>(user); })
              : true;
      assert(allUsersAreGeneric && "expected all generic op users to also be "
                                   "generic ops on dynamic path");
      SmallVector<Value> accOffsets;
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      emitNestedLoops(op, adaptor, rewriter, /*dim=*/0, accOffsets,
                      b.i32_val(0), blockShape, tileShape, vectorSize,
                      iterArgVals, tensorAccPtrs, tensorElemTys);
    }

    // Collect all replacement values atomically. Mixing replaceAllUsesWith +
    // eraseOp in a ConversionPatternRewriter causes double-replacement
    // assertions in the framework; a single replaceOp call avoids that.
    unsigned numIterArgs = op.getNumIterArgs();
    SmallVector<Value> allReplacements(iterArgVals.begin(), iterArgVals.end());
    allReplacements.append(tensorAccPtrs);
    if (allReplacements.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, allReplacements);
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateGenericOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GenericOpConversion>(typeConverter, benefit);
}
