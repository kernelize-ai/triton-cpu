#include "TargetInfo.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "PatternTritonGPUOpToLLVM.h"
#include <functional>

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
    auto result = dyn_cast<OpResult>(op.getOperand(opIdx));
    if (!result)
      return {};
    auto defGeneric = dyn_cast<cpu::GenericOp>(result.getOwner());
    if (!defGeneric)
      return {};
    // TODO: overly defensive?
    if (result.getResultNumber() < defGeneric.getCombiners().getBlocks().size())
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

    for (auto [opIdx, origArg, llvmArg] : llvm::enumerate(
             body->getArguments().drop_front(op.getNumInductionVars()),
             adaptor.getOperands())) {

      if (!isa<RankedTensorType>(origArg.getType())) {
        // forward constants and scalars without chunking
        assert(isa<PointerType>(origArg.getType()) ||
               origArg.getType() == llvmArg.getType() &&
                   "expected non-tensor arguments to be unchanged by type "
                   "conversion");
        chunkedArgs.push_back(op.getOperand(opIdx));
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

  SmallVector<Value> emitTileBody(cpu::GenericOp op,
                                  ConversionPatternRewriter &rewriter,
                                  ArrayRef<Value> chunkedArgs,
                                  ArrayRef<Value> tileOffsets,
                                  Value &result) const {
    Block *body = &op.getBody().front();
    const bool hasReductions = !op.getCombiners().empty();

    // clone the body of the generic op for this chunk only
    IRMapping mapping;
    for (auto [blockArg, offset] : llvm::zip(
             body->getArguments().take_front(tileOffsets.size()), tileOffsets))
      mapping.map(blockArg, offset);
    for (auto [bodyArg, chunkedArg] :
         llvm::zip(body->getArguments().drop_front(op.getNumInductionVars()),
                   chunkedArgs))
      mapping.map(bodyArg, chunkedArg);

    SmallVector<Value> tensorTiles;

    for (Operation &bOp : *body) {
      if (auto yieldOp = dyn_cast<cpu::YieldOp>(bOp)) {
        if (yieldOp.getValues().empty())
          continue;

        auto yieldOpValues = llvm::to_vector(llvm::map_range(
            yieldOp.getValues(), [&](Value v) { return mapping.lookup(v); }));

        if (hasReductions) {
          if (!result) {
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

          // materialzied tensor values are currently added to yield op after
          // scalars
          unsigned numCombinerBlocks = op.getCombiners().getBlocks().size();
          for (unsigned k = numCombinerBlocks; k < yieldOpValues.size(); ++k)
            tensorTiles.push_back(yieldOpValues[k]);
        } else {
#if 1
          for (unsigned k = 0; k < yieldOpValues.size(); ++k)
            tensorTiles.push_back(yieldOpValues[k]);
#else
          // TODO: we should assert that all users are generics since we are
          // about to materialize a tensor in a format only generics can handle
          assert(yieldOpValues.size() == 1 &&
                 "only support scattering one generic tensor result currently");
          assert(op.getBlockShape() == op.getTileShape() &&
                 "only support materializing tensors in static path for "
                 "generics with num_tiles = 1");
          Location loc = yieldOp.getLoc();
          auto b = TritonLLVMOpBuilder(loc, rewriter);
          scatterTiles(rewriter, loc, yieldOpValues,
                       /*tileOffset= */ b.i32_val(0), /*vectorSize= */ 1, {},
                       {});
#endif
        }

      } else {
        rewriter.clone(bOp, mapping);
      }
    }

    return tensorTiles;
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

    for (auto [opIdx, origArg, llvmArg] : llvm::enumerate(
             body->getArguments().drop_front(op.getNumInductionVars()),
             adaptor.getOperands())) {
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
        tileArgs.push_back(UnrealizedConversionCastOp::create(
                               rewriter, loc, origArg.getType(), tileStruct)
                               .getResult(0));
      } else if (isa<PointerType>(origArg.getType())) {
        // we might have a tt.ptr hiding in an unrealized conversion cast due to
        // late conversion of generic op block arguments. Use unrealized
        // conversion cast which will be folded away later
        if (origArg.getType() != llvmArg.getType()) {
          tileArgs.push_back(
              UnrealizedConversionCastOp::create(
                  rewriter, loc, llvmArg.getType(), op.getOperand(opIdx))
                  .getResult(0));
        } else {
          tileArgs.push_back(op.getOperand(opIdx));
        }
      } else {
        assert(!isa<RankedTensorType>(origArg.getType()) &&
               "tensor types are not allowed in compile-time generated "
               "generic tile loops");
        assert(origArg.getType() == llvmArg.getType() &&
               "expected non-tensor arguments to be unchanged by type "
               "conversion");
        tileArgs.push_back(op.getOperand(opIdx));
      }
    }
    return tileArgs;
  }

  // statically unroll all tiles. Required when any tensor input uses
  // llvm.extractvalue (which requires compile-time indices), or when there is
  // only one tile and loop overhead is unnecessary.
  void emitUnrolled(cpu::GenericOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, unsigned numChunks,
                    unsigned vectorSize, ArrayRef<int32_t> blockShape,
                    ArrayRef<int32_t> tileShape, Value &result,
                    ArrayRef<Value> tensorAccPtrs,
                    ArrayRef<Type> tensorElemTys) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned i = 0; i < numChunks; ++i) {
      SmallVector<Value> chunkedArgs =
          buildStaticChunkedArgs(op, adaptor, rewriter, i, vectorSize);

      unsigned rank = blockShape.size();
      SmallVector<Value> perDimOffsets(rank);
      unsigned remaining = i;
      for (int d = rank - 1; d >= 0; --d) {
        unsigned nc = blockShape[d] / tileShape[d];
        unsigned chunkIdx = remaining % nc;
        perDimOffsets[d] = b.i32_val(chunkIdx * tileShape[d]);
        remaining /= nc;
      }

      Value flatOffset = b.i32_val(i * vectorSize);
      auto tiles =
          emitTileBody(op, rewriter, chunkedArgs, perDimOffsets, result);
      scatterTiles(rewriter, loc, tiles, flatOffset, vectorSize, tensorAccPtrs,
                   tensorElemTys);
    }
  }

  // emit a loop over tiles carrying a reduction accumulator.
  // The first tile is peeled to establish the initial accumulator value, then
  // tiles 1..numChunks-1 are processed in a loop with (i, acc) as loop-carried
  // values. The final accumulator is forwarded through the after block.
  void emitLoopWithReductions(cpu::GenericOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              unsigned numChunks, unsigned vectorSize,
                              Value &result, ArrayRef<Value> tensorAccPtrs,
                              ArrayRef<Type> tensorElemTys) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // peel tile 0 to establish the initial reduction accumulator
    auto firstArgs =
        buildStaticChunkedArgs(op, adaptor, rewriter, 0, vectorSize);
    auto firstTiles =
        emitTileBody(op, rewriter, firstArgs, {b.i32_val(0)}, result);
    scatterTiles(rewriter, loc, firstTiles, b.i32_val(0), vectorSize,
                 tensorAccPtrs, tensorElemTys);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Value afterResult = afterBlock->addArgument(result.getType(), loc);

    // loop-carried values: (i: i32, acc: result type)
    Block *loopHeader = rewriter.createBlock(
        afterBlock, {i32_ty, result.getType()}, {loc, loc});
    Block *loopBody = rewriter.createBlock(afterBlock);

    // currentBlock -> loopHeader(1, initialAcc)
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{b.i32_val(1), result},
                       loopHeader);

    // loopHeader: check i < numChunks, branch to body or exit
    rewriter.setInsertionPointToEnd(loopHeader);
    Value loopI = loopHeader->getArgument(0);
    Value loopAcc = loopHeader->getArgument(1);
    Value cond = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                      loopI, b.i32_val(numChunks));
    LLVM::CondBrOp::create(rewriter, loc, cond, loopBody, {}, afterBlock,
                           ValueRange{loopAcc});

    // loop body: emit tile, update accumulator, increment counter
    rewriter.setInsertionPointToEnd(loopBody);
    Value tileOffset =
        LLVM::MulOp::create(rewriter, loc, loopI, b.i32_val(vectorSize));
    auto tileArgs =
        buildDynamicChunkedArgs(op, adaptor, rewriter, tileOffset, vectorSize);
    Value tileResult = loopAcc;
    auto loopTiles =
        emitTileBody(op, rewriter, tileArgs, {tileOffset}, tileResult);
    scatterTiles(rewriter, loc, loopTiles, tileOffset, vectorSize,
                 tensorAccPtrs, tensorElemTys);
    Value nextI = LLVM::AddOp::create(rewriter, loc, loopI, b.i32_val(1));
    LLVM::BrOp::create(rewriter, loc, ValueRange{nextI, tileResult},
                       loopHeader);

    // propagate final accumulator out of the loop
    rewriter.setInsertionPointToStart(afterBlock);
    result = afterResult;
  }

  // Path 3: emit a loop over tiles with no reduction accumulator.
  // Loops from i=0 to numChunks-1 carrying only the loop counter. No first-
  // iteration peeling is needed since there is no accumulator to bootstrap.
  void emitLoop(cpu::GenericOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, unsigned numChunks,
                unsigned vectorSize, ArrayRef<int32_t> blockShape,
                ArrayRef<int32_t> tileShape, ArrayRef<Value> tensorAccPtrs,
                ArrayRef<Type> tensorElemTys) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // loop-carried value: (i: i32) only
    Block *loopHeader = rewriter.createBlock(afterBlock, {i32_ty}, {loc});
    Block *loopBody = rewriter.createBlock(afterBlock);

    // currentBlock -> loopHeader(0)
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{b.i32_val(0)}, loopHeader);

    // loopHeader: check i < numChunks, branch to body or exit
    rewriter.setInsertionPointToEnd(loopHeader);
    Value loopI = loopHeader->getArgument(0);
    Value cond = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                      loopI, b.i32_val(numChunks));
    LLVM::CondBrOp::create(rewriter, loc, cond, loopBody, {}, afterBlock, {});

    // loop body: emit tile, increment counter
    rewriter.setInsertionPointToEnd(loopBody);
#if 1
    // Decompose flat tile index into per-dim tile offsets
    unsigned rank = blockShape.size();
    SmallVector<Value> tileOffsets(rank);
    Value remaining = loopI;
    for (int d = rank - 1; d >= 0; --d) {
      unsigned nc = blockShape[d] / tileShape[d];
      Value chunkIdx = b.urem(remaining, b.i32_val(nc));
      tileOffsets[d] = b.mul(chunkIdx, b.i32_val(tileShape[d]));
      remaining = b.udiv(remaining, b.i32_val(nc));
    }
    // Flat offset for accumulator scatter (tile-major layout)
    Value flatTileOffset = b.mul(loopI, b.i32_val(vectorSize));
#else
    Value tileOffset =
        LLVM::MulOp::create(rewriter, loc, loopI, b.i32_val(vectorSize));
#endif
    auto tileArgs = buildDynamicChunkedArgs(op, adaptor, rewriter,
                                            flatTileOffset, vectorSize);
    Value unused;
    // TODO: can we always inline here instead of clone?
#if 1

    const bool hasReductions = !op.getCombiners().empty();
    // TODO: consider asserting that generic body region has one block on the
    // emitTileBody/clone path
    assert(!hasReductions &&
           "cannot handle generic reductions on the inline loop path");

    Region &bodyRegion = op.getBody();
    Block *bodyEntry = &bodyRegion.front();

    // Move all blocks into the parent region before afterBlock.
    rewriter.inlineRegionBefore(bodyRegion, *afterBlock->getParent(),
                                afterBlock->getIterator());

    TypeConverter::SignatureConversion sigConv(bodyEntry->getNumArguments());
    for (auto [idx, arg] : llvm::enumerate(bodyEntry->getArguments()))
      sigConv.addInputs(idx, getTypeConverter()->convertType(arg.getType()));
    bodyEntry = rewriter.applySignatureConversion(bodyEntry, sigConv,
                                                  getTypeConverter());

    // Branch from loopBody into the inlined region, passing induction vars then
    // tile args as block arguments to match bodyEntry's argument list.
    rewriter.setInsertionPointToEnd(loopBody);
    SmallVector<Value> entryArgs;
    entryArgs.append(tileOffsets.begin(), tileOffsets.end());
    entryArgs.append(tileArgs.begin(), tileArgs.end());
    LLVM::BrOp::create(rewriter, loc, entryArgs, bodyEntry);

    for (Block &block : llvm::make_range(bodyEntry->getIterator(),
                                         afterBlock->getIterator())) {
      auto yieldOp = dyn_cast<cpu::YieldOp>(block.getTerminator());
      if (!yieldOp)
        continue;

      rewriter.setInsertionPoint(yieldOp);
      SmallVector<Value> loopTiles = llvm::to_vector(yieldOp.getValues());
      scatterTiles(rewriter, loc, loopTiles, flatTileOffset, vectorSize,
                   tensorAccPtrs, tensorElemTys);
      Value nextI =
          LLVM::AddOp::create(rewriter, op.getLoc(), loopI, b.i32_val(1));
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(
          yieldOp, SmallVector<Value>{nextI}, loopHeader);
      break; // only one ttc.yield per generic body
    }

#else
    auto loopTiles = emitTileBody(op, rewriter, tileArgs, tileOffsets, unused);
    scatterTiles(rewriter, loc, loopTiles, flatTileOffset, vectorSize,
                 tensorAccPtrs, tensorElemTys);

    Value nextI = LLVM::AddOp::create(rewriter, loc, loopI, b.i32_val(1));
    LLVM::BrOp::create(rewriter, loc, ValueRange{nextI}, loopHeader);
#endif
  }

  LogicalResult
  matchAndRewrite(cpu::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto blockShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("blockShape");
    assert(blockShapeAttr &&
           "expected generic op to have blockShape attribute");
    auto tileShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("tileShape");
    assert(tileShapeAttr && "expected generic op to have tileShape attribute");

    ArrayRef<int32_t> blockShape = blockShapeAttr.asArrayRef();
    ArrayRef<int32_t> tileShape = tileShapeAttr.asArrayRef();
    assert(blockShape.size() == tileShape.size() && !blockShape.empty() &&
           "blockShape and tileShape must be non-empty and of the same size");

    unsigned vectorSize = 1, numChunks = 1;
    for (unsigned d = 0; d < blockShape.size(); ++d) {
      vectorSize *= tileShape[d];
      numChunks *= blockShape[d] / tileShape[d];
    }

    const bool hasReductions = !op.getCombiners().empty();
    const unsigned numCombinerBlocks = op.getCombiners().getBlocks().size();

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
         llvm::drop_begin(op.getResultTypes(), numCombinerBlocks)) {
      auto tensorTy = cast<RankedTensorType>(resultTy);
      int64_t blockSize = std::accumulate(blockShape.begin(), blockShape.end(),
                                          0, std::multiplies<>());
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
      Value globalPtr = LLVM::AddressOfOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          globalName);
      tensorAccPtrs.push_back(globalPtr);
    }

    Block *body = &op.getBody().front();

    // Tensor args passed as LLVM structs require static tile indices
    // (llvm.extractvalue only accepts attribute indices). Alloca-backed tensors
    // from a prior generic are GEP-indexed and support dynamic offsets.
    const bool requiresTensorArgMaterialization = llvm::any_of(
        llvm::enumerate(
            llvm::zip(body->getArguments().drop_front(op.getNumInductionVars()),
                      adaptor.getOperands())),
        [this, &op](auto pair) {
          auto [opIdx, argPair] = pair;
          auto [origArg, llvmArg] = argPair;
          if (!isa<RankedTensorType>(origArg.getType()))
            return false;
          if (getGenericOutputTensorAsPtr(op, opIdx, llvmArg))
            return false;
          return true;
        });

    Value result;
    if (requiresTensorArgMaterialization || numChunks == 1) {
      emitUnrolled(op, adaptor, rewriter, numChunks, vectorSize, blockShape,
                   tileShape, result, tensorAccPtrs, tensorElemTys);
    } else if (hasReductions) {
      assert(blockShape.size() == 1 &&
             "only support rank-1 generic tiles in reductions path");
      emitLoopWithReductions(op, adaptor, rewriter, numChunks, vectorSize,
                             result, tensorAccPtrs, tensorElemTys);
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
      emitLoop(op, adaptor, rewriter, numChunks, vectorSize, blockShape,
               tileShape, tensorAccPtrs, tensorElemTys);
    }

    SmallVector<Value> replacements;
    if (result)
      replacements.push_back(result);
    replacements.append(tensorAccPtrs);
    if (!replacements.empty())
      rewriter.replaceOp(op, replacements);
    else
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
