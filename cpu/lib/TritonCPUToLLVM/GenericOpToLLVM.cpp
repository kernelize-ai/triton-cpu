#include "TargetInfo.h"
#include <numeric>

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct GenericOpConversion : public ConvertOpToLLVMPattern<cpu::GenericOp> {
  using ConvertOpToLLVMPattern<cpu::GenericOp>::ConvertOpToLLVMPattern;

  GenericOpConversion(LLVMTypeConverter &converter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<cpu::GenericOp>(converter, benefit),
        targetInfo(targetInfo) {}

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

  Block *inlineTileBody(
      cpu::GenericOp op, ConversionPatternRewriter &rewriter, Value tileOffset,
      Block *afterBlock, Block *loopHeader, Value loopI, Value one,
      const LLVMTypeConverter *typeConverter,
      function_ref<void(ArrayRef<Value>, Value)> scatterTiles) const {
    assert(
        op.getCombiners().empty() &&
        "reductions not supported when inlining generic body into loop nest");

    Region &bodyRegion = op.getBody();
    Block *bodyEntry = &bodyRegion.front();

    // Move all blocks into the parent region before afterBlock.
    rewriter.inlineRegionBefore(bodyRegion, *afterBlock->getParent(),
                                afterBlock->getIterator());

    // Find the ttc.yield in the moved blocks and replace it with
    // scatter + back-branch. Iterate only the moved blocks
    // (bodyEntry..afterBlock) to avoid touching any nested generic's yield.
    for (Block &block : llvm::make_range(bodyEntry->getIterator(),
                                         afterBlock->getIterator())) {
      auto yieldOp = dyn_cast<cpu::YieldOp>(block.getTerminator());
      if (!yieldOp)
        continue;

      rewriter.setInsertionPoint(yieldOp);
      SmallVector<Value> tiles = llvm::to_vector(yieldOp.getValues());
      scatterTiles(tiles, tileOffset);
      Value nextI = LLVM::AddOp::create(rewriter, op.getLoc(), loopI, one);
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(
          yieldOp, SmallVector<Value>{nextI}, loopHeader);
      break; // only one ttc.yield per generic body
    }

    return bodyEntry;
  }

  SmallVector<Value> cloneTileBody(cpu::GenericOp op,
                                   ConversionPatternRewriter &rewriter,
                                   ArrayRef<Value> chunkedArgs,
                                   Value tileOffset, Value &result) const {
    Block *body = &op.getBody().front();
    const bool hasReductions = !op.getCombiners().empty();

    assert(op.getVectorShape().size() == 1 &&
           "clone tile body not yet supported for 2D generics");

    // clone the body of the generic op for this chunk only
    IRMapping mapping;
    mapping.map(body->getArgument(0), tileOffset);
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
        // TODO: this is messy. we should really differentiate the reductions
        // path
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
        }

        // materialzied tensor values are currently added to yield op after
        // scalars
        unsigned numCombinerBlocks = op.getCombiners().getBlocks().size();
        for (unsigned k = numCombinerBlocks; k < yieldOpValues.size(); ++k)
          tensorTiles.push_back(yieldOpValues[k]);
      } else {
        rewriter.clone(bOp, mapping);
      }
    }

    return tensorTiles;
  }

  LogicalResult
  matchAndRewrite(cpu::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    auto blockShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("blockShape");
    auto vectorShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("vectorShape");

    ArrayRef<int32_t> blockShape = blockShapeAttr.asArrayRef();
    ArrayRef<int32_t> vectorShape = vectorShapeAttr.asArrayRef();
    assert(blockShape.size() == vectorShape.size() && !blockShape.empty() &&
           "blockShape and vectorShape must be non-empty and of the same size");

    int64_t blockSize = std::accumulate(blockShape.begin(), blockShape.end(),
                                        int64_t{1}, std::multiplies<int64_t>{});
    int64_t vectorSize =
        std::accumulate(vectorShape.begin(), vectorShape.end(), int64_t{1},
                        std::multiplies<int64_t>{});
    unsigned numChunks = blockSize / vectorSize;

    Value result;
    const bool hasReductions = !op.getCombiners().empty();

    const unsigned numCombinerBlocks = op.getCombiners().getBlocks().size();

    // Tensor results are materialized as alloca'd arrays rather than LLVM
    // vectors. This avoids loop-carried <blockSize x elemTy> phi nodes which
    // would allocate tens of kilobytes on the stack and cause stack overflows
    // for large block sizes (e.g. blockSize=4096).
    //
    // Allocas are hoisted to the function entry block so they are allocated
    // once per kernel invocation regardless of how many times the enclosing
    // block loop iterates. Allocas inside loops would grow the stack on every
    // iteration without being freed, causing a bus error.
    //
    // TODO: we should probably check that generic results are only used by
    // other generics or we will run into conversion problems
    SmallVector<Value> tensorAccPtrs;
    SmallVector<RankedTensorType> tensorAccTys;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(func && "expected generic op to be inside an LLVM function");
    auto module = op->getParentOfType<ModuleOp>();
    assert(module && "expected generic op to be inside a module");
    for (Type resultTy :
         llvm::drop_begin(op.getResultTypes(), numCombinerBlocks)) {
      auto tensorTy = cast<RankedTensorType>(resultTy);
      tensorAccTys.push_back(tensorTy);

      // Use a thread-local global rather than an alloca to hold the tile
      // cache. Allocas of this size (blockSize * sizeof(elem)) overflow the
      // stack when the kernel runs on threads with limited stack space.
      // Thread-local globals are allocated once per thread, reused across
      // kernel invocations, and are safe for parallel execution since each
      // thread gets its own copy.
      Type elemTy = getTypeConverter()->convertType(tensorTy.getElementType());
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
      // Emit address-of in the function entry block so it dominates all uses.
      // The global pointer is invariant — no need to recompute it per tile.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&func.getBody().front());
      Value globalPtr = LLVM::AddressOfOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          globalName);
      tensorAccPtrs.push_back(globalPtr);
    }

    Block *body = &op.getBody().front();

    const bool requiresTensorArgMaterialization = llvm::any_of(
        llvm::enumerate(
            llvm::zip(body->getArguments().drop_front(op.getNumInductionVars()),
                      adaptor.getOperands())),
        [this, &op](auto pair) {
          auto [opIdx, argPair] = pair;
          auto [origArg, llvmArg] = argPair;
          if (!isa<RankedTensorType>(origArg.getType()))
            return false;
          // Alloca-backed args (materialized by a prior generic) support
          // dynamic indices via GEP — no static unrolling needed for them.
          if (getGenericOutputTensorAsPtr(op, opIdx, llvmArg))
            return false;
          return true;
        });

    // TODO: cleanup this conditional soup
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

    // Store each tile's results into the alloca'd arrays. tileOffset is the
    // starting global element index for this tile (i * vectorSize).
    // TODO: a few things.
    // (1) we cannot flatten the vector size here, really. if we do flatten it
    // then we need to convert back to (x, y) coords. maybe we can do that with
    // a linear layout? (2) because we're working in x,y space we want the
    // global tensor register index for our (x,y) coord. so we need to invert
    // the global linear layout, give it our (x,y) coord, and get back a
    // register index for the gep
    auto scatterTiles = [&](ArrayRef<Value> tiles, Value tileOffset) {
      assert(allUsersAreGeneric &&
             "all users of generic op must be generic ops to support scattered "
             "tile materialization");
      for (auto [tile, accPtr, tensorTy] :
           llvm::zip(tiles, tensorAccPtrs, tensorAccTys)) {
        Type tileStructTy = getTypeConverter()->convertType(tile.getType());
        Value llvmTile = UnrealizedConversionCastOp::create(rewriter, loc,
                                                            tileStructTy, tile)
                             .getResult(0);

        Type elemTy =
            getTypeConverter()->convertType(tensorTy.getElementType());
        llvm::errs() << "scatter tensor ty = " << tensorTy << "\n";
        auto ll = triton::gpu::toLinearLayout(tensorTy);
        llvm::errs() << "scatter ll = " << ll << "\n";
        auto llFlat = ll.flattenOuts();

        Value zero = b.i32_val(0);
        StringAttr kBlock = str_attr("block");
        StringAttr kWarp = str_attr("warp");
        StringAttr kLane = str_attr("lane");
        StringAttr kRegister = str_attr("register");

        for (unsigned j = 0; j < (unsigned)vectorSize; ++j) {
          Value registerIdx =
              LLVM::AddOp::create(rewriter, loc, tileOffset, b.i32_val(j));
          auto offset = applyLinearLayout(loc, rewriter, llFlat,
                                          {{kRegister, registerIdx},
                                           {kLane, zero},
                                           {kWarp, zero},
                                           {kBlock, zero}});
          assert(offset.size() == 1);
          Value gep = LLVM::GEPOp::create(
              rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
              elemTy, accPtr, ValueRange{registerIdx});
          Value elem =
              LLVM::ExtractValueOp::create(rewriter, loc, llvmTile, {j});
          targetInfo.printf(rewriter, "scatter %d (%d) = %f",
                            {registerIdx, b.i32_val(j), elem});
          LLVM::StoreOp::create(rewriter, loc, elem, gep);
        }
      }
    };

    // if the generic op has tensor args we materialize the input tensors then
    // unroll the loop over tiles, slicing each tensor with the current tile
    // index. llvm.extractvalue only supports attr indices, so we need to know
    // the individual tile indices at compile time. However, a generic with no
    // tensor args can be lowered as a loop which dramatically reduces code
    // size. Note that generics without tensor args can still materialize
    // tensors within the body of the generic. If we only have 1 chunk then
    // there's no need to generate the runtime loop and we "unroll" regardless
    // of the input type
    if (requiresTensorArgMaterialization /*|| numChunks == 1*/) {
      assert(allUsersAreGeneric && "generics materializing tensors for other "
                                   "ops should not be unrolled");
      for (unsigned i = 0; i < numChunks; ++i) {

        SmallVector<Value> chunkedArgs =
            buildStaticChunkedArgs(op, adaptor, rewriter, i, vectorSize);

        Value chunkOffset = b.i32_val(i * vectorSize);

        auto tiles =
            cloneTileBody(op, rewriter, chunkedArgs, chunkOffset, result);
        scatterTiles(tiles, chunkOffset);
      }
    } else {
      Value one = b.i32_val(1);
      Value numChunksVal = b.i32_val(numChunks);

      if (hasReductions) {
        // peel the first chunk to establish an initial "result" for reductions
        auto firstArgs =
            buildStaticChunkedArgs(op, adaptor, rewriter, 0, vectorSize);
        auto firstTiles = cloneTileBody(op, rewriter, firstArgs,
                                        /*tileOffset=*/b.i32_val(0), result);
        scatterTiles(firstTiles, b.i32_val(0));
      }

      Block *currentBlock = rewriter.getInsertionBlock();
      Block *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

      // Tensor accs are alloca ptrs — they don't need to be block arguments.
      // Only the loop counter and (optionally) the reduction accumulator are
      // loop-carried values.
      Value afterResult;
      if (hasReductions)
        afterResult = afterBlock->addArgument(result.getType(), loc);

      SmallVector<Type> headerArgTypes(1, i32_ty);
      SmallVector<Location> headerArgLocs(1, loc);
      if (hasReductions) {
        headerArgTypes.push_back(result.getType());
        headerArgLocs.push_back(loc);
      }

      Block *loopHeader =
          rewriter.createBlock(afterBlock, headerArgTypes, headerArgLocs);
      Block *loopBody;
      if (hasReductions)
        loopBody = rewriter.createBlock(afterBlock);

      // currentBlock -> loopHeader(startI [, initAcc])
      rewriter.setInsertionPointToEnd(currentBlock);
      if (hasReductions) {
        LLVM::BrOp::create(rewriter, loc,
                           SmallVector<Value>{b.i32_val(1), result},
                           loopHeader);
      } else {
        LLVM::BrOp::create(rewriter, loc, SmallVector<Value>{b.i32_val(0)},
                           loopHeader);
      }

      // loopHeader: check bound, branch to body or exit
      rewriter.setInsertionPointToEnd(loopHeader);
      Value loopI = loopHeader->getArgument(0);
      Value loopAcc = hasReductions ? loopHeader->getArgument(1) : Value{};

      // TODO: we should not have both tile offsets and tile offset. we need to
      // unify this somehow, or decide that we only allow multi-dimensional
      // generic loops with tensor operand inputs.
      Value tileOffset =
          LLVM::MulOp::create(rewriter, loc, loopI, b.i32_val(vectorSize));

      Attribute genericTensorEncoding = op.getEncoding();
      SmallVector<Value> tileOffsets;
      // TODO: we need to be making the decision about the generic op body
      // encoding during tile and fuse, not here
      if (genericTensorEncoding) {
        Value zero = b.i32_val(0);
        StringAttr kBlock = str_attr("block");
        StringAttr kWarp = str_attr("warp");
        StringAttr kLane = str_attr("lane");
        StringAttr kRegister = str_attr("register");

        SmallVector<int64_t> blockShape64(op.getBlockShape().begin(),
                                          op.getBlockShape().end());
        auto ll =
            triton::gpu::toLinearLayout(blockShape64, genericTensorEncoding);
        llvm::errs() << "generic tensor encoding layout = " << ll << "\n";
        for (unsigned x = 0; x < numChunks * vectorSize; x += vectorSize) {
          auto outs =
              ll.apply({{kRegister, x}, {kWarp, 0}, {kLane, 0}, {kBlock, 0}});
          for (auto [d, v] : outs)
            llvm::errs() << d << "(" << x << ") = " << v << "\n";
        }

        auto coords = applyLinearLayout(loc, rewriter, ll,
                                        {{kRegister, tileOffset},
                                         {kLane, zero},
                                         {kWarp, zero},
                                         {kBlock, zero}});
        assert(coords.size() == op.getNumInductionVars());
        for (unsigned i = 0; i < op.getNumInductionVars(); i++) {
          tileOffsets.push_back(coords[i].second);
        }
      } else if (op.getResults().size() > 0 &&
                 isa<RankedTensorType>(op.getResult(0).getType())) {
        auto resultTenorEncoding =
            cast<RankedTensorType>(op.getResult(0).getType());
        Value zero = b.i32_val(0);
        StringAttr kBlock = str_attr("block");
        StringAttr kWarp = str_attr("warp");
        StringAttr kLane = str_attr("lane");
        StringAttr kRegister = str_attr("register");

        auto ll = triton::gpu::toLinearLayout(resultTenorEncoding);
        llvm::errs() << "generic tensor result encoding layout = " << ll
                     << "\n";
        for (unsigned x = 0; x < numChunks * vectorSize; x += vectorSize) {
          auto outs =
              ll.apply({{kRegister, x}, {kWarp, 0}, {kLane, 0}, {kBlock, 0}});
          for (auto [d, v] : outs)
            llvm::errs() << d << "(" << x << ") = " << v << "\n";
        }

        auto coords = applyLinearLayout(loc, rewriter, ll,
                                        {{kRegister, tileOffset},
                                         {kLane, zero},
                                         {kWarp, zero},
                                         {kBlock, zero}});
        assert(coords.size() == op.getNumInductionVars());
        for (unsigned i = 0; i < op.getNumInductionVars(); i++) {
          tileOffsets.push_back(coords[i].second);
        }
      } else {
        for (unsigned i = 0; i < op.getNumInductionVars(); i++) {
          tileOffsets.push_back(tileOffset);
        }
      }

      Value cond = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                        loopI, numChunksVal);

      if (hasReductions) {
        assert(loopBody);
        SmallVector<Value> exitArgs = {loopAcc};
        LLVM::CondBrOp::create(rewriter, loc, cond, loopBody, {}, afterBlock,
                               exitArgs);

        // populate the loop body
        rewriter.setInsertionPointToEnd(loopBody);
        SmallVector<Value> tileArgs;
        for (auto [opIdx, origArg, llvmArg] : llvm::enumerate(
                 body->getArguments().drop_front(op.getNumInductionVars()),
                 adaptor.getOperands())) {
          if (Value ptrArg = getGenericOutputTensorAsPtr(op, opIdx, llvmArg)) {
            // Load this tile's elements from the alloca produced by the prior
            // generic and pack them into the tile struct.
            Type tileStructTy =
                getTypeConverter()->convertType(origArg.getType());
            auto structTy = cast<LLVM::LLVMStructType>(tileStructTy);
            Type elemTy = structTy.getBody()[0];
            Value tileStruct =
                LLVM::UndefOp::create(rewriter, loc, tileStructTy);
            for (unsigned j = 0; j < vectorSize; ++j) {
              Value globalIdx =
                  LLVM::AddOp::create(rewriter, loc, tileOffset, b.i32_val(j));
              Value gep = LLVM::GEPOp::create(
                  rewriter, loc,
                  LLVM::LLVMPointerType::get(rewriter.getContext()), elemTy,
                  ptrArg, ValueRange{globalIdx});
              Value elem = LLVM::LoadOp::create(rewriter, loc, elemTy, gep);
              tileStruct = LLVM::InsertValueOp::create(rewriter, loc,
                                                       tileStruct, elem, {j});
            }
            tileArgs.push_back(UnrealizedConversionCastOp::create(
                                   rewriter, loc, origArg.getType(), tileStruct)
                                   .getResult(0));
          } else {
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
        }
        Value tileResult = loopAcc;
        auto loopTiles =
            cloneTileBody(op, rewriter, tileArgs, tileOffset, tileResult);
        scatterTiles(loopTiles, tileOffset);
        Value nextI = LLVM::AddOp::create(rewriter, loc, loopI, one);
        LLVM::BrOp::create(rewriter, loc, SmallVector<Value>{nextI, tileResult},
                           loopHeader);
      } else {
        // inline path: move the entire body region, no loopBody block needed
        Block *bodyEntry =
            inlineTileBody(op, rewriter, tileOffset, afterBlock, loopHeader,
                           loopI, one, getTypeConverter(), scatterTiles);
        // restore insertion point to loopHeader before creating the cond_br
        rewriter.setInsertionPointToEnd(loopHeader);

        // tileOffset (i32) followed by the LLVM-converted operands.
        SmallVector<Value> entryArgs = tileOffsets;
        targetInfo.printf(rewriter, "loop %d --> %d, %d",
                          {loopI, entryArgs[0], entryArgs[1]});

        for (auto [opIdx, origArg, llvmArg] : llvm::enumerate(
                 body->getArguments().drop_front(op.getNumInductionVars()),
                 adaptor.getOperands())) {
          if (Value ptrArg = getGenericOutputTensorAsPtr(op, opIdx, llvmArg)) {
            Type tileStructTy =
                getTypeConverter()->convertType(origArg.getType());
            auto structTy = cast<LLVM::LLVMStructType>(tileStructTy);
            Type elemTy = structTy.getBody()[0];
            Value tileStruct =
                LLVM::UndefOp::create(rewriter, loc, tileStructTy);

            llvm::errs() << "operand = " << op.getOperand(opIdx) << "\n";
            auto operandType =
                cast<RankedTensorType>(op.getOperand(opIdx).getType());
            llvm::errs() << "operand type = " << operandType << "\n";
            auto ll = triton::gpu::toLinearLayout(operandType);
            llvm::errs() << "ll = " << ll << "\n";
            auto llFlat = ll.flattenOuts();
            llvm::errs() << "llFlat = " << llFlat << "\n";
            Value zero = b.i32_val(0);
            StringAttr kBlock = str_attr("block");
            StringAttr kWarp = str_attr("warp");
            StringAttr kLane = str_attr("lane");
            StringAttr kRegister = str_attr("register");
            for (unsigned j = 0; j < vectorSize; ++j) {
              Value registerIdx =
                  LLVM::AddOp::create(rewriter, loc, tileOffset, b.i32_val(j));
              auto offsetPrint = llFlat.apply(
                  {{kRegister, j}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
              assert(offsetPrint.size() == 1);
              llvm::errs() << j << " = " << offsetPrint.front().second << "\n";
              auto offset = applyLinearLayout(loc, rewriter, llFlat,
                                              {{kRegister, registerIdx},
                                               {kLane, zero},
                                               {kWarp, zero},
                                               {kBlock, zero}});
              assert(offset.size() == 1);
              Value gep = LLVM::GEPOp::create(
                  rewriter, loc,
                  LLVM::LLVMPointerType::get(rewriter.getContext()), elemTy,
                  ptrArg, ValueRange{registerIdx});
              Value elem = LLVM::LoadOp::create(rewriter, loc, elemTy, gep);
              targetInfo.printf(rewriter, "gather %d (%d) = %f",
                                {registerIdx, b.i32_val(j), elem});
              tileStruct = LLVM::InsertValueOp::create(rewriter, loc,
                                                       tileStruct, elem, {j});
            }
            entryArgs.push_back(tileStruct);
          } else {
            assert(!isa<RankedTensorType>(origArg.getType()) &&
                   "tensor types are not allowed in compile-time generated "
                   "generic tile loops");
            // forward the type from the generic body to the loop body
            assert(isa<PointerType>(origArg.getType()) ||
                   origArg.getType() == llvmArg.getType() &&
                       "expected non-tensor arguments to be unchanged by type "
                       "conversion");
            entryArgs.push_back(llvmArg);
          }
        }

        // entryArgs.append(adaptor.getOperands().begin(),
        //                  adaptor.getOperands().end());
        LLVM::CondBrOp::create(rewriter, loc, cond, bodyEntry, entryArgs,
                               afterBlock, /*exitArgs=*/{});

        if (failed(rewriter.convertRegionTypes(bodyEntry->getParent(),
                                               *getTypeConverter()))) {
          return rewriter.notifyMatchFailure(op,
                                             "could not convert body types");
        }
      }

      // forward the final reduction result through the after block
      rewriter.setInsertionPointToStart(afterBlock);
      if (hasReductions)
        result = afterResult;
      // tensor accs don't need forwarding — they're accessed via alloca ptrs
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

  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateGenericOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GenericOpConversion>(typeConverter, targetInfo, benefit);
}
