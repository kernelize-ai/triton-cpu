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

static Value allocateGlobalBuffer(ConversionPatternRewriter &rewriter,
                                  Location loc, Type elemTy, unsigned numElems,
                                  unsigned nameIdx, cpu::GenericOp generic) {
  auto func = generic->getParentOfType<LLVM::LLVMFuncOp>();
  assert(func && "expected generic op to be inside an LLVM function");
  auto module = generic->getParentOfType<ModuleOp>();
  assert(module && "expected generic op to be inside a module");

  auto globalArrayTy = LLVM::LLVMArrayType::get(elemTy, numElems);

  std::string globalName;
  do {
    globalName = (func.getName() + "_tac_" + std::to_string(nameIdx++)).str();
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
       << globalName << "` for tensor of size " << numElems);
  Value globalPtr = LLVM::AddressOfOp::create(
      rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
      globalName);
  return globalPtr;
}

struct ArgInfo {
  enum class Kind { IV, IterArg, Ins };

  ArgInfo(Kind k) : kind(k) {}
  ArgInfo(Kind k, Type tritonType, Type llvmType)
      : kind(k), tritonType(tritonType), llvmType(llvmType) {}

  Kind kind;
  Type tritonType;
  Type llvmType;
  Value operand;

  Value bufferPtr;       // only for global memory backed args
  unsigned numElems = 0; // only for global memory backed args
  Value convertedArg; // only for non-alloca tensor args that need to be tiled
                      // (unroll / static path)

  bool requiresMaterialization() const {
    return kind == Kind::Ins && isa<RankedTensorType>(tritonType) && !bufferPtr;
  }
};

class LoopHelper {
public:
  LoopHelper(ArrayRef<ArgInfo> args, ArrayRef<Value> loopIterArgs,
             Value flatOffset, cpu::GenericOp generic,
             const TypeConverter *typeConverter,
             ConversionPatternRewriter &rewriter);

  SmallVector<Value> getEntryArgs() {
    return {tileOffsets.begin(), tileOffsets.end()};
  }

  void addTileOffset(Value offset) { tileOffsets.push_back(offset); }
  void setTileOffset(unsigned dim, Value offset) { tileOffsets[dim] = offset; }

  void popTileOffset() { tileOffsets.pop_back(); }

  void incrementFlatOffset(ConversionPatternRewriter &rewriter, Value inc) {
    auto b = TritonLLVMOpBuilder(flatOffset.getLoc(), rewriter);
    flatOffset = b.add(flatOffset, inc);
  }

  // build loop body block arguments for the current loop level. Adds
  // arguments for loop induction vars, generic iter args, and generic
  // operands. Handles extracting the appropriate tile for non-alloca tensor
  // operands in the static (unrolled) path. Slices memory backed tensors into
  // tiles (LLVM struct type).
  SmallVector<Value>
  getLoopBodyBlockArgs(ConversionPatternRewriter &rewriter, unsigned vectorSize,
                       std::optional<unsigned> loopIndex = std::nullopt);

  // stores generic body tensor results into global memory buffers.
  void scatterResults(ConversionPatternRewriter &rewriter,
                      ArrayRef<Value> results, ArrayRef<Type> resultTypes,
                      unsigned vectorSize);

  void updateIterArgs(ArrayRef<Value> newIterArgVals) {
    assert(newIterArgVals.size() == loopIterArgs.size());
    for (auto [i, newVal] : llvm::enumerate(newIterArgVals))
      loopIterArgs[i] = newVal;
  }

  SmallVector<Value> getIterArgVals() const { return loopIterArgs; }

  SmallVector<Value> getResults() const {
    SmallVector<Value> ret(loopIterArgs.begin(), loopIterArgs.end());
    ret.append(materializedResults.begin(), materializedResults.end());
    return ret;
  }

private:
  SmallVector<ArgInfo> args;
  SmallVector<Value> loopIterArgs;

  SmallVector<Value> materializedResults;
  SmallVector<Type> materializedResultElementTypes;

  SmallVector<Value> tileOffsets;
  Value flatOffset;
};

LoopHelper::LoopHelper(ArrayRef<ArgInfo> args, ArrayRef<Value> loopIterArgs,
                       Value flatOffset, cpu::GenericOp generic,
                       const TypeConverter *typeConverter,
                       ConversionPatternRewriter &rewriter)
    : args(args.begin(), args.end()),
      loopIterArgs(loopIterArgs.begin(), loopIterArgs.end()),
      flatOffset(flatOffset) {

  // Create temporary buffers in thread local global memory for materialized
  // results. This avoids loop-carried <blockSize x elemTy> phi nodes
  // which would allocate tens of kilobytes on the stack and cause stack
  // overflows for large block sizes (e.g. blockSize=4096).
  //
  // Globals are addressed via AddressOf in the function entry block so the
  // pointer dominates all uses. Each thread gets its own copy since the
  // globals are thread-local.
  //
  // TODO: we should probably check that generic results are only used by
  // other generics or we will run into conversion problems
  for (auto [i, resultTy] : llvm::enumerate(llvm::drop_begin(
           generic.getResultTypes(), generic.getNumIterArgs()))) {
    auto tensorTy = cast<RankedTensorType>(resultTy);
    int64_t tensorElems =
        std::accumulate(tensorTy.getShape().begin(), tensorTy.getShape().end(),
                        int64_t{1}, std::multiplies<>());
    Type elemTy = typeConverter->convertType(tensorTy.getElementType());
    materializedResultElementTypes.push_back(elemTy);

    Value globalPtr = allocateGlobalBuffer(
        rewriter, generic.getResult(i + generic.getNumIterArgs()).getLoc(),
        elemTy, tensorElems, materializedResults.size(), generic);
    materializedResults.push_back(globalPtr);
  }
}

SmallVector<Value>
LoopHelper::getLoopBodyBlockArgs(ConversionPatternRewriter &rewriter,
                                 unsigned vectorSize,
                                 std::optional<unsigned> loopIndex) {
  // start with IVs
  SmallVector<Value> blockArgs = {tileOffsets.begin(), tileOffsets.end()};

  // iter args
  blockArgs.append(loopIterArgs.begin(), loopIterArgs.end());

  // Add ins args
  for (const auto &argInfo : args) {
    if (argInfo.kind == ArgInfo::Kind::Ins) {
      Location loc = argInfo.operand.getLoc();

      // if we are in the static (unrolled) path, we need to handle
      // non-alloca tensor args by extracting the appropriate chunk for this
      // iteration.
      if (loopIndex.has_value()) {
        auto tensorTy = dyn_cast<RankedTensorType>(argInfo.tritonType);
        if (tensorTy && !argInfo.bufferPtr) {
          assert(argInfo.convertedArg && "expected adaptor-converted arg "
                                         "for non-alloca tensor operand");
          auto tileStructTy = cast<LLVM::LLVMStructType>(argInfo.llvmType);
          Value tile = LLVM::UndefOp::create(rewriter, loc, tileStructTy);
          TritonLLVMOpBuilder b(argInfo.operand.getLoc(), rewriter);

          for (unsigned j = 0; j < vectorSize; ++j) {
            int64_t srcIndex = loopIndex.value() * vectorSize + j;
            LDBG("Read " << srcIndex);
            Value extractedElement = b.extract_val(
                tileStructTy.getBody()[j], argInfo.convertedArg, srcIndex);
            tile = b.insert_val(tileStructTy, tile, extractedElement, j);
          }

          Value castedTile = UnrealizedConversionCastOp::create(
                                 rewriter, loc, argInfo.tritonType, tile)
                                 .getResult(0);
          blockArgs.push_back(castedTile);
          continue;
        }
      }

      if (argInfo.bufferPtr) {
        // load this tiles elements from the global buffer
        auto tileStructTy = cast<LLVM::LLVMStructType>(argInfo.llvmType);
        Type elemTy = tileStructTy.getBody()[0];
        Value tileStruct = LLVM::UndefOp::create(rewriter, loc, tileStructTy);
        auto b = TritonLLVMOpBuilder(loc, rewriter);

        // TODO: should this be vectorSize?
        for (unsigned j = 0; j < vectorSize; ++j) {
          Value globalIdx = b.add(flatOffset, b.i32_val(j));
          Value gep = b.gep(LLVM::LLVMPointerType::get(rewriter.getContext()),
                            elemTy, argInfo.bufferPtr, ValueRange{globalIdx});
          Value elem = b.load(elemTy, gep);
          tileStruct = b.insert_val(tileStructTy, tileStruct, elem, j);
        }
        blockArgs.push_back(tileStruct);
      } else if (isa<PointerType>(argInfo.tritonType)) {
        // we might have a tt.ptr hiding in an unrealized conversion cast
        // due to late conversion of generic op block arguments. Use
        // unrealized conversion cast which will be folded away later
        if (argInfo.tritonType != argInfo.llvmType) {
          blockArgs.push_back(
              UnrealizedConversionCastOp::create(
                  rewriter, loc, argInfo.llvmType, argInfo.operand)
                  .getResult(0));
        } else {
          blockArgs.push_back(argInfo.operand);
        }
      } else {
        assert(!isa<RankedTensorType>(argInfo.tritonType));
        assert(argInfo.tritonType == argInfo.llvmType &&
               "expected non-tensor arguments to be unchanged by type "
               "conversion");
        blockArgs.push_back(argInfo.operand);
      }
    }
  }

  return blockArgs;
}

void LoopHelper::scatterResults(ConversionPatternRewriter &rewriter,
                                ArrayRef<Value> results,
                                ArrayRef<Type> resultTypes,
                                unsigned vectorSize) {
  for (auto [tile, tileType, ptr, elemTy] :
       llvm::zip(results, resultTypes, materializedResults,
                 materializedResultElementTypes)) {
    Location loc = tile.getLoc(); // or ptr?
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto llvmStruct = cast<LLVM::LLVMStructType>(tileType);
    Value llvmTile =
        UnrealizedConversionCastOp::create(rewriter, loc, tileType, tile)
            .getResult(0);
    // TODO: should this be vector size or should we read the number of
    // elements from the tile struct?
    for (unsigned j = 0; j < vectorSize; ++j) {
      Value elem = b.extract_val(llvmStruct.getBody()[j], llvmTile, j);
      Value idx = b.add(flatOffset, b.i32_val(j));
      Value gep =
          b.gep(ptr_ty(rewriter.getContext()), elemTy, ptr, ValueRange{idx});
      b.store(elem, gep);
    }
  }
}

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

  SmallVector<ArgInfo>
  buildArgInfos(cpu::GenericOp op, OpAdaptor adaptor,
                ArrayRef<int32_t> tileShape,
                ConversionPatternRewriter &rewriter) const {
    SmallVector<ArgInfo> args;

    // handle loop induction vars first
    for (unsigned i = 0; i < tileShape.size(); i++) {
      args.emplace_back(ArgInfo(ArgInfo::Kind::IV, nullptr, i32_ty));
    }

    for (unsigned i = 0; i < op.getNumIterArgs(); i++) {
      Type tritonType = op.getIterArg(i).getType();
      assert(!isa<RankedTensorType>(tritonType) &&
             "tensor type iter args not yet supported");
      Type llvmType = getTypeConverter()->convertType(tritonType);

      // TODO: create alloca backed buffers corresponding to init arg and copy
      // init arg in if we have a tensor type.
      args.emplace_back(ArgInfo(ArgInfo::Kind::IterArg, tritonType, llvmType));
    }

    Block *body = &op.getBody().front();
    const auto &ins = op.getIns();

    for (auto [i, origArg, llvmArg] :
         llvm::enumerate(body->getArguments().drop_front(op.getInsArgOffset()),
                         adaptor.getIns())) {
      ArgInfo argInfo(ArgInfo::Kind::Ins);

      argInfo.tritonType = origArg.getType();
      argInfo.operand = op.getIns()[i];
      Value ptrArg = getGenericOutputTensorAsPtr(op, i, llvmArg);
      if (ptrArg) {
        auto allocationTensorTy = cast<RankedTensorType>(ins[i].getType());
        int64_t numElems =
            std::accumulate(allocationTensorTy.getShape().begin(),
                            allocationTensorTy.getShape().end(), int64_t{1},
                            std::multiplies<>());
        argInfo.bufferPtr = ptrArg;
        argInfo.numElems = numElems;
        argInfo.llvmType = getTypeConverter()->convertType(origArg.getType());
      } else {
        argInfo.llvmType = llvmArg.getType();
        if (isa<RankedTensorType>(argInfo.tritonType)) {
          // tensor arguments that aren't backed by an alloca need to store the
          // adaptor value so they can be sliced when inputs to the cloned loop
          // body are created during unrolling
          argInfo.convertedArg = adaptor.getIns()[i];
        }
      }
      args.push_back(argInfo);
    }

    return args;
  }

  // Returns {newIterArgVals, tensorTiles} from the tile's ttc.yield.
  // iterArgVals provides the current accumulator values for iter args.
  std::pair<SmallVector<Value>, SmallVector<Value>>
  cloneTileBody(cpu::GenericOp op, ConversionPatternRewriter &rewriter,
                ArrayRef<Value> chunkedArgs) const {
    assert(op.getBody().getBlocks().size() == 1 &&
           "expected generic op body to have one block");
    Block *body = &op.getBody().front();

    IRMapping mapping;
    for (auto [bodyArg, newArg] :
         llvm::zip(body->getArguments(), chunkedArgs)) {
      mapping.map(bodyArg, newArg);
    }

    unsigned numIterArgs = op.getNumIterArgs();
    SmallVector<Value> newIterArgVals, tensorTiles;

    // clone the body remapping operands. When handling the. yield, track tiles
    // requiring materialization separately
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

  // statically unroll all tiles. Required when any tensor input uses
  // llvm.extractvalue (which requires compile-time indices), or when there is
  // only one tile and loop overhead is unnecessary.
  void emitUnrolled(cpu::GenericOp op, LoopHelper &helper,
                    ConversionPatternRewriter &rewriter, unsigned numChunks,
                    unsigned vectorSize, ArrayRef<int32_t> blockShape,
                    ArrayRef<int32_t> tileShape) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    LDBG("Unrolling generic, cloning body for " << numChunks
                                                << " loop iterations");

    // initialize tile offsets inside the helper
    unsigned rank = blockShape.size();
    for (int d = rank - 1; d >= 0; --d) {
      helper.addTileOffset(Value());
    }

    for (unsigned i = 0; i < numChunks; ++i) {
      SmallVector<Value> perDimOffsets(rank);
      unsigned remaining = i;
      LDBG("i = " << i);
      for (int d = rank - 1; d >= 0; --d) {
        unsigned nc = blockShape[d] / tileShape[d];
        unsigned chunkIdx = remaining % nc;
        LDBG("perDimOffsets[" << d << "] = " << chunkIdx << " * "
                              << tileShape[d] << " = "
                              << (chunkIdx * tileShape[d]));
        helper.setTileOffset(d, b.i32_val(chunkIdx * tileShape[d]));
        remaining /= nc;
      }

      helper.incrementFlatOffset(rewriter, b.i32_val(vectorSize));

      // uses the tile offset state above
      SmallVector<Value> chunkedArgs =
          helper.getLoopBodyBlockArgs(rewriter, vectorSize, i);

      Value flatOffset = b.i32_val(i * vectorSize);
      LDBG("flatOffset = " << (i * vectorSize));
      auto [newIterArgVals, loopTiles] =
          cloneTileBody(op, rewriter, chunkedArgs);
      helper.updateIterArgs(newIterArgVals);
      SmallVector<Type> resultTypes =
          llvm::map_to_vector(loopTiles, [&](Value tile) {
            return getTypeConverter()->convertType(tile.getType());
          });
      helper.scatterResults(rewriter, loopTiles, resultTypes, vectorSize);
    }
  }

  // Emits a loop over numChunks tiles, carrying initCarried values as
  // loop-carried state alongside the loop counter. bodyFn receives the current
  // carried values and returns the updated carried values for the back edge.
  // Returns the final carried values available after the loop.
  SmallVector<Value>
  emitSingleLoop(ConversionPatternRewriter &rewriter, Location loc,
                 Value numChunks, int32_t tileSize, ArrayRef<Value> initCarried,
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
                                      loopI, numChunks);
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
  // state.
  void emitNestedLoops(cpu::GenericOp op, LoopHelper &helper,
                       ConversionPatternRewriter &rewriter, unsigned dim,
                       ArrayRef<Value> blockShape, ArrayRef<int32_t> tileShape,
                       unsigned vectorSize,
                       Block *innerAfterBlock = nullptr) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    unsigned rank = tileShape.size();

    if (dim == rank) {
      // innermost: inline the body region
      assert(innerAfterBlock &&
             "expected inner after block for innermost loop");

      Region &bodyRegion = op.getBody();
      Block *bodyEntry = &bodyRegion.front();

      // build tile args before inlining (references block arguments)
      auto entryArgs = helper.getLoopBodyBlockArgs(rewriter, vectorSize);

      rewriter.inlineRegionBefore(bodyRegion, *innerAfterBlock->getParent(),
                                  innerAfterBlock->getIterator());

      TypeConverter::SignatureConversion sigConv(bodyEntry->getNumArguments());
      for (auto [idx, arg] : llvm::enumerate(bodyEntry->getArguments()))
        sigConv.addInputs(idx, getTypeConverter()->convertType(arg.getType()));
      bodyEntry = rewriter.applySignatureConversion(bodyEntry, sigConv,
                                                    getTypeConverter());

      LLVM::BrOp::create(rewriter, loc, entryArgs, bodyEntry);

      unsigned numIterArgs = op.getNumIterArgs();
      for (Block &block : llvm::make_range(bodyEntry->getIterator(),
                                           innerAfterBlock->getIterator())) {
        auto yieldOp = dyn_cast<cpu::YieldOp>(block.getTerminator());
        if (!yieldOp)
          continue;

        rewriter.setInsertionPoint(yieldOp);
        // extract new iter arg values from the leading yield operands
        SmallVector<Value> newIterArgVals(yieldOp.getValues().begin(),
                                          yieldOp.getValues().begin() +
                                              numIterArgs);
        helper.updateIterArgs(newIterArgVals);

        // scatter non-iter-arg tensor tiles
        SmallVector<Value> loopTiles(yieldOp.getValues().begin() + numIterArgs,
                                     yieldOp.getValues().end());
        SmallVector<Type> resultTypes =
            llvm::map_to_vector(loopTiles, [&](Value tile) {
              return getTypeConverter()->convertType(tile.getType());
            });
        helper.scatterResults(rewriter, loopTiles, resultTypes, vectorSize);

        rewriter.eraseOp(yieldOp);
        rewriter.setInsertionPointToEnd(&block);
        break;
      }

      return;
    }

    // emit single loop, recurse to next dimension
    Value numChunks = b.sdiv(blockShape[dim], b.i32_val(tileShape[dim]));

    Value innerStride = b.i32_val(vectorSize);
    for (unsigned d = dim + 1; d < rank; ++d)
      innerStride =
          b.mul(innerStride, b.sdiv(blockShape[d], b.i32_val(tileShape[d])));

    auto finalCarried = emitSingleLoop(
        rewriter, loc, numChunks, tileShape[dim], helper.getIterArgVals(),
        [&](Value loopI, Value dimTileOffset, ArrayRef<Value> currentCarried,
            Block *afterBlock) -> SmallVector<Value> {
          helper.addTileOffset(dimTileOffset);

          auto bb = TritonLLVMOpBuilder(loc, rewriter);
          helper.incrementFlatOffset(rewriter, bb.mul(loopI, innerStride));
          SmallVector<Value> innerIterArgVals(currentCarried.begin(),
                                              currentCarried.end());

          helper.updateIterArgs(innerIterArgVals);
          emitNestedLoops(op, helper, rewriter, dim + 1, blockShape, tileShape,
                          vectorSize, afterBlock);

          helper.popTileOffset();
          return helper.getIterArgVals();
        });

    helper.updateIterArgs(finalCarried);
  }

  LogicalResult
  matchAndRewrite(cpu::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto tileShapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("tileShape");
    assert(tileShapeAttr && "expected generic op to have tileShape attribute");
    ArrayRef<int32_t> tileShape = tileShapeAttr.asArrayRef();

    assert(!op.getBlockShape().empty() &&
           op.getBlockShape().size() == tileShape.size() &&
           "blockShape and tileShape must be non-empty and of the same size");

    unsigned vectorSize = 1;
    for (unsigned d = 0; d < tileShape.size(); ++d) {
      vectorSize *= tileShape[d];
    }

    // TODO: put this into extraClassDefinitions?
    std::string s;
    llvm::raw_string_ostream os(s);
    op->print(os, OpPrintingFlags().skipRegions());
    LDBG("Lowering generic op: " << s
                                 << "\n  with vectorSize = " << vectorSize);

    SmallVector<ArgInfo> argInfos =
        buildArgInfos(op, adaptor, tileShape, rewriter);
    for (auto argInfo : argInfos) {
      LDBG("ArgInfo: kind = " << static_cast<int>(argInfo.kind)
                              << ", tritonType = " << argInfo.tritonType
                              << ", llvmType = " << argInfo.llvmType
                              << ", bufferPtr = " << argInfo.bufferPtr
                              << ", numElems = " << argInfo.numElems);
    }

    // Tensor args passed as LLVM structs require static tile indices
    // (llvm.extractvalue only accepts attribute indices). Alloca-backed tensors
    // from a prior generic are GEP-indexed and support dynamic offsets.
    const bool requiresTensorArgMaterialization =
        llvm::any_of(argInfos, [](const ArgInfo &argInfo) {
          return argInfo.requiresMaterialization();
        });

    // Iter arg values start from init_vals and are updated tile-by-tile.
    SmallVector<Value> iterArgVals(adaptor.getInitVals().begin(),
                                   adaptor.getInitVals().end());

    int numChunks = -1;
    SmallVector<int32_t> blockShape;
    for (auto [i, dim] : llvm::enumerate(op.getBlockShape())) {
      APInt val;
      if (!matchPattern(dim, m_ConstantInt(&val))) {
        numChunks = -1;
        break;
      }
      int64_t dimSize = val.getSExtValue();
      blockShape.push_back(static_cast<int32_t>(dimSize));
      if (numChunks == -1) {
        numChunks = dimSize / tileShape[i];
      } else {
        numChunks *= (dimSize / tileShape[i]);
      }
    }
    if (requiresTensorArgMaterialization && numChunks < 0) {
      op.emitError()
          << "non-constant block shape is not supported when generic op has "
             "tensor args that require materialization";
      return failure();
    }

    SmallVector<Value> results;
    if (requiresTensorArgMaterialization || numChunks == 1) {
      assert(numChunks > 0 && "expected numChunks to be positive when required "
                              "or statically computable");
      assert(blockShape.size() == tileShape.size() &&
             "expected blockShape and tileShape to have "
             "the same rank");
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      LoopHelper helper(argInfos, iterArgVals, /*flatOffset=*/b.i32_val(0), op,
                        getTypeConverter(), rewriter);
      emitUnrolled(op, helper, rewriter, numChunks, vectorSize, blockShape,
                   tileShape);
      results = helper.getResults();
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

      auto b = TritonLLVMOpBuilder(loc, rewriter);
      LoopHelper helper(argInfos, iterArgVals, /*flatOffset=*/b.i32_val(0), op,
                        getTypeConverter(), rewriter);
      SmallVector<Value> blockShapeVals(op.getBlockShape().begin(),
                                        op.getBlockShape().end());
      emitNestedLoops(op, helper, rewriter, /*dim=*/0, blockShapeVals,
                      tileShape, vectorSize);
      results = helper.getResults();
    }

    if (results.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateGenericOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GenericOpConversion>(typeConverter, benefit);
}
