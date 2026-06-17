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
  ArgInfo(Kind k, Type tritonType, Type llvmType, Value operand)
      : kind(k), tritonType(tritonType), llvmType(llvmType), operand(operand) {}

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

  void print(llvm::raw_ostream &os) const {
    static constexpr StringRef kindNames[] = {"IV", "IterArg", "Ins"};
    os << "ArgInfo{kind=" << kindNames[static_cast<int>(kind)]
       << ", tritonType=" << tritonType << ", llvmType=" << llvmType
       << ", operand=" << operand << ", bufferPtr=" << bufferPtr
       << ", numElems=" << numElems << ", convertedArg=" << convertedArg << "}";
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ArgInfo &a) {
  a.print(os);
  return os;
}

class LoopHelper {
public:
  LoopHelper(ArrayRef<ArgInfo> args, cpu::GenericOp generic,
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

  void incrementIterArgOffset(ConversionPatternRewriter &rewriter, Value inc) {
    auto b = TritonLLVMOpBuilder(flatOffset.getLoc(), rewriter);
    iterArgOffset = b.add(iterArgOffset, inc);
  }

  // build loop body block arguments for the current loop level. Adds
  // arguments for loop induction vars, generic iter args, and generic
  // operands. Handles extracting the appropriate tile for non-alloca tensor
  // operands in the static (unrolled) path. Slices memory backed tensors into
  // tiles (LLVM struct type).
  SmallVector<Value>
  getLoopBodyBlockArgs(ConversionPatternRewriter &rewriter,
                       ArrayRef<unsigned> elemOffset,
                       std::optional<unsigned> loopIndex = std::nullopt);

  // stores generic body tensor results into global memory buffers.
  void scatterResults(ConversionPatternRewriter &rewriter,
                      ArrayRef<Value> results, ArrayRef<Type> resultTypes);

  // update loop carried iter arg state, scattering to global buffers as
  // necessary.
  void updateIterArgs(ConversionPatternRewriter &rewriter,
                      ArrayRef<Value> newIterArgVals);

  SmallVector<Value> getIterArgVals() const { return loopIterArgs; }

  SmallVector<Value> getResults(ConversionPatternRewriter &rewriter,
                                const bool requiresTensorMaterialization,
                                ValueRange genericResults,
                                const TypeConverter *typeConverter) const {
    SmallVector<Value> ret;
    ret.append(loopIterArgs.begin(), loopIterArgs.end());
    if (requiresTensorMaterialization) {
      auto tensorResults =
          llvm::to_vector(llvm::make_filter_range(genericResults, [](Value v) {
            return isa<RankedTensorType>(v.getType());
          }));

      for (auto [materializedResultPtr, opResult] :
           llvm::zip(materializedResults, tensorResults)) {
        assert(isa<LLVM::LLVMPointerType>(materializedResultPtr.getType()) &&
               "expected materialized result buffer to be an LLVM pointer");
        auto outputStructType = cast<LLVM::LLVMStructType>(
            typeConverter->convertType(opResult.getType()));
        Type elemTy = outputStructType.getBody()[0];
        Value result = LLVM::UndefOp::create(rewriter, opResult.getLoc(),
                                             outputStructType);

        auto b = TritonLLVMOpBuilder(result.getLoc(), rewriter);
        for (unsigned j = 0; j < outputStructType.getBody().size(); ++j) {
          Value gep = b.gep(LLVM::LLVMPointerType::get(rewriter.getContext()),
                            elemTy, materializedResultPtr, b.i32_val(j));
          Value loaded = b.load(elemTy, gep);
          result = b.insert_val(outputStructType, result, loaded, j);
        }
        ret.push_back(result);
      }
    } else {
      ret.append(materializedResults.begin(), materializedResults.end());
    }
    return ret;
  }

  bool isReductionDim(unsigned d) const {
    return llvm::is_contained(reductionDims, d);
  }

private:
  SmallVector<ArgInfo> args;
  SmallVector<Value> loopIterArgs;

  SmallVector<Value> materializedResults;
  SmallVector<RankedTensorType> materializedResultTensorTypes;

  SmallVector<Value> tileOffsets;
  Value flatOffset;
  Value iterArgOffset;
  SmallVector<int32_t> reductionDims;
};

LoopHelper::LoopHelper(ArrayRef<ArgInfo> args, cpu::GenericOp generic,
                       const TypeConverter *typeConverter,
                       ConversionPatternRewriter &rewriter)
    : args(args.begin(), args.end()),
      reductionDims(generic.getReductionDims()) {

  auto gb = TritonLLVMOpBuilder(generic.getLoc(), rewriter);
  flatOffset = gb.i32_val(0);
  iterArgOffset = gb.i32_val(0);

  for (auto [idx, arg] : llvm::enumerate(this->args)) {
    if (arg.kind == ArgInfo::Kind::IterArg) {
      if (auto tileTensorTy = dyn_cast<RankedTensorType>(arg.tritonType)) {
        Type elemTy = typeConverter->convertType(tileTensorTy.getElementType());

        Location loc = arg.operand.getLoc();
        // allocate a global buffer for this iter arg
        auto tensorTy = cast<RankedTensorType>(arg.operand.getType());
        int64_t tensorElems = std::accumulate(tensorTy.getShape().begin(),
                                              tensorTy.getShape().end(),
                                              int64_t{1}, std::multiplies<>());

        Value globalPtr = allocateGlobalBuffer(
            rewriter, loc, elemTy, tensorElems, loopIterArgs.size(), generic);
        this->args[idx].bufferPtr = globalPtr;

        // copy the init value
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        assert(arg.operand.getDefiningOp() &&
               "expected iter arg operand to be defined by an op");
        auto castedInitOp =
            cast<UnrealizedConversionCastOp>(arg.operand.getDefiningOp());
        auto initVal = castedInitOp.getOperand(0);
        auto initStructTy = cast<LLVM::LLVMStructType>(initVal.getType());
        for (unsigned i = 0; i < initStructTy.getBody().size(); ++i) {
          Value elem = b.extract_val(initStructTy.getBody()[i], initVal, i);
          Value gep = b.gep(LLVM::LLVMPointerType::get(rewriter.getContext()),
                            elemTy, globalPtr, b.i32_val(i));
          b.store(elem, gep);
        }

        loopIterArgs.push_back(globalPtr);
      } else {
        loopIterArgs.push_back(arg.operand);
      }
    }
  }

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
    materializedResultTensorTypes.push_back(tensorTy);

    Value globalPtr = allocateGlobalBuffer(
        rewriter, generic.getResult(i + generic.getNumIterArgs()).getLoc(),
        elemTy, tensorElems, materializedResults.size() + loopIterArgs.size(),
        generic);
    materializedResults.push_back(globalPtr);
  }
}

SmallVector<Value> LoopHelper::getLoopBodyBlockArgs(
    ConversionPatternRewriter &rewriter,
    ArrayRef<unsigned> elemOffset, // only used if loopIndex is present
    std::optional<unsigned> loopIndex) {
  // start with IVs
  SmallVector<Value> blockArgs = {tileOffsets.begin(), tileOffsets.end()};

  // iter args
  unsigned scalarIdx = 0;
  for (const auto &arg : args) {
    if (arg.kind == ArgInfo::Kind::IterArg) {
      unsigned crtIndex = scalarIdx++;
      if (arg.bufferPtr) {
        // load this tiles elements from the global buffer
        auto tileStructTy = cast<LLVM::LLVMStructType>(arg.llvmType);
        Type elemTy = tileStructTy.getBody()[0];
        Value tileStruct =
            LLVM::UndefOp::create(rewriter, arg.operand.getLoc(), tileStructTy);
        auto b = TritonLLVMOpBuilder(arg.operand.getLoc(), rewriter);

        for (unsigned j = 0; j < tileStructTy.getBody().size(); ++j) {
          Value globalIdx = b.add(iterArgOffset, b.i32_val(j));
          Value gep = b.gep(LLVM::LLVMPointerType::get(rewriter.getContext()),
                            elemTy, arg.bufferPtr, ValueRange{globalIdx});
          Value elem = b.load(elemTy, gep);
          tileStruct = b.insert_val(tileStructTy, tileStruct, elem, j);
        }
        blockArgs.push_back(tileStruct);
      } else {
        blockArgs.push_back(loopIterArgs[crtIndex]);
      }
    }
  }

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

          auto kRegister = StringAttr::get(rewriter.getContext(), "register");
          LinearLayout tileLayout = triton::gpu::toLinearLayout(tensorTy);

          // get the starting register index for this tile in the source tensor
          LinearLayout srcLayout =
              triton::gpu::toLinearLayout(
                  cast<RankedTensorType>(argInfo.operand.getType()))
                  .pseudoinvert();
          SmallVector<std::pair<StringAttr, int32_t>> srcLayoutOffsets;
          assert(srcLayout.getNumInDims() == elemOffset.size() &&
                 "expected number of tile element offsets to match source "
                 "layout in dims");
          for (auto [dim, layoutOffset] :
               llvm::zip(srcLayout.getInDimNames(), elemOffset)) {
            // the srcLayout is inverted, so map to the tile layouts out dims to
            // get the correct tensor size
            auto tileDimSize = tileLayout.getOutDimSize(dim);
            // if the src layout dim size matches the tile layout dim size then
            // this dimension is not tiled
            int32_t offset =
                tileDimSize == srcLayout.getInDimSize(dim) ? 0 : layoutOffset;
            srcLayoutOffsets.emplace_back(dim, offset);
          }
          auto originOffset = srcLayout.apply(srcLayoutOffsets).front();
          assert(originOffset.first == kRegister &&
                 "expected origin offset to be in the register space");
          int32_t origin = originOffset.second;

          tileLayout = tileLayout.flattenOuts().pseudoinvert();

          LDBG("tile for loop index " << loopIndex.value()
                                      << " has origin register " << origin);

          auto inDimName = llvm::to_vector(tileLayout.getInDimNames())[0];
          for (unsigned j = 0; j < tileLayout.getInDimSize(inDimName); ++j) {
            auto regOffset = tileLayout.apply({{inDimName, j}}).front();
            assert(regOffset.first == kRegister &&
                   "expected offset to be in the register space");
            int32_t dstReg = regOffset.second;
            int32_t srcReg = dstReg ^ origin;

            Value extractedElement = b.extract_val(
                tileStructTy.getBody()[j], argInfo.convertedArg, srcReg);
            tile = b.insert_val(tileStructTy, tile, extractedElement, dstReg);
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

        for (unsigned j = 0; j < tileStructTy.getBody().size(); ++j) {
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
                                ArrayRef<Type> resultTypes) {
  for (auto [tile, tileType, ptr, tensorTy] :
       llvm::zip(results, resultTypes, materializedResults,
                 materializedResultTensorTypes)) {
    Location loc = tile.getLoc(); // or ptr?
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto llvmStruct = cast<LLVM::LLVMStructType>(tileType);
    Value llvmTile =
        UnrealizedConversionCastOp::create(rewriter, loc, llvmStruct, tile)
            .getResult(0);
    Type elemTy = llvmStruct.getBody()[0];

    auto tileTensorTy = cast<RankedTensorType>(tile.getType());
    LinearLayout tileLayout = triton::gpu::toLinearLayout(tileTensorTy);
    LinearLayout fullTensorInverseLayout =
        triton::gpu::toLinearLayout(tensorTy).pseudoinvert();

    auto tileRegisterToTensorRegister =
        tileLayout.compose(fullTensorInverseLayout);
    auto kRegister = StringAttr::get(rewriter.getContext(), "register");
    // can we use flattenOffsets or sublayout to avoid needing these?
    auto kLane = StringAttr::get(rewriter.getContext(), "lane");
    auto kWarp = StringAttr::get(rewriter.getContext(), "warp");
    auto kBlock = StringAttr::get(rewriter.getContext(), "block");

    SmallVector<std::pair<StringAttr, Value>> originInputDims;
    for (auto [dim, offset] :
         llvm::zip(fullTensorInverseLayout.getInDimNames(), tileOffsets)) {
      originInputDims.push_back({dim, offset});
    }

    // maps the origin input dims to an origin register which we use to convert
    // the relative tile registers to absolute positions in the output tensor
    auto tileOriginMappedDims = applyLinearLayout(
        loc, rewriter, fullTensorInverseLayout, originInputDims);
    assert(!tileOriginMappedDims.empty() &&
           tileOriginMappedDims.front().first == kRegister);
    Value originRegister = tileOriginMappedDims.front().second;

    for (unsigned j = 0; j < llvmStruct.getBody().size(); ++j) {
      // compute the tile register to full tensor register mapping
      Value elem = b.extract_val(llvmStruct.getBody()[j], llvmTile, j);

      auto relativeRegister = tileRegisterToTensorRegister.apply(
          {{kRegister, j}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      assert(!relativeRegister.empty() &&
             relativeRegister.front().first == kRegister);

      Value bufferIndex =
          b.xor_(originRegister, b.i32_val(relativeRegister.front().second));
      Value gep =
          b.gep(ptr_ty(rewriter.getContext()), elemTy, ptr, bufferIndex);
      b.store(elem, gep);
    }
  }
}

void LoopHelper::updateIterArgs(ConversionPatternRewriter &rewriter,
                                ArrayRef<Value> newIterArgVals) {
  unsigned totalIterArgs = llvm::count_if(
      args, [](const ArgInfo &a) { return a.kind == ArgInfo::Kind::IterArg; });
  assert(newIterArgVals.size() == totalIterArgs &&
         "expected iter arg update size to match total number of iter args");

  loopIterArgs.clear();

  unsigned scalarIdx = 0, yieldIdx = 0;
  for (const auto &arg : args) {
    if (arg.kind == ArgInfo::Kind::IterArg) {
      Value yieldVal = newIterArgVals[yieldIdx++];
      if (arg.bufferPtr) {
        // buffer ptr inter args may be forwarded from the previous loop
        // level. only update the tile data for iter args that are the result
        // of a ttc.yield, which will have triton types
        if (isa<RankedTensorType>(yieldVal.getType())) {
          Location loc = yieldVal.getLoc();

          auto b = TritonLLVMOpBuilder(loc, rewriter);
          auto llvmStruct = cast<LLVM::LLVMStructType>(arg.llvmType);
          auto castedYieldVal = UnrealizedConversionCastOp::create(
                                    rewriter, loc, arg.llvmType, yieldVal)
                                    .getResult(0);
          for (unsigned j = 0; j < llvmStruct.getBody().size(); ++j) {
            Value elem =
                b.extract_val(llvmStruct.getBody()[j], castedYieldVal, j);
            Value idx = b.add(iterArgOffset, b.i32_val(j));
            Value gep =
                b.gep(ptr_ty(rewriter.getContext()), llvmStruct.getBody()[j],
                      arg.bufferPtr, ValueRange{idx});
            b.store(elem, gep);
          }
        }
        loopIterArgs.push_back(arg.bufferPtr);
      } else {
        loopIterArgs.push_back(yieldVal);
      }
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
    if (!isa<RankedTensorType>(result.getType()))
      return {};
    auto defGeneric = dyn_cast<cpu::GenericOp>(result.getOwner());
    if (!defGeneric)
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
      args.emplace_back(ArgInfo(ArgInfo::Kind::IV, nullptr, i32_ty, nullptr));
    }

    for (unsigned i = 0; i < op.getNumIterArgs(); i++) {
      Type tritonType = op.getIterArg(i).getType();
      Type llvmType = getTypeConverter()->convertType(tritonType);

      args.emplace_back(ArgInfo(ArgInfo::Kind::IterArg, tritonType, llvmType,
                                op.getInitVals()[i]));
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
        argInfo.llvmType = getTypeConverter()->convertType(origArg.getType());
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

    // clone the body remapping operands. When handling the yield, track tiles
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

    // collect the per-dimension tile offsets for this chunk in elemOffset,
    // which is used for calculating the appropriate tile slice for non-alloca
    // tensor inputs. These offsets are also stored in the helper as state for
    // use when building loop body block arguments but we need scalar values for
    // insert/extract value operations.
    SmallVector<unsigned> elemOffset(rank);
    for (unsigned i = 0; i < numChunks; ++i) {
      unsigned remaining = i;
      LDBG("i = " << i);
      for (int d = rank - 1; d >= 0; --d) {
        unsigned nc = blockShape[d] / tileShape[d];
        unsigned chunkIdx = remaining % nc;
        elemOffset[d] = chunkIdx * tileShape[d];
        LDBG("perDimOffsets[" << d << "] = " << chunkIdx << " * "
                              << tileShape[d] << " = "
                              << (chunkIdx * tileShape[d]));
        helper.setTileOffset(d, b.i32_val(chunkIdx * tileShape[d]));
        remaining /= nc;
      }

      // uses the tile offset state above
      SmallVector<Value> chunkedArgs =
          helper.getLoopBodyBlockArgs(rewriter, elemOffset, i);

      auto [newIterArgVals, loopTiles] =
          cloneTileBody(op, rewriter, chunkedArgs);
      helper.updateIterArgs(rewriter, newIterArgVals);
      SmallVector<Type> resultTypes =
          llvm::map_to_vector(loopTiles, [&](Value tile) {
            return getTypeConverter()->convertType(tile.getType());
          });
      helper.scatterResults(rewriter, loopTiles, resultTypes);
      helper.incrementFlatOffset(rewriter, b.i32_val(vectorSize));
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
      auto entryArgs = helper.getLoopBodyBlockArgs(rewriter, {});

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
        helper.updateIterArgs(rewriter, newIterArgVals);

        // scatter non-iter-arg tensor tiles
        SmallVector<Value> loopTiles(yieldOp.getValues().begin() + numIterArgs,
                                     yieldOp.getValues().end());
        SmallVector<Type> resultTypes =
            llvm::map_to_vector(loopTiles, [&](Value tile) {
              return getTypeConverter()->convertType(tile.getType());
            });
        helper.scatterResults(rewriter, loopTiles, resultTypes);

        rewriter.eraseOp(yieldOp);
        rewriter.setInsertionPointToEnd(&block);
        break;
      }

      return;
    }

    // emit single loop, recurse to next dimension
    Value numChunks = b.sdiv(blockShape[dim], b.i32_val(tileShape[dim]));

    // TODO: can we encapsulate this in the loop helper?
    Value innerStride = b.i32_val(vectorSize);
    for (unsigned d = dim + 1; d < rank; ++d)
      innerStride =
          b.mul(innerStride, b.sdiv(blockShape[d], b.i32_val(tileShape[d])));
    Value innerIterArgStride = b.i32_val(0);
    if (!helper.isReductionDim(dim)) {
      unsigned innerArgVectorSize = 1;
      // TODO: don't recompute this over and over
      for (unsigned d = 0; d < tileShape.size(); ++d) {
        if (!helper.isReductionDim(d)) {
          innerArgVectorSize *= tileShape[d];
        }
      }
      innerIterArgStride = b.i32_val(innerArgVectorSize);
      for (unsigned d = dim + 1; d < rank; ++d)
        if (!helper.isReductionDim(d))
          innerIterArgStride =
              b.mul(innerIterArgStride,
                    b.sdiv(blockShape[d], b.i32_val(tileShape[d])));
    }

    auto finalCarried = emitSingleLoop(
        rewriter, loc, numChunks, tileShape[dim], helper.getIterArgVals(),
        [&](Value loopI, Value dimTileOffset, ArrayRef<Value> currentCarried,
            Block *afterBlock) -> SmallVector<Value> {
          helper.addTileOffset(dimTileOffset);

          auto bb = TritonLLVMOpBuilder(loc, rewriter);
          helper.incrementFlatOffset(rewriter, bb.mul(loopI, innerStride));
          helper.incrementIterArgOffset(rewriter,
                                        bb.mul(loopI, innerIterArgStride));
          SmallVector<Value> innerIterArgVals(currentCarried.begin(),
                                              currentCarried.end());

          helper.updateIterArgs(rewriter, innerIterArgVals);
          emitNestedLoops(op, helper, rewriter, dim + 1, blockShape, tileShape,
                          vectorSize, afterBlock);

          helper.popTileOffset();
          return helper.getIterArgVals();
        });

    helper.updateIterArgs(rewriter, finalCarried);
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

    LDBG("Lowering generic op: " << op.getHeader()
                                 << "\n  with vectorSize = " << vectorSize);

    SmallVector<ArgInfo> argInfos =
        buildArgInfos(op, adaptor, tileShape, rewriter);

    LLVM_DEBUG({
      for (auto [i, argInfo] : llvm::enumerate(argInfos)) {
        DBGS() << "Arg " << i << ": ";
        argInfo.print(llvm::dbgs());
        DBGS() << "\n";
      }
    });

    // Tensor args passed as LLVM structs require static tile indices
    // (llvm.extractvalue only accepts attribute indices). Alloca-backed tensors
    // from a prior generic are GEP-indexed and support dynamic offsets.
    const bool requiresTensorArgMaterialization =
        llvm::any_of(argInfos, [](const ArgInfo &argInfo) {
          return argInfo.requiresMaterialization();
        });

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

    SmallVector<Value> results;
    if (requiresTensorArgMaterialization || numChunks == 1) {
      assert(numChunks > 0 && "expected numChunks to be positive when required "
                              "or statically computable");
      assert(blockShape.size() == tileShape.size() &&
             "expected blockShape and tileShape to have "
             "the same rank");
      LoopHelper helper(argInfos, op, getTypeConverter(), rewriter);
      emitUnrolled(op, helper, rewriter, numChunks, vectorSize, blockShape,
                   tileShape);
      results = helper.getResults(rewriter, !allUsersAreGeneric,
                                  op.getResults(), getTypeConverter());
    } else {
      assert(allUsersAreGeneric && "expected all generic op users to also be "
                                   "generic ops on dynamic path");

      LoopHelper helper(argInfos, op, getTypeConverter(), rewriter);
      SmallVector<Value> blockShapeVals(op.getBlockShape().begin(),
                                        op.getBlockShape().end());
      emitNestedLoops(op, helper, rewriter, /*dim=*/0, blockShapeVals,
                      tileShape, vectorSize);
      results =
          helper.getResults(rewriter, /*requiresTensorMaterialization=*/false,
                            op.getResults(), getTypeConverter());
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
