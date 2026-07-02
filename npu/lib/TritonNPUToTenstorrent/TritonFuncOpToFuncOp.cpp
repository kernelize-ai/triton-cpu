#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "PatternTritonNPUToTenstorrent.h"
#include "Utility.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/FunctionTypes.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline tt::ttkernel::ThreadType
getThreadTypeFromFunctionName(StringRef funcName) {
  if (funcName.ends_with("__compute"))
    return tt::ttkernel::ThreadType::Compute;
  else if (funcName.ends_with("__reader") || funcName.ends_with("__writer"))
    return tt::ttkernel::ThreadType::Noc;

  llvm_unreachable("unexpected function name suffix");
}

struct ConvertTritonFunc : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::isKernel(funcOp)) {
      return rewriter.notifyMatchFailure(
          funcOp, "non-kernel functions are not yet supported");
    }

    Location loc = funcOp.getLoc();
    MLIRContext *context = funcOp.getContext();
    auto typeConverter = getTypeConverter();

    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    // build the arg spec using the function ops - tt.ptr ops become cb ports,
    // others are ignored
    // TODO: this holds true for the compute kernel - what about reader/writer?
    SmallVector<ttkernel::ArgAttr> ctArgs;
    for (auto argType : tritonTy.getInputs()) {
      if (isa<PointerType>(argType) || isa<TensorDescType>(argType)) {
        ctArgs.push_back(rewriter.getAttr<ttkernel::ArgAttr>(
            ttkernel::ArgType::CBPort, ctArgs.size()));
      }
    }

    Block &oldEntry = funcOp.getBody().front();

    // create a new function with no arguments
    mlir::FunctionType newTy = mlir::FunctionType::get(
        context, /*inputs=*/TypeRange{}, /*results=*/TypeRange{});
    // skip triton attributes

    auto newFunc =
        mlir::func::FuncOp::create(rewriter, loc, funcOp.getName(), newTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFunc->setAttr(SymbolTable::getVisibilityAttrName(), vis);

    newFunc->setAttr(tt::ttkernel::ThreadTypeAttr::name,
                     rewriter.getAttr<tt::ttkernel::ThreadTypeAttr>(
                         getThreadTypeFromFunctionName(funcOp.getName())));

    // copy the body
    Region &newRegion = newFunc.getBody();

    auto *newEntry = rewriter.createBlock(&newRegion);
    // copy all the blocks first, then we'll fix up the arguments and merge the
    // fix in
    rewriter.inlineRegionBefore(funcOp.getBody(), newRegion, newRegion.end());

    SmallVector<unsigned> ptrArgIndices;
    int32_t numArgs = -1;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(newEntry);

      SmallVector<Value> argReplacements;
      argReplacements.reserve(oldEntry.getNumArguments());

      auto getArg = [&](const unsigned idx, Type newType) -> Value {
        Value indexVal = arith::createIndexConstant(loc, rewriter, idx);
        // function arguments are common across kernels
        // block start, block end arguments are per-core but are added later,
        // during SPMD op lowering
        Value getArgVal = ttkernel::GetCommonArgValOp::create(
            rewriter, loc, rewriter.getIntegerType(32), indexVal);
        if (isa<IntegerType>(newType) && newType.getIntOrFloatBitWidth() < 32) {
          getArgVal =
              arith::TruncIOp::create(rewriter, loc, newType, getArgVal);
        }
        return getArgVal;
      };

      // The multicast lowering (ConvertMulticastOp) appends 2 semaphore
      // compile-time args per multicast op, placed right after the CB ports.
      // Reserve those slots so the TensorAccessorArgs chain starts *after* the
      // semaphores and doesn't collide with them. We count multicasts
      // program-wide (not just in this func) so the reservation is uniform
      // across the reader/writer/compute kernels, which share one compile_args
      // vector in the driver -- otherwise the writer (no multicast) would put
      // its accessors where the reader's semaphores live.
      //
      // TODO: Match by op-name string (`triton_tenstorrent.multicast`) rather
      // than the C++ type to avoid including the TritonTenstorrent dialect
      // header here, which would pull the `mlir::triton::npu::tt` namespace
      // into scope and shadow the `tt::ttkernel` references throughout this
      // file.
      unsigned numReservedSemaphoreArgs = 0;
      if (auto moduleOp = funcOp->getParentOfType<ModuleOp>())
        moduleOp.walk([&](Operation *op) {
          if (op->getName().getStringRef() == "triton_tenstorrent.multicast")
            numReservedSemaphoreArgs += 2;
        });

      // Build one chained `ttkernel.TensorAccessorArgs` per tensor input (in
      // input order) at the start of the kernel. The first sits at compile-time
      // arg offset `ctArgs.size() + numReservedSemaphoreArgs` (CB ports, then
      // the reserved multicast semaphores); the rest chain via `prev_args` so
      // the runtime resolves their variable-size offsets. Each is tagged with
      // the common-arg index of its buffer's base address so the memory-op
      // lowering can look up the right accessor when building the
      // `TensorAccessor` for a load/store. The compute thread reads from CBs,
      // never communicates via semaphores, and never issues NOC tile reads, so
      // it has no use for TensorAccessors.
      const bool isComputeThread =
          getThreadTypeFromFunctionName(funcOp.getName()) ==
          tt::ttkernel::ThreadType::Compute;
      Value prevAccessorArgs = nullptr;
      auto buildAccessorArgs = [&](unsigned baseArgIdx) {
        if (isComputeThread)
          return;
        ttkernel::TensorAccessorArgsOp argsOp;
        if (!prevAccessorArgs) {
          Value ctaBase = arith::createConstantI32(
              loc, rewriter,
              static_cast<int32_t>(ctArgs.size() + numReservedSemaphoreArgs));
          Value crtaBase = arith::createConstantI32(loc, rewriter, 0);
          argsOp = ttkernel::TensorAccessorArgsOp::create(
              rewriter, loc, ttkernel::TensorAccessorArgsType::get(context),
              /*cta_base=*/ctaBase, /*crta_base=*/crtaBase,
              /*prev_args=*/Value(), /*cta_expr=*/StringAttr(),
              /*crta_expr=*/StringAttr());
        } else {
          argsOp = ttkernel::TensorAccessorArgsOp::create(
              rewriter, loc, ttkernel::TensorAccessorArgsType::get(context),
              /*cta_base=*/Value(), /*crta_base=*/Value(),
              /*prev_args=*/prevAccessorArgs, /*cta_expr=*/StringAttr(),
              /*crta_expr=*/StringAttr());
        }
        argsOp->setAttr(kAccessorBaseArgIndexAttr,
                        rewriter.getI32IntegerAttr(baseArgIdx));
        prevAccessorArgs = argsOp.getResult();
      };

      unsigned idx = 0;
      for (auto arg : oldEntry.getArguments()) {
        Type oldType = arg.getType();
        if (isa<TensorDescType>(oldType)) {
          SmallVector<Type> unpacked;
          assert(typeConverter->convertType(oldType, unpacked).succeeded() &&
                 "failed to convert tensor descriptor type");
          ptrArgIndices.push_back(idx);
          buildAccessorArgs(idx);

          SmallVector<Value> unpackedValues;
          unpackedValues.reserve(unpacked.size());
          for (unsigned i = 0; i < unpacked.size(); i++) {
            Type t = unpacked[i];
            LDBG("Replacing arg " << idx << " of type " << oldType
                                  << " with type " << t);
            unpackedValues.push_back(getArg(idx++, t));
          }
          Value packed = UnrealizedConversionCastOp::create(
                             rewriter, loc, oldType, unpackedValues)
                             .getResult(0);
          argReplacements.push_back(packed);
        } else {
          if (isa<PointerType>(oldType)) {
            ptrArgIndices.push_back(idx);
            buildAccessorArgs(idx);
          }
          Type newType = typeConverter->convertType(oldType);

          LDBG("Replacing arg " << idx << " of type " << oldType
                                << " with type " << newType);

          Value replacement = getArg(idx++, newType);
          // TODO: Do we need the unrealized conversion cast type converter
          // source materialization? This if is necessary to avoid source
          // materializations on the type converter from failing the funcTarget
          // section of the NPU To TTKernel lowering
          if (isa<PointerType>(oldType))
            replacement = UnrealizedConversionCastOp::create(
                              rewriter, loc, oldType, replacement)
                              .getResult(0);
          argReplacements.push_back(replacement);
        }
      }
      numArgs = idx;
      rewriter.mergeBlocks(&oldEntry, newEntry, argReplacements);
    }

    // Number of user args
    // TODO: add launch params (grid size, block size, shared memory size, etc)
    assert(numArgs >= 0 && "numArgs not set");
    newFunc->setAttr(kTTNumCommonArgsAttr, rewriter.getI32IntegerAttr(numArgs));
    newFunc->setAttr(kTTNumPerCoreArgsAttr, rewriter.getI32IntegerAttr(0));
    ttmlir::utils::setFunctionType(newFunc,
                                   ttmlir::utils::FunctionType::Kernel);

    // handle populating arg spec post tensor descriptor argument expansion
    // TODO: can we get input arguments from the newFunc here? I don't think so,
    // since the rewriter is still holding all our changes to the module.
    SmallVector<ttkernel::ArgAttr> rtArgs;
    // TODO: Currently we have to rewrite all scalar common args to constants
    // in ttnn.generic and rewrite the ptr common args indices to be 0 indexed
    // within the set of ptr args. Eventually we will want to support scalars as
    // uniform runtime args but this requires tt-mlir changes.
    for (unsigned i = 0; i < ptrArgIndices.size(); i++) {
      rtArgs.push_back(rewriter.getAttr<ttkernel::ArgAttr>(
          ttkernel::ArgType::BufferAddress, i));
    }

    ttkernel::ArgSpecAttr::setArgSpec(
        newFunc, rewriter.getAttr<ttkernel::ArgSpecAttr>(rtArgs, ctArgs));

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ConvertReturnOp : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

} // namespace

void populateFuncOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<ConvertTritonFunc>(typeConverter, patterns.getContext(),
                                  benefit);
  patterns.add<ConvertReturnOp>(typeConverter, patterns.getContext(), benefit);
}

} // namespace npu
} // namespace triton
} // namespace mlir
