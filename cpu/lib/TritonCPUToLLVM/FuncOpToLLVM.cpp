
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

// NOTE: [Additional Function Arguments]
// Adds additional function arguments for program ID and grid size after
// upstream arguments (shared/global memory).

struct FuncOpSPMDParamConversion
    : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpSPMDParamConversion(LLVMTypeConverter &converter,
                            const cpu::TargetInfo &targetInfo,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back SPMD program args
    //  - launch size: &{ grid_x, grid_y, grid_z, block_x, block_y, block_z }
    //  - launch id: &{ grid_x, grid_y, grid_z, block_x, block_y, block_z }
    //  - shared memory ptr
    //  - cpu barrier
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto sharedPtrTy =
        LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
    auto voidPtrTy = LLVM::LLVMPointerType::get(ctx);

    // 1. Modify the function type to add the new arguments.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    bool isKernel = triton::isKernel(funcOp);
    if (!isKernel)
      return funcOp; // TODO: pass shared memory to child functions

    amendedInputTy.push_back(voidPtrTy);   // launch sz
    amendedInputTy.push_back(voidPtrTy);   // launch id
    amendedInputTy.push_back(sharedPtrTy); // shared memory ptr
    amendedInputTy.push_back(voidPtrTy);   // cpu barrier

    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, funcTy.getResults());

    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    int sharedMemoryOffset = amendedInputTy.size() + cpu::kSharedMemoryOffset;
    if (auto argAttrs = funcOp.getAllArgAttrs()) {
      llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
                                                         argAttrs.end());
      while (amendedArgAttrs.size() < amendedInputTy.size()) {
        SmallVector<NamedAttribute> attrs{
            rewriter.getNamedAttr("llvm.nonnull", rewriter.getUnitAttr())};
        // add alignment attribute for the shared memory pointer
        if (amendedArgAttrs.size() == sharedMemoryOffset) {
          attrs.push_back(rewriter.getNamedAttr(
              "llvm.align",
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 64)));
        } else {
          attrs.push_back(
              rewriter.getNamedAttr("llvm.noalias", rewriter.getUnitAttr()));
        }
        amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx, attrs));
      }
      amendedAttrs.push_back(
          rewriter.getNamedAttr(funcOp.getArgAttrsAttrName(),
                                rewriter.getArrayAttr(amendedArgAttrs)));
    }

    // 3. Add the new arguments to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();

    auto nameLoc = [&](const char *name) {
      return NameLoc::get(rewriter.getStringAttr(name));
    };

    region.addArgument(voidPtrTy, nameLoc("launch_sz"));
    region.addArgument(voidPtrTy, nameLoc("launch_id"));
    region.addArgument(sharedPtrTy, nameLoc("shared_mem_ptr"));
    region.addArgument(voidPtrTy, nameLoc("cpu_barrier"));

    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;

    auto ctx = funcOp->getContext();

    if (triton::isKernel(funcOp)) {
      // TODO: is this needed? should we use a CPU specific attribute?
      // Set an attribute to indicate this function is a kernel entry.
      //   newFuncOp->setAttr(NVVM::NVVMDialect::getKernelFuncAttrName(),
      //  rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }

    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);

    // set the alignment on the shared memory pointer argument
    if (triton::isKernel(funcOp)) {
      const int sharedMemoryPtrArgIndex = newFuncOp.getNumArguments() - 2;
      assert(sharedMemoryPtrArgIndex >= 0 &&
             "expected at least one function argument");
      auto sharedMemoryPtrArg = newFuncOp.getArgument(sharedMemoryPtrArgIndex);
      assert(isa<LLVM::LLVMPointerType>(sharedMemoryPtrArg.getType()) &&
             "expected the shared memory function argument to be a pointer");
      const auto i32_type = mlir::IntegerType::get(newFuncOp.getContext(), 32);
      newFuncOp.setArgAttr(
          sharedMemoryPtrArgIndex, LLVM::LLVMDialect::getAlignAttrName(),
          mlir::IntegerAttr::get(i32_type, targetInfo.CacheLineSizeBytes));
    }

    return success();
  }

private:
  const cpu::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<FuncOpSPMDParamConversion>(typeConverter, targetInfo, benefit);
}
