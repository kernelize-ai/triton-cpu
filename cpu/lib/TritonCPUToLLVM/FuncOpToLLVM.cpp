
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"

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
    // Push back SPMD program args - (x,y,z) grid index and (gridX, gridY,
    // gridZ) grid sizes.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto sharedPtrTy =
        LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());

    // 1. Modify the function type to add the new arguments.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    bool isKernel = triton::isKernel(funcOp);
    if (!isKernel)
      return funcOp; // TODO: pass shared memory to child functions

    amendedInputTy.push_back(i32_ty);      // thread_id
    amendedInputTy.push_back(i32_ty);      // x
    amendedInputTy.push_back(i32_ty);      // y
    amendedInputTy.push_back(i32_ty);      // z
    amendedInputTy.push_back(i32_ty);      // gridX
    amendedInputTy.push_back(i32_ty);      // gridY
    amendedInputTy.push_back(i32_ty);      // gridZ
    amendedInputTy.push_back(sharedPtrTy); // shared memory ptr

    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, funcTy.getResults());

    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    if (auto argAttrs = funcOp.getAllArgAttrs()) {
      llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
                                                         argAttrs.end());
      while (amendedArgAttrs.size() < amendedInputTy.size()) {
        amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
      }
      amendedAttrs.push_back(
          rewriter.getNamedAttr(funcOp.getArgAttrsAttrName(),
                                rewriter.getArrayAttr(amendedArgAttrs)));
    }

    // 3. Add the new arguments to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();

    region.addArgument(i32_ty, loc);      // thread_id
    region.addArgument(i32_ty, loc);      // x
    region.addArgument(i32_ty, loc);      // y
    region.addArgument(i32_ty, loc);      // z
    region.addArgument(i32_ty, loc);      // gridX
    region.addArgument(i32_ty, loc);      // gridY
    region.addArgument(i32_ty, loc);      // gridZ
    region.addArgument(sharedPtrTy, loc); // shared memory ptr

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
      const int sharedMemoryPtrArgIndex = newFuncOp.getNumArguments() - 1;
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
