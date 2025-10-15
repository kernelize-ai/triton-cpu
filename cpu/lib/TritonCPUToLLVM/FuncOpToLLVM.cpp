
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
    bool isKernel = triton::isKernel(funcOp);
    if (!isKernel)
      return funcOp; // TODO: pass shared memory to child functions

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

    auto funcTy = funcOp.getFunctionType();
    SmallVector<Type> amendedInputTy;
    // 0. Make all user arguments void*
    for (auto paramTy : funcTy.getInputs()) {
      if (isa<triton::PointerType>(paramTy)) {
        amendedInputTy.push_back(paramTy);
      } else {
        amendedInputTy.push_back(voidPtrTy);
      }
    }
    int userArgSize = amendedInputTy.size();

    // 1. Append launch arguments to the function type.
    amendedInputTy.push_back(voidPtrTy);   // launch sz
    amendedInputTy.push_back(voidPtrTy);   // launch id
    amendedInputTy.push_back(sharedPtrTy); // shared memory ptr
    amendedInputTy.push_back(voidPtrTy);   // cpu barrier

    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, funcTy.getResults());
    // 2. Modify the user argument attributes to add noalias
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    int sharedMemoryOffset = userArgSize + cpu::kSharedMemoryOffset;
    llvm::SmallVector<mlir::Attribute> amendedArgAttrs;
    SmallVector<mlir::DictionaryAttr> userArgAttrs;
    funcOp.getAllArgAttrs(userArgAttrs);
    for (auto attr : userArgAttrs) {
      SmallVector<NamedAttribute> newArgAttrs{attr.begin(), attr.end()};
      if (!attr.contains("llvm.noalias")) {
        newArgAttrs.push_back(rewriter.getNamedAttr(
          "llvm.noalias", rewriter.getUnitAttr()));
        newArgAttrs.push_back(rewriter.getNamedAttr(
          "llvm.nonnull", rewriter.getUnitAttr()));
      }
      amendedArgAttrs.push_back(DictionaryAttr::get(ctx, newArgAttrs));
    }
    while (amendedArgAttrs.size() < userArgSize) {
      amendedArgAttrs.push_back(DictionaryAttr::get(ctx, {
          rewriter.getNamedAttr("llvm.nonnull", rewriter.getUnitAttr()),
          rewriter.getNamedAttr("llvm.noalias", rewriter.getUnitAttr())
      }));
    }

    // Add attributes for the launch arguments
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
    // 3. Create the amended function op
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();

    // 4. Update the user params to be pointers
    OpBuilder argBuilder(region);
    for (int i = 0; i < userArgSize; i++) {
      auto arg = region.getArgument(i);
      auto argType = arg.getType();
      if (!isa<triton::PointerType>(argType)) {
        auto argUsers = arg.getUsers();
        arg.setType(voidPtrTy);
        auto b = TritonLLVMOpBuilder(arg.getLoc(), argBuilder);
        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto idxTy = typeConverter->convertType(argType);
        auto gep = b.gep(ptrTy, idxTy, arg, b.i32_val(0));
        auto newArgValue = b.load(idxTy, gep).getResult();
        auto newArgConvValue = argBuilder.create<UnrealizedConversionCastOp>(arg.getLoc(), argType, newArgValue).getResult(0);
        for (auto user : argUsers) {
          user->replaceUsesOfWith(arg, newArgConvValue);
        }
      }
    }

    // 5. Add the launch arguments to the region
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
