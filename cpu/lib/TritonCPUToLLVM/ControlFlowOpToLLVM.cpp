#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct CallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  CallOpConversion(LLVMTypeConverter &converter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::CallOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *calleeOp =
        SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
    auto funcOp = dyn_cast_or_null<FunctionOpInterface>(calleeOp);
    if (triton::isKernel(funcOp)) {
      return failure();
    }

    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter);
    auto newCallOp =
        convertCallOpToLLVMCallOp(callOp, promotedOperands, rewriter);
    if (!newCallOp)
      return failure();
    auto results = getCallOpResults(callOp, newCallOp, rewriter);
    rewriter.replaceOp(callOp, results);
    return success();
  }

  LLVM::CallOp
  convertCallOpToLLVMCallOp(triton::CallOp callOp,
                            ArrayRef<Value> promotedOperands,
                            ConversionPatternRewriter &rewriter) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return nullptr;
    }
    auto newCallOp = LLVM::CallOp::create(rewriter, callOp.getLoc(),
                                          packedResult ? TypeRange(packedResult)
                                                       : TypeRange(),
                                          promotedOperands, callOp->getAttrs());
    newCallOp.getProperties().setOpBundleSizes(
        rewriter.getDenseI32ArrayAttr({}));
    newCallOp.getProperties().setOperandSegmentSizes(
        {static_cast<int>(promotedOperands.size()), 0});
    return newCallOp;
  }

  SmallVector<Value>
  getCallOpResults(triton::CallOp callOp, LLVM::CallOp newCallOp,
                   ConversionPatternRewriter &rewriter) const {
    auto numResults = callOp.getNumResults();
    SmallVector<Value> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(LLVM::ExtractValueOp::create(
            rewriter, callOp.getLoc(), newCallOp->getResult(0), i));
      }
    }
    return results;
  }

  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateControlFlowOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<CallOpConversion>(typeConverter, targetInfo, benefit);
}
