#include "npu/include/TritonNPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

TritonNPUToLLVMTypeConverter::TritonNPUToLLVMTypeConverter(
    MLIRContext *ctx, const LowerToLLVMOptions &option,
    const TargetInfoBase &targetInfo, const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, option, targetInfo, analysis) {}
