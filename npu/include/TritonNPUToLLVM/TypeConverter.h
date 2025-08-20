#ifndef TRITON_CONVERSION_TRITONNPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONNPU_TO_LLVM_TYPECONVERTER_H

#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

class TritonNPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonNPUToLLVMTypeConverter(MLIRContext *ctx,
                               const LowerToLLVMOptions &option,
                               const TargetInfoBase &targetInfo,
                               const DataLayoutAnalysis *analysis = nullptr);
};

#endif
