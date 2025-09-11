#ifndef TRITON_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_TRITONNPU_OP_TO_TENSTORRENT_H
#define TRITON_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_TRITONNPU_OP_TO_TENSTORRENT_H

#include "../TritonNPUToLLVM/TargetInfo.h" // TODO

namespace mlir {
namespace triton {
namespace npu::tt {

constexpr int patternBenefitDefault = 1;

void populateFuncOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);

} // namespace npu::tt
} // namespace triton
} // namespace mlir
#endif
