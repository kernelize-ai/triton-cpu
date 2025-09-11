#ifndef TRITON_CONVERSION_TRITONNPU_TO_TENSTORRENT_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONNPU_TO_TENSTORRENT_TYPE

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::triton;

class TritonNPUToTenstorrentTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;
  TritonNPUToTenstorrentTypeConverter() : TypeConverter() {}
};

#endif
