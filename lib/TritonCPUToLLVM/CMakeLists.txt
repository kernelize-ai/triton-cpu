add_triton_library(TritonCPUToLLVM
    TypeConverter.cpp

    DEPENDS
    TritonCPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
