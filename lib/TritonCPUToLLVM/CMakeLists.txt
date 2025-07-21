add_triton_library(TritonCPUToLLVM
    MemoryOpToLLVM.cpp
    TypeConverter.cpp 
    
    DEPENDS
    TritonCPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
