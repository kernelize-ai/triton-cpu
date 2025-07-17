add_triton_library(TritonNPUToLLVM
    TritonNPUToLLVM.cpp
    TargetInfo.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
