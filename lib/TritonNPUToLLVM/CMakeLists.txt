add_triton_library(TritonNPUToLLVM
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonNPUToLLVM.cpp
    TargetInfo.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
