add_triton_library(TritonNPUToLLVM
    ElementwiseOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonNPUToLLVM.cpp
    TargetInfo.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
