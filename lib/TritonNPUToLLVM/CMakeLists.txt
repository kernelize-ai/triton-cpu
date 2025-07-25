add_triton_library(TritonNPUToLLVM
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonNPUToLLVM.cpp
    TargetInfo.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
