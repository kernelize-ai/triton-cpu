add_triton_library(TritonCPUToLLVM
    Allocation.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    SharedMemoryGlobalConversion.cpp
    TritonCPUToLLVM.cpp
    TargetInfo.cpp
    Utility.cpp

    DEPENDS
    TritonCPUConversionPassIncGen
)
