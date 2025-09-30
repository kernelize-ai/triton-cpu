add_triton_library(TritonCPUToLLVM
    Allocation.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    KernelWrapper.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    MemoryOpToLLVM.cpp
    SharedMemoryGlobalConversion.cpp
    TritonCPUToLLVM.cpp
    TargetInfo.cpp
    Utility.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
