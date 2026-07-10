add_triton_library(TritonCPUToLLVM
    Allocation.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GenericOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    MakeDynamicRangeOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    SharedMemoryGlobalConversion.cpp
    TritonCPUToLLVM.cpp
    TargetInfo.cpp
    Utility.cpp

    DEPENDS
    TritonCPUConversionPassIncGen
)
