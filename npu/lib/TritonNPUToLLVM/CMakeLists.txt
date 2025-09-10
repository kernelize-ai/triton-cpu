add_triton_library(TritonNPUToLLVM
    Allocation.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    SharedMemoryGlobalConversion.cpp
    TritonNPUToLLVM.cpp
    TargetInfo.cpp
    Utility.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
