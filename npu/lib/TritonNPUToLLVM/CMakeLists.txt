add_triton_library(TritonNPUToLLVM
    Allocation.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TargetInfo.cpp
    TritonNPUToLLVM.cpp
    TypeConverter.cpp
    UKernelToLLVM.cpp
    Utility.cpp

    DEPENDS
    TritonNPUConversionPassIncGen
)
