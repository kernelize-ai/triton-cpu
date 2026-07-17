add_triton_library(TritonCPUToLLVM
    Allocation.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GenericOpToLLVM.cpp
    GPUOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    LowerDotMicrokernelToSME.cpp
    LowerSMEMicrokernelToLLVM.cpp
    MakeDynamicRangeOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    MemoryOpToLLVM.cpp
    OutlineDotMicrokernel.cpp
    SharedMemoryGlobalConversion.cpp
    TritonCPUToLLVM.cpp
    TargetInfo.cpp
    Utility.cpp

    DEPENDS
    TritonCPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRArmSMEDialect
)
