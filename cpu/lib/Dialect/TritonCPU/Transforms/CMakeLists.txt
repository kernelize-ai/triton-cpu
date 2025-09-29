add_triton_library(TritonCPUTransforms
    AddKernelStream.cpp
    Coalesce.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
