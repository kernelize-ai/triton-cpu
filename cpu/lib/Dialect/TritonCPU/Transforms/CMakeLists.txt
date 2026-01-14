add_triton_library(TritonCPUTransforms
    AccelerateMatmul.cpp
    Coalesce.cpp
    MakePersistentKernel.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
