add_triton_library(TritonCPUTransforms
    Coalesce.cpp
    MakePersistentKernel.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
