add_triton_library(TritonCPUTransforms
    Coalesce.cpp
    MakePersistentKernel.cpp
    LoopPeeling.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
