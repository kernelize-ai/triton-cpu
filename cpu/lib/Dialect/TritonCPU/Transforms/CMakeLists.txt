add_triton_library(TritonCPUTransforms
    Coalesce.cpp
    MakePersistentKernel.cpp
    SplitBlocks.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
