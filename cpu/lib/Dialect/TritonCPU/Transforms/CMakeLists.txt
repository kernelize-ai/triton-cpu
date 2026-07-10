add_triton_library(TritonCPUTransforms
    AccelerateMatmul.cpp
    Coalesce.cpp
    MakePersistentKernel.cpp
    TileAndFuse.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
