add_triton_library(TritonCPUTransforms
    AccelerateMatmul.cpp
    Coalesce.cpp
    MakePersistentKernel.cpp
    TestLaneMapAnalysis.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
