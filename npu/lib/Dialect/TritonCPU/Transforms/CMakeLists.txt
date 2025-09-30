add_triton_library(TritonCPUTransforms
    Coalesce.cpp

    DEPENDS
    TritonCPUTransformsIncGen

    LINK_LIBS PUBLIC
    TritonGPUIR
)
