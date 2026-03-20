add_triton_library(TritonCPUAnalysis
    AxisInfo.cpp

    DEPENDS
    TritonTableGen
    TritonGPUTableGen
    TritonCPUTableGen

    LINK_LIBS PUBLIC
    MLIRAnalysis
    TritonAnalysis
)
