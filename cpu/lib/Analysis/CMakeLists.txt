add_triton_library(TritonCPUAnalysis
    LaneMapAnalysis.cpp

    DEPENDS
    TritonCPUTableGen
    TritonTableGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRLLVMDialect
    TritonIR
    TritonGPUIR
    TritonCPUIR
)
