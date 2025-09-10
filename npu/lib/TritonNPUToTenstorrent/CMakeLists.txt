add_triton_library(TritonNPUToTenstorrent
    FuncOpToMLIRFunc.cpp
    TritonNPUToTenstorrent.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen
)
