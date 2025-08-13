add_triton_library(TritonNPUToTenstorrent
    TritonNPUToTenstorrent.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen
)
