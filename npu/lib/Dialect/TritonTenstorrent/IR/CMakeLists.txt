add_triton_library(TritonTenstorrentIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonTenstorrentTableGen

  LINK_LIBS PUBLIC
  TritonIR
)
