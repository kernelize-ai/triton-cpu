add_triton_library(TritonCPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonCPUTableGen

  LINK_LIBS PUBLIC
  TritonIR
)
