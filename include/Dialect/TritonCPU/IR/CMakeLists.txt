set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonCPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=triton_cpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=triton_cpu)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
#mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=triton_cpu)
#mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=triton_cpu)
add_mlir_doc(TritonCPUDialect TritonCPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TritonCPUOps TritonCPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(TritonCPUTableGen)

