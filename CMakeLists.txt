include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)

# Tenstorrent includes
set(TENSTORRENT_MLIR $ENV{TENSTORRENT_MLIR}) # TODO
set(TTMLIR_LIB_DIR "${TENSTORRENT_MLIR}/build/lib")

set(TTMLIR_LIB_NAMES
    MLIRTTKernelDialect
    MLIRTTCoreDialect
    MLIRTTMetalDialect
)

# Build full paths to each dylib
set(TTMLIR_LIBS)
foreach(lib ${TTMLIR_LIB_NAMES})
    list(APPEND TTMLIR_LIBS
        "${TTMLIR_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib}.a"
    )
endforeach()


include_directories(
    "${TENSTORRENT_MLIR}/include"
    "${TENSTORRENT_MLIR}/build/include"
)

add_subdirectory(npu)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonNPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu.cc LINK_LIBS
    TritonNPUToLLVM
    TritonNPUToTenstorrent

    LINK_LIBS ${TTMLIR_LIBS} MLIRQuantDialect
  )
  target_link_libraries(TritonNPU PRIVATE Python3::Module pybind11::headers)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()

# uKernel Libs

include(ExternalProject)

# Sleef

set(SLEEF_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/sleef-src")

# Sleef clones TLFloat as an external project which is problematic, as it will
# cache temp ninja paths and not get the new path on a re-configure. Delete the
# directory manually if it exists to force a re-install for each re-configure.
if(EXISTS "${SLEEF_SRC_DIR}/src/sleef/submodules/tlfloat")
  file(REMOVE_RECURSE "${SLEEF_SRC_DIR}/src/sleef/submodules/tlfloat")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps/sleef-build/ext_tlfloat-prefix")
endif()

set(SLEEF_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/sleef-install)

ExternalProject_Add(sleef
  PREFIX ${SLEEF_SRC_DIR}
  # Or if you want to pull from git directly:
  GIT_REPOSITORY https://github.com/shibatch/sleef.git
  GIT_TAG master

  CMAKE_ARGS
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_INSTALL_PREFIX=${SLEEF_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=Release
    -DSLEEF_BUILD_SHARED_LIBS=OFF
    -DSLEEF_BUILD_DFT=OFF
    -DSLEEF_BUILD_GNUABI_LIBS=OFF
    -DSLEEF_BUILD_TESTS=OFF
    -DSLEEF_BUILD_SCALAR_LIB=OFF

  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/sleef-build

  # Control which steps happen
  UPDATE_DISCONNECTED 1    # don’t re-fetch every time
  INSTALL_DIR ${SLEEF_INSTALL_DIR}
)

# Add a custom step to copy libsleef.a after install
ExternalProject_Add_Step(sleef copy_lib
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  COMMAND ${CMAKE_COMMAND} -E copy
          ${SLEEF_INSTALL_DIR}/lib/libsleef.a
          ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  DEPENDEES install
  ALWAYS 1
)

# So you can depend on a phony target to ensure the copy happens
add_custom_target(sleef_artifact ALL DEPENDS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
add_dependencies(sleef_artifact sleef)
