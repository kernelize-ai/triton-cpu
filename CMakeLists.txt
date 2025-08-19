include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)
add_subdirectory(npu)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonNPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu.cc LINK_LIBS TritonNPUToLLVM)
  target_link_libraries(TritonNPU PRIVATE Python3::Module pybind11::headers)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()

# uKernel Libs

include(ExternalProject)

# Sleef

set(SLEEF_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/sleef-src")

# Sleef clones TLFloat as an external project which is problematic, as it will cache temp ninja paths and not get the new path on a re-configure. Delete the directory manually if it exists to force a re-install for each re-configure.
if(EXISTS "${SLEEF_SRC_DIR}/src/sleef/submodules/tlfloat")
  file(REMOVE_RECURSE "${SLEEF_SRC_DIR}/src/sleef/submodules/tlfloat")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps/sleef-build/ext_tlfloat-prefix")
endif()

 # TODO: override CMAKE_LIBRARY_OUTPUT_DIRECTORY

ExternalProject_Add(sleef
  PREFIX ${SLEEF_SRC_DIR}
  # Or if you want to pull from git directly:
  GIT_REPOSITORY https://github.com/shibatch/sleef.git
  GIT_TAG master

  CMAKE_ARGS
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/sleef-install
    -DCMAKE_BUILD_TYPE=Release
    -DSLEEF_BUILD_SHARED_LIBS=OFF
    -DSLEEF_BUILD_DFT=OFF
    -DSLEEF_BUILD_GNUABI_LIBS=OFF
    -DSLEEF_BUILD_TESTS=OFF
    -DSLEEF_BUILD_SCALAR_LIB=OFF
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}

  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/sleef-build

  # Control which steps happen
  UPDATE_DISCONNECTED 1    # don’t re-fetch every time
  INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/sleef-install
)
