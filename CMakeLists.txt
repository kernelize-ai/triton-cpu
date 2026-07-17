include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)

set(triton_cpu_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(utils)

set(CPU_LLVM_LIBS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR # Linux arm64
    CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR # macOS arm64
    CMAKE_OSX_ARCHITECTURES MATCHES "arm64")  # also macOS arm64
    list(APPEND CPU_LLVM_LIBS
        LLVMAArch64CodeGen
        LLVMAArch64AsmParser
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    list(APPEND CPU_LLVM_LIBS
        LLVMX86CodeGen
        LLVMX86AsmParser
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
    list(APPEND CPU_LLVM_LIBS
      LLVMPowerPCAsmParser
      LLVMPowerPCCodeGen
    )
else()
  message(FATAL_ERROR "LLVM codegen/ASM parser libs: This HW architecture (${CMAKE_SYSTEM_PROCESSOR}) is not configured in cmake lib dependencies.")
endif()

set(ARM_SME_LIBS "")
list(APPEND ARM_SME_LIBS
  MLIRArmSMEToSCF
  MLIRArmSMEToLLVM
  MLIRArmSMEToLLVMIRTranslation
  MLIRVectorToLLVMPass
)

add_subdirectory(cpu)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonCPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_cpu.cc LINK_LIBS TritonCPUToLLVM)
  target_link_libraries(TritonCPU PRIVATE ${CPU_LLVM_LIBS} ${ARM_SME_LIBS} Python3::Module nanobind-static)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()

add_subdirectory(test)

# uKernel Libs

option(BUILD_SLEEF "Build SLEEF library" OFF)
if(BUILD_SLEEF)
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

  # Try lib64 first, fall back to lib.
  if(EXISTS ${SLEEF_INSTALL_DIR}/lib64/libsleef.a)
    set(SLEEF_LIB_PATH ${SLEEF_INSTALL_DIR}/lib64/libsleef.a)
  else()
    set(SLEEF_LIB_PATH ${SLEEF_INSTALL_DIR}/lib/libsleef.a)
  endif()

  # Add a custom step to copy libsleef.a after install
  ExternalProject_Add_Step(sleef copy_lib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E copy ${SLEEF_LIB_PATH} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    DEPENDEES install
    ALWAYS 1
  )

  # So you can depend on a phony target to ensure the copy happens
  add_custom_target(sleef_artifact ALL DEPENDS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(sleef_artifact sleef)
endif()
