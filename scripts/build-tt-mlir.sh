#!/usr/bin/env bash
set -euo pipefail

export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"
if [ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "TTMLIR toolchain directory `$TTMLIR_TOOLCHAIN_DIR` does not exist, create it before running this script."
    exit 1
fi
: "${LLVM_BUILD_DIR:?LLVM_BUILD_DIR must be set to the triton LLVM build root directory (typically in ~/.triton/llvm/llvm-OSDISTRO-ARCH)}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
# GitHub actions workflows pull triton into the triton-npu plugin directory root
TRITON_HOME="${TRITON_HOME:-"${REPO_ROOT}/triton"}"

TRITON_VENV_DIR="${TRITON_VENV_DIR:-"$REPO_ROOT/.venv"}"
# override TTMLIR venv with triton venv
export TTMLIR_VENV_DIR="$TRITON_VENV_DIR"
echo "Using tt-mlir venv dir: $TTMLIR_VENV_DIR"

if [ ! -x "$TTMLIR_VENV_DIR/bin/python" ]; then
    echo "No python interpreter found at $TTMLIR_VENV_DIR/bin/python; create the venv first."
    exit 1
fi

echo "Changing to tt-mlir directory"
cd "$REPO_ROOT/third_party/tt-mlir" || exit 1

export _ACTIVATE_ECHO_TOOLCHAIN_DIR_AND_EXIT=""
source env/activate

if ! python -c "import nanobind, sys; sys.exit(nanobind.__version__ != '2.12.0')" 2>/dev/null; then
    echo "Installing tt-mlir python dependencies"
    python -m pip install 'nanobind==2.12.0'
fi


LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib"
MLIR_DIR="$LLVM_LIBRARY_DIR/cmake/mlir"
LLVM_DIR="$LLVM_LIBRARY_DIR/cmake/llvm"
LLD_DIR="$LLVM_LIBRARY_DIR/cmake/lld"

mkdir -p "$TTMLIR_TOOLCHAIN_DIR/bin"
[ ! -f "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-ar" ] && ln -s "$LLVM_BUILD_DIR/bin/llvm-ar" "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-ar"
[ ! -f "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-ranlib" ] && ln -s "$LLVM_BUILD_DIR/bin/llvm-ranlib" "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-ranlib"
[ ! -f "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-tblgen" ] && ln -s "$LLVM_BUILD_DIR/bin/llvm-tblgen" "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-tblgen"

if [[ -z "${NO_TTMLIR_RUNTIME:-}" ]]; then
    echo "Building tt-mlir with runtime"
    # Quick hack for the RPATH issues encountered when running unit tests during build
    export LD_LIBRARY_PATH="$PWD/build/runtime/lib:$PWD/third_party/tt-metal/src/tt-metal/build/lib/:${LD_LIBRARY_PATH:-}"


    cmake -G Ninja -B build \
        -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" -DLLD_DIR="$LLD_DIR" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -DTTMLIR_ENABLE_OPMODEL=OFF \
        -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DTTMLIR_ENABLE_RUNTIME=ON \
        -DTT_RUNTIME_DEBUG=ON \
        -DTT_RUNTIME_ENABLE_PERF_TRACE=OFF \
        -DTT_RUNTIME_ENABLE_TTNN=ON \
        -DTT_RUNTIME_ENABLE_TTMETAL=ON \
        -DTTMLIR_ENABLE_RUNTIME_TESTS=ON \
        -DTTMLIR_ENABLE_ALCHEMIST=OFF \
        -DTTMLIR_ENABLE_SHARED_LIB=OFF \
        -DTT_USE_SYSTEM_SFPI=OFF
    cmake --build build
else
    echo "Building tt-mlir without runtime"
    # Hack: create the lib64 directory to avoid cmake error about missing dir on MacOS
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/lib64"

    cmake -G Ninja -B build \
        -DMLIR_DIR="$MLIR_DIR" \
        -DLLVM_DIR="$LLVM_DIR" \
        -DLLD_DIR="$LLD_DIR" \
        -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DTTMLIR_ENABLE_RUNTIME=OFF \
        -DTT_RUNTIME_ENABLE_TTNN=OFF \
        -DTT_RUNTIME_ENABLE_TTMETAL=OFF \
        -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF \
        -DTTMLIR_ENABLE_ALCHEMIST=OFF \
        -DTTMLIR_ENABLE_SHARED_LIB=OFF \
        -Dnanobind_DIR="$(python -c 'import os, nanobind; print(os.path.join(os.path.dirname(nanobind.__file__), "cmake"))')" \
        -Dpybind11_DIR="$(python -c 'import os, pybind11; print(os.path.join(os.path.dirname(pybind11.__file__), "share", "cmake", "pybind11"))')"
    cmake --build build
fi
