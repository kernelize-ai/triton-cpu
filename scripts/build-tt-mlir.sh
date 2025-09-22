#!/usr/bin/env bash
set -euo pipefail

TTMLIR_TOOLCHAIN_DIR="/opt/ttmlir-toolchain/"
export TTMLIR_PYTHON_VERSION="${TTMLIR_PYTHON_VERSION:-python3.11}"

: "${LLVM_BUILD_DIR:?LLVM_BUILD_DIR must be set to the triton LLVM build root directory (typically in ~/.triton/llvm/llvm-OSDISTRO-ARCH)}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
# GitHub actions workflows pull triton into the triton-npu plugin directory root
TRITON_HOME="${TRITON_HOME:-"${REPO_ROOT}/triton"}"

TRITON_VENV_DIR="${TRITON_VENV_DIR:-"$REPO_ROOT/.venv"}"
# override TTMLIR venv with triton venv
export TTMLIR_VENV_DIR="$TRITON_VENV_DIR"

echo "Changing to tt-mlir directory"
cd "$REPO_ROOT/third_party/tt-mlir" || exit 1

echo "Building tt-mlir env"
cmake -B env/build env -DTTMLIR_BUILD_LLVM=OFF
cmake --build env/build
source env/activate

echo "Installing tt-mlir python dependencies"
python -m pip install nanobind

#export LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include"
#export MLIR_INCLUDE_DIRS="$LLVM_BUILD_DIR/include"
LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib"
#export LLVM_SYSPATH="$LLVM_BUILD_DIR"
MLIR_DIR="$LLVM_LIBRARY_DIR/cmake/mlir"
LLVM_DIR="$LLVM_LIBRARY_DIR/cmake/llvm"

ln -s "$LLVM_BUILD_DIR/bin/llvm-ar" "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-ar"
ln -s "$LLVM_BUILD_DIR/bin/llvm-ranlib" "$TTMLIR_TOOLCHAIN_DIR/bin/llvm-ranlib"

if [[ -z "${NO_TTMLIR_RUNTIME:-}" ]]; then
    echo "Building tt-mlir with runtime"
    cmake -G Ninja -B build -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" \
    -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF -DTTMLIR_ENABLE_RUNTIME=ON -DTT_RUNTIME_ENABLE_TTNN=ON -DTT_RUNTIME_ENABLE_TTMETAL=ON -DTTMLIR_ENABLE_RUNTIME_TESTS=ON

    cmake --build build
    cmake --build build -- ttrt
else
    echo "Building tt-mlir without runtime"
    cmake -G Ninja -B build -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" \
    -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF -DTTMLIR_ENABLE_RUNTIME=OFF -DTT_RUNTIME_ENABLE_TTNN=OFF -DTT_RUNTIME_ENABLE_TTMETAL=OFF -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF
    cmake --build build
fi
