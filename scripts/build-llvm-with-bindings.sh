#!/usr/bin/env bash
# Builds LLVM/MLIR with MLIR python bindings enabled, for use as LLVM_BUILD_DIR
# in scripts/build-tt-mlir.sh. tt-mlir's python-binding tools (ttrt, builder,
# golden, ...) are gated on MLIR_ENABLE_BINDINGS_PYTHON, which is OFF in the
# LLVM that triton ships, so we need our own.
#
# Mirrors triton/.github/workflows/llvm-build.yml step
#   "Configure, Build, Test, and Install LLVM (Ubuntu and macOS x64)"
# with -DMLIR_ENABLE_BINDINGS_PYTHON=ON. The LLVM pin tracks triton/cmake/llvm-hash.txt.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRITON_SRC_DIR="${TRITON_SRC_DIR:-"$REPO_ROOT/../triton"}"

if [ ! -f "$TRITON_SRC_DIR/cmake/llvm-hash.txt" ]; then
    echo "Expected $TRITON_SRC_DIR/cmake/llvm-hash.txt; set TRITON_SRC_DIR to your triton checkout."
    exit 1
fi

LLVM_SHA="$(cat "$TRITON_SRC_DIR/cmake/llvm-hash.txt")"
SHORT_SHA="${LLVM_SHA:0:8}"

LLVM_WORK_DIR="${LLVM_WORK_DIR:-"$HOME/.triton/llvm-with-python-bindings"}"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-"$LLVM_WORK_DIR/llvm-project"}"
LLVM_BUILD_TREE="${LLVM_BUILD_TREE:-"$LLVM_WORK_DIR/build"}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-"$LLVM_WORK_DIR/llvm-${SHORT_SHA}-ubuntu-x64"}"

TRITON_VENV_DIR="${TRITON_VENV_DIR:-"$TRITON_SRC_DIR/.venv"}"
if [ ! -x "$TRITON_VENV_DIR/bin/python" ]; then
    echo "No python interpreter found at $TRITON_VENV_DIR/bin/python; create the venv first."
    exit 1
fi
PYTHON="$TRITON_VENV_DIR/bin/python"

echo "LLVM SHA:      $LLVM_SHA"
echo "Source dir:    $LLVM_SRC_DIR"
echo "Build dir:     $LLVM_BUILD_TREE"
echo "Install dir:   $LLVM_INSTALL_DIR"
echo "Python:        $PYTHON ($("$PYTHON" --version))"

mkdir -p "$LLVM_WORK_DIR"

if [ ! -d "$LLVM_SRC_DIR/.git" ]; then
    echo "Cloning llvm-project..."
    git clone https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
fi

echo "Checking out $LLVM_SHA..."
git -C "$LLVM_SRC_DIR" fetch origin
git -C "$LLVM_SRC_DIR" checkout "$LLVM_SHA"

echo "Installing MLIR python requirements..."
"$PYTHON" -m pip install -r "$LLVM_SRC_DIR/mlir/python/requirements.txt"

SCCACHE_FLAGS=()
if command -v sccache >/dev/null 2>&1; then
    echo "sccache found, enabling compiler caching"
    SCCACHE_FLAGS+=(-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache)
fi

cmake -G Ninja -S "$LLVM_SRC_DIR/llvm" -B "$LLVM_BUILD_TREE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    "${SCCACHE_FLAGS[@]}" \
    -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
    -DCMAKE_LINKER=lld \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;lld" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DNB_USE_SUBMODULE_DEPS=ON

cmake --build "$LLVM_BUILD_TREE" --target install

echo
echo "Done. Install prefix: $LLVM_INSTALL_DIR"
echo
echo "To rebuild triton against this LLVM (tt-mlir and triton-npu will pick it up"
echo "transitively), per triton/README.md:"
echo
echo "  export LLVM_BUILD_DIR=\"$LLVM_INSTALL_DIR\""
echo "  cd \"$TRITON_SRC_DIR\""
echo "  LLVM_INCLUDE_DIRS=\$LLVM_BUILD_DIR/include \\"
echo "    LLVM_LIBRARY_DIR=\$LLVM_BUILD_DIR/lib \\"
echo "    LLVM_SYSPATH=\$LLVM_BUILD_DIR \\"
echo "    pip install -e ."
