# Triton NPU Tenstorrent Plugin Setup Guide


## System Dependencies

#### Ubuntu

```
sudo apt update
sudo apt install -y build-essential git vim python3 python3-venv python3-dev clang-19 libclang-cpp17-dev libgtest-dev python3-sphinx libnuma-dev numactl libhwloc-dev pkg-config
```

#### MacOS

_TODO_

## Tenstorrent Dependencies (`tt-mlir`)

The tenstorrent MLIR compiler (`tt-mlir`) is the entrypoint for Triton into the Tenstorrent software stack. `tt-mlir` is supplied inside the Triton Tenstorrent plugin source repo as a `third_party` git submodule. The provided `build-tt-mlir.sh` install script sets up the directory structure and components necessary to build and install `tt-mlir` in a location that Triton can access. The `build-tt-mlir.sh` script overrides the default LLVM/MLIR compiler version to use the Triton provided LLVM/MLIR sources.


Run the `build-tt-mlir.sh` script from the Triton NPU plugin root directory with the following parameters (replace e.g. OS architecture as appropriate):

#### Ubuntu

```
export CC=path/to/clang
export CXX=path/to/clang++
TRITON_VENV_DIR=triton/.venv/ LLVM_BUILD_DIR=$HOME/.triton/llvm/llvm-ubuntu-x86 TTMLIR_PYTHON_VERSION=python3.12 ./scripts/build-tt-mlir.sh
```

#### MacOS

```
NO_TTMLIR_RUNTIME=ON TRITON_VENV_DIR=triton/.venv/ LLVM_BUILD_DIR=$HOME/.triton/llvm/llvm-macos-arm64/ ./scripts/build-tt-mlir.sh
```


## Triton Install

_TODO_
