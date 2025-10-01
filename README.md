# triton-cpu
CPU Plugin for Triton

### To install into Triton:

Clone the Triton CPU repository:
```
git clone git@github.com:kernelize-ai/triton-cpu.git
```
From the Triton repository, build Triton setting `TRITON_PLUGIN_DIRS='path/to/triton-cpu' during the build:
```
TRITON_PLUGIN_DIRS=triton-cpu pip install -e .
```
