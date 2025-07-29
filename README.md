# triton-npu
NPU Plugin for Triton

### To install into Triton:

Clone the Triton NPU repository:
```
git clone git@github.com:kernelize-ai/triton-npu.git
```
From the Triton repository, build Triton setting `TRITON_PLUGIN_DIRS='path/to/triton-npu' during the build:
```
TRITON_PLUGIN_DIRS=triton-npu pip install -e .
```
