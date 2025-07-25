# triton-npu
NPU Plugin for Triton

### To install into Triton: 

```
cd third_party/
git submodule add git@github.com:kernelize-ai/triton-npu.git npu
```
then modify `setup.py` to add the plugin into local backends (`BackendInstaller.copy(["nvidia", "amd", "npu"])`), or set `TRITON_PLUGIN_DIRS='third_party/npu'`. 
