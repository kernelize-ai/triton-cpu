import functools
import triton
import os
import tempfile
from pathlib import Path

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = []


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir]


class NpuUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NpuUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            # TODO: properly handle max number threads
            return (lib, fn_ptr_as_void_p, 0, 0, 128)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


class NPULauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        assert (False)


class NPUDriver(DriverBase):

    @staticmethod
    def is_active():
        try:
            import torch
            #return torch.cuda.is_available() and (torch.version.hip is None)
            return True
        except ImportError:
            return False

    def __init__(self):
        self.utils = NpuUtils()
        import torch
        self.get_current_stream = lambda idx: torch.cpu.Stream()
        self.launcher_cls = NPULauncher

    def get_current_device(self):
        import torch
        return torch.cpu.current_device()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        capability = "npu"
        warp_size = 32
        return GPUTarget("npu", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench
