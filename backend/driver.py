import functools
import triton
import os
import subprocess
import tempfile
import time
import platform
import importlib
from pathlib import Path

from triton.runtime.build import compile_module_from_src
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()


@functools.lru_cache()
def is_macos():
    return platform.system() == "Darwin"


@functools.lru_cache()
def external_openmp_path():
    return os.environ.get("TRITON_LOCAL_LIBOMP_PATH", "/opt/homebrew/opt/libomp/")


dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")
                ] + [os.path.join(external_openmp_path(), "include") if is_macos() else []]
libdevice_dir = os.path.join(dirname, "lib")
libraries = []


@functools.lru_cache()
def system_ccflags():
    ccflags = []
    if is_macos():
        ccflags.extend(["-undefined", "dynamic_lookup", "-Xclang", "-fopenmp"])
    else:
        ccflags.extend(["-fopenmp"])
    return ccflags


@functools.lru_cache()
def library_dirs():
    lib_dirs = [_triton_C_dir]
    if is_macos():
        lib_dirs.extend([os.path.join(external_openmp_path(), "lib")])
    return lib_dirs


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
            os.fsync(f.fileno())
            os.stat(f.name)
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            # TODO: properly handle num registers / max number threads
            return (lib, fn_ptr_as_void_p, 1, 0, 2**12)

    def get_device_properties(self, *args):
        return {"max_num_regs": 1000, "max_shared_mem": 8192, "multiprocessor_count": 1, "warpSize": 2}


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
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature):

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        if ty.startswith("tensordesc"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")
    arg_types = ', '.join(f"{ty_to_cpp(ty)}" for ty in signature.values() if ty != "constexpr")

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    # TODO: float_storage_decls?
    kernel_params = [f"arg{i}" for i, ty in signature.items() if ty != "constexpr"]

    has_spmd_args = True
    if has_spmd_args:
        kernel_params.extend(["coord.x", "coord.y", "coord.z", "gridX", "gridY", "gridZ"])
        arg_types += ', '
        arg_types += ', '.join(["int32_t", "int32_t", "int32_t", "int32_t", "int32_t", "int32_t"])

    src = f"""
#include <stdbool.h>
#include <Python.h>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

typedef void(*kernel_ptr_t)({arg_types});

typedef struct _GridCoordinate {{
    int x;
    int y;
    int z;
}} GridCoordinate;

static GridCoordinate get_grid_coordinate(int idx, int gridX, int gridY, int gridZ) {{
    GridCoordinate coord;
    coord.z = idx / (gridX * gridY);
    coord.y = (idx % (gridX * gridY)) / gridX;
    coord.x = (idx % gridX);
    return coord;
}}

static void _launch(int gridX, int gridY, int gridZ, kernel_ptr_t kernel_ptr{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
    size_t N = gridX * gridY * gridZ;
    if (N == 1) {{
      GridCoordinate coord = get_grid_coordinate(0, gridX, gridY, gridZ);
      (*kernel_ptr)({', '.join(kernel_params) if len(kernel_params) > 0 else ''});
      return;
    }}

    int maxThreads = 1;
#ifdef _OPENMP
    const int ompMaxThreads = omp_get_max_threads();
    maxThreads = N < ompMaxThreads ? N : ompMaxThreads;
#endif // _OPENMP

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(maxThreads)
#endif // _OPENMP
 for (size_t i = 0; i < N; ++i) {{
    GridCoordinate coord = get_grid_coordinate(i, gridX, gridY, gridZ);
    (*kernel_ptr)({', '.join(kernel_params) if len(kernel_params) > 0 else ''});
 }}


}}

typedef struct _DevicePtrInfo {{
    void* dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  PyErr_Print();
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    // TODO: validate the ptr?
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}

  PyErr_Format(PyExc_TypeError, "Pointer argument (at %d) must be either uint64 or have data_ptr method", idx);
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  void* _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *global_scratch_obj = NULL; // UNUSED in NPU backend
  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}

  kernel_ptr_t kernel_ptr = (kernel_ptr_t)(_function);

   // Extract num_threads metadata.
  int num_threads = 0;
  if (PyObject_HasAttrString(kernel_metadata, "num_threads")) {{
    PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_threads");
    assert(PyLong_Check(num_threads_attr));
    num_threads = PyLong_AsLong(num_threads_attr);
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {newline.join(ptr_decls)}
  _launch(gridX, gridY, gridZ, kernel_ptr{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class NPULauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature)
        mod = compile_module_from_src(src, name="__triton_launcher", library_dirs=library_dirs(),
                                      include_dirs=include_dirs, libraries=libraries, ccflags=system_ccflags())
        self.launch = mod.launch

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        self.launch(gridX, gridY, gridZ, stream, function, *args)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


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

    def get_device_interface(self):
        return CPUDeviceInterface()

    def get_current_device(self):
        return 0

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

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    def clear_cache(self, cache):
        cache.zero_()
