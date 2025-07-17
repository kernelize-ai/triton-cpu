from pdb import pm
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, npu

from dataclasses import dataclass
import functools
from typing import Dict
from types import ModuleType
import hashlib


@dataclass(frozen=True)
class NPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    backend_name: str = 'npu'
    sanitize_overflow: bool = True

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class NPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "npubin"

    def parse_options(self, options):
        args = {k: options[k] for k in NPUOptions.__dataclass_fields__.keys() if k in options if options[k] is not None}
        return NPUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            #metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        return dict()

    def get_module_map(self) -> Dict[str, ModuleType]:
        # TODO
        return {"triton.language.extra.libdevice": None}

    def load_dialects(self, ctx):
        npu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, "npu", 1, 1, 1)
        pm.run(mod) 
        return mod
    
    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # Triton -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # TODO: need triton to llvmir - can we do some simple convert triton to triton gpu?
        npu.passes.ttnpuir.add_to_llvmir(pm)

        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_asm(src, metadata, options):
        # TODO
        pass
        # return llvm.translate_to_host_asm(src, options.enable_fp_fusion, options.enable_fast_math)

    @staticmethod
    def make_library(src, metadata, options):
        # TODO
        pass

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            raise NotImplementedError("Gluon language support is not implemented for NPU backend")
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        #stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        #stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = 0.1
        return f'{version}-{self.target.arch}'
