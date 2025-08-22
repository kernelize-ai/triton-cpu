from importlib.metadata import metadata
import tempfile
import functools
import hashlib
import re
import os
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, npu
from triton import knobs
from triton.runtime.build import _build
import triton.backends.npu.driver as npu_driver

from dataclasses import dataclass
from typing import Dict
from types import ModuleType


@dataclass(frozen=True)
class NPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    arch: str = None
    enable_fp_fusion: bool = True
    backend_name: str = 'npu'
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class NPUBackend(BaseBackend):
    instrumentation = None  # TODO: intra-kernel instrumentation not yet supported

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "so"

    def parse_options(self, options):
        args = {'arch': npu.get_processor_name()}
        if "enable_fp_fusion" not in options:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args.update(
            {k: options[k]
             for k in NPUOptions.__dataclass_fields__.keys()
             if k in options if options[k] is not None})

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
        num_warps = 2
        threads_per_warp = 1
        num_ctas = 1
        passes.ttir.add_convert_to_ttgpuir(pm, "npu", num_warps, threads_per_warp, num_ctas)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # Triton -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        npu.passes.ttnpuir.add_allocate_shared_memory(pm)
        npu.passes.ttnpuir.add_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        npu.attach_target_triple(llvm_mod, npu.get_default_target_triple())
        target_features = ''
        llvm.attach_datalayout(llvm_mod, npu.get_default_target_triple(), options.arch, target_features)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        return str(llvm_mod)

    @staticmethod
    def make_asm(src, metadata, options):
        names = re.findall(r"define void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        metadata["shared"] = 1

        flags = []
        return llvm.translate_to_asm(src, npu.get_default_target_triple(), options.arch, '', flags,
                                     options.enable_fp_fusion, False)

    @staticmethod
    def make_library(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = npu_driver.library_dirs()
            libs = ["sleef"]  # TODO: conditionally include?
            include_dirs = []
            so = _build("kernel", asm_path, tmpdir, lib_dirs, include_dirs, libs, [])
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            raise NotImplementedError("Gluon language support is not implemented for NPU backend")
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = 0.1
        return f'{version}-{self.target.arch}'
