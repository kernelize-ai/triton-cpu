from importlib.metadata import metadata
import tempfile
import functools
import hashlib
import re
import os
from typing import Tuple
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, cpu
from triton import knobs
from triton.runtime.build import _build
import triton.backends.cpu.driver as cpu_driver

from dataclasses import dataclass
from typing import Dict
from types import ModuleType


@dataclass(frozen=True)
class CPUOptions:
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    arch: str = None
    enable_fp_fusion: bool = True
    backend_name: str = 'tenstorrent'
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    matrix_instr_nonkdim: int = 32
    warp_size: int = 1
    min_dot_size: int = 1
    enable_dot_op_multicast: bool = False
    ttmlir_target: str = os.environ.get("TRITON_TTMLIR_TARGET", "ttkernel")

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    instrumentation = None  # TODO: intra-kernel instrumentation not yet supported

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "tenstorrent"

    def get_target_name(self, options) -> str:
        return "tenstorrent"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "flatbuffer" if os.environ.get("TRITON_TTMLIR_TARGET", "") == "d2m" else "cpp"
        self.device = 'Tenstorrent'

    def parse_options(self, options):
        args = {'arch': "BLACKHOLE"}
        if "enable_fp_fusion" not in options:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        if "enable_dot_op_multicast" not in options:
            args["enable_dot_op_multicast"] = os.environ.get("TRITON_ENABLE_DOT_OP_MULTICAST", "0") == "1"
        args.update(
            {k: options[k]
             for k in CPUOptions.__dataclass_fields__.keys()
             if k in options if options[k] is not None})

        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        # TODO
        return {"triton.language.extra.libdevice": None}

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx, self.device)

    @staticmethod
    def parse_attr(desc):
        ret = []
        if "D" in desc:
            ret += [["tt.divisibility", 8]]
        # pop D from desc
        desc = desc.replace("D", "")
        ret += BaseBackend.parse_attr(desc)
        return ret

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)

        cpu.passes.ttgpuir.add_make_persistent_kernel(pm)
        passes.common.add_inliner(pm)
        passes.ttir.add_triton_licm(pm)

        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        threads_per_warp = 1
        metadata["warp_size"] = threads_per_warp
        num_ctas = 1
        passes.ttir.add_convert_to_ttgpuir(pm, "cpu", options.num_warps, threads_per_warp, num_ctas)
        cpu.passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)

        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, 'make_ttgir')

        return mod

    def gluon_to_ttgir(self, src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        cpu.passes.ttgpuir.add_make_persistent_kernel(pm)
        passes.common.add_inliner(pm)

        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # Triton -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        cpu.passes.ttcpuir.add_allocate_shared_memory(pm)
        passes.convert.add_index_to_llvmir(pm)

        cpu.passes.ttcpuir.add_to_llvmir(pm)
        cpu.passes.ttcpuir.add_masked_ops_to_llvm(pm)
        cpu.passes.ttcpuir.add_shared_memory_global_conversion(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        cpu.attach_target_triple(llvm_mod, cpu.get_default_target_triple())
        target_features = ''
        llvm.attach_datalayout(llvm_mod, cpu.get_default_target_triple(), options.arch, target_features)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)
        metadata["shared"] = src.get_int_attr("ttg.shared")

        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_d2m(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        cpu.passes.tenstorrent.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        cpu.passes.tenstorrent.add_convert_compute_ops(pm)
        cpu.passes.tenstorrent.add_remove_dot_load_layout_conversions(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        cpu.passes.tenstorrent.remove_redundant_masks(pm)
        passes.common.add_canonicalizer(pm)

        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

        cpu.passes.tenstorrent.add_tag_ios(pm)

        sys_desc_path = os.getenv("TT_SYSTEM_DESC_PATH", "")
        cpu.passes.tenstorrent.add_ttcore_register_device_pass(pm, sys_desc_path)

        cpu.passes.tenstorrent.add_to_d2m_dialect(pm)

        # D2M pipeline from createTTIRToTTMetalMiddleendPipeline
        cpu.passes.d2m.add_generic_fusion(pm)
        passes.common.add_canonicalizer(pm)

        cpu.passes.d2m.add_allocate(pm)
        cpu.passes.d2m.add_lower_multicast_loads(pm)

        cpu.passes.d2m.add_lower_to_explicit_form(pm)
        passes.common.add_canonicalizer(pm)
        cpu.passes.d2m.add_decompose_masking(pm)
        cpu.passes.d2m.add_decompose_arange(pm)

        cpu.passes.d2m.add_generic_tile_compute_loops(pm)
        cpu.passes.d2m.add_linalg_to_affine(pm)
        cpu.passes.d2m.add_op_scheduler(pm)

        cpu.passes.d2m.add_insert_spill_and_scratch(pm)
        passes.common.add_canonicalizer(pm)
        cpu.passes.d2m.add_lower_scratch_allocate(pm)
        passes.common.add_canonicalizer(pm)

        cpu.passes.d2m.add_insert_dst_register_access(pm)
        cpu.passes.d2m.add_sfpu_tile_loop_fission(pm)

        cpu.passes.common.add_affine_licm(pm)

        cpu.passes.common.add_lower_affine(pm)
        cpu.passes.common.add_fold_memref_alias_ops(pm)
        cpu.passes.common.add_lower_affine(pm)
        cpu.passes.d2m.add_linearize_memref(pm)
        cpu.passes.common.add_lower_affine(pm)

        cpu.passes.d2m.add_hoist_cb_allocs(pm)
        cpu.passes.d2m.add_split_unified_thread(pm)

        cpu.passes.d2m.add_preallocate_mcast_semaphores(pm)
        cpu.passes.d2m.add_schedule_dma(pm)
        passes.common.add_canonicalizer(pm)
        cpu.passes.d2m.add_lower_load_store_ops_to_dma(pm)
        cpu.passes.d2m.add_optimize_dma(pm)
        cpu.passes.d2m.add_expand_dma_read_composite_view(pm)
        cpu.passes.d2m.add_lower_dma_to_fully_indexed_form(pm)

        cpu.passes.d2m.add_normalize_thread_args(pm)
        passes.common.add_canonicalizer(pm)
        cpu.passes.d2m.add_d2m_generic_regions_to_funcs(pm)

        pm.run(mod, "make_d2m")
        return mod

    @staticmethod
    def make_ttkernel(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # set TTNN mode to True
        cpu.passes.d2m.add_convert_d2m_to_ttkernel(pm, True)
        passes.common.add_canonicalizer(pm)
        cpu.passes.tenstorrent.add_ttkernel_control_dst_selection(pm)

        passes.common.add_canonicalizer(pm)
        passes.common.add_licm(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)

        cpu.passes.tenstorrent.add_ttkernel_hoist_inits(pm)

        pm.run(mod, "make_ttkernel")
        return mod

    @staticmethod
    def make_tenstorrent_mlir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        cpu.passes.tenstorrent.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        cpu.passes.tenstorrent.add_convert_compute_ops(pm)
        cpu.passes.tenstorrent.add_remove_dot_load_layout_conversions(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        cpu.passes.tenstorrent.remove_redundant_masks(pm)
        passes.common.add_canonicalizer(pm)

        cpu.passes.tenstorrent.add_core_specialize(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

        if options.enable_dot_op_multicast:
            cpu.passes.tenstorrent.add_materialize_multicasts(pm)
        cpu.passes.tenstorrent.add_register_allocation(pm)
        cpu.passes.tenstorrent.add_canonicalize_matmul_loops(pm)
        cpu.passes.tenstorrent.add_to_ttkernel_dialect(pm)
        passes.common.add_canonicalizer(pm)

        pm.run(mod, "make_ttmlir")
        return mod

    @staticmethod
    def make_emitc(mod, metadata, options, d2m=False):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # tt-mlir continued
        cpu.passes.tenstorrent.add_ttkernel_device_zone_scopes(pm)
        if not d2m:
            passes.common.add_canonicalizer(pm)
            passes.common.add_licm(pm)
            passes.common.add_sccp(pm)
            passes.common.add_cse(pm)

        if not d2m:
            cpu.passes.common.add_arith_int_range_opts(pm)
        cpu.passes.common.add_arith_expand(pm)
        if not d2m:
            sys_desc_path = os.getenv("TT_SYSTEM_DESC_PATH", "")
            cpu.passes.tenstorrent.add_ttcore_register_device_pass(pm, sys_desc_path)
        cpu.passes.tenstorrent.add_ttkernel_to_emitc(pm)
        passes.common.add_canonicalizer(pm)
        cpu.passes.tenstorrent.add_form_expressions_pass(pm)

        if d2m:
            cpu.passes.d2m.add_convert_d2m_to_ttnn(pm)
            cpu.passes.tenstorrent.add_tensorize_scalars(pm)

        pm.run(mod, "make_emit_c")
        return mod

    @staticmethod
    def make_ttmlir_cpp_file(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        pm.run(mod, "make_ttmlir_cpp_file")

        # find function names
        src = str(mod)
        names = re.findall(r"func.func public @(?!(?:barrier)\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 3
        compute_kernel = next((name for name in names if name.endswith("__compute")), None)
        reader_kernel = next((name for name in names if name.endswith("__reader")), None)
        writer_kernel = next((name for name in names if name.endswith("__writer")), None)
        assert compute_kernel is not None and reader_kernel is not None and writer_kernel is not None

        metadata["name"] = compute_kernel[:-9]  # remove __compute suffix
        metadata["shared"] = 0  # TODO: store cb sizes in module attributes?
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1

        cpp_file = "#ifdef COMPUTE_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, compute_kernel)
        cpp_file += "\n#endif  // COMPUTE_KERNEL\n\n"
        cpp_file += "\n#ifdef READER_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, reader_kernel)
        cpp_file += "\n#endif  // READER_KERNEL\n\n"
        cpp_file += "\n#ifdef WRITER_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, writer_kernel)
        cpp_file += "\n#endif  // WRITER_KERNEL\n"
        return cpp_file

    @staticmethod
    def make_flatbuffer(mod, metadata, options):
        metadata["shared"] = 0

        src = str(mod)
        names = re.findall(r"func.func @(?!(?:barrier)\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]

        fb = cpu.serialize_ttnn_to_flatbuffer(mod)
        return fb

    @staticmethod
    def make_d2m_cpp_file(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        pm.run(mod, "make_d2m_cpp_file")

        # find function names
        src = str(mod)
        names = re.findall(r"func.func private @(?!(?:barrier)\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 3
        compute_kernel = next((n for n in names if n.startswith("compute")), None)
        datamovement_kernels = sorted(n for n in names if n.startswith("datamovement"))
        assert compute_kernel is not None and len(datamovement_kernels) == 2

        metadata["name"] = compute_kernel[:-9]  # remove __compute suffix
        metadata["shared"] = 0  # TODO: store cb sizes in module attributes?
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1

        cpp_file = "#ifdef COMPUTE_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, compute_kernel)
        cpp_file += "\n#endif  // COMPUTE_KERNEL\n\n"
        cpp_file += "\n#ifdef READER_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, datamovement_kernels[0])
        cpp_file += "\n#endif  // READER_KERNEL\n\n"
        cpp_file += "\n#ifdef WRITER_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, datamovement_kernels[1])
        cpp_file += "\n#endif  // WRITER_KERNEL\n"
        return cpp_file

    @staticmethod
    def make_asm(src, metadata, options):
        names = re.findall(r"define void @(?!(?:barrier)\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]

        flags = []
        return llvm.translate_to_asm(src, cpu.get_default_target_triple(), options.arch, '', flags,
                                     options.enable_fp_fusion, False)

    @staticmethod
    def make_library(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = cpu_driver.library_dirs()
            libs = ["sleef", "cpu_utils"]  # TODO: conditionally include?
            include_dirs = []
            ccflags = []
            for lib_dir in lib_dirs:
                ccflags.extend(["-Xlinker", "-rpath", "-Xlinker", lib_dir])
            if cpu_driver.is_macos():
                ccflags.extend(["-undefined", "dynamic_lookup"])
            so = _build("kernel", asm_path, tmpdir, lib_dirs, include_dirs, libs, ccflags)
            with open(so, "rb") as f:
                return f.read()

    @staticmethod
    def make_tenstorrent_binary(src, metadata, options):
        # TODO: Implement the Tenstorrent binary generation
        pass

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            raise NotImplementedError("Gluon language support is not implemented for NPU backend")

        if self.device == 'CPU':
            stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
            stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
            stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)
        elif self.device == 'Tenstorrent':
            if options.ttmlir_target == "ttkernel":
                stages["ttmlir"] = lambda src, metadata: self.make_tenstorrent_mlir(src, metadata, options)
                stages["emitc"] = lambda src, metadata: self.make_emitc(src, metadata, options)
                stages["cpp"] = lambda src, metadata: self.make_ttmlir_cpp_file(src, metadata, options)
                stages['so'] = lambda src, metadata: self.make_tenstorrent_binary(src, metadata, options)
            elif options.ttmlir_target == "d2m":
                stages["d2m"] = lambda src, metadata: self.make_d2m(src, metadata, options)
                stages["ttkernel"] = lambda src, metadata: self.make_ttkernel(src, metadata, options)
                stages["ttnn"] = lambda src, metadata: self.make_emitc(src, metadata, options, d2m=True)
                stages["flatbuffer"] = lambda src, metadata: self.make_flatbuffer(src, metadata, options)
            else:
                raise ValueError(f"Unsupported TTMLIR target: {options.ttmlir_target}")

    @functools.lru_cache()
    def hash(self):
        version = 0.1
        return f'{version}-{self.target.arch}'
