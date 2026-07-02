#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"
#include "npu/include/TritonNPUToD2M/Passes.h"
#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"

#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"
#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h" // TODO: remove if we drop the metal path
#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

std::string getDefaultTargerOrProcessTriple() {
  // Return process triple iff the default target triple is empty.
  std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.empty()) {
    // host
    triple = llvm::sys::getProcessTriple();
  }
  return triple;
}

void init_triton_cpu_passes(py::module &&m) {
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertTritonCPUToLLVMPass());
  });
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createAllocateSharedMemoryPass());
  });
  m.def("add_shared_memory_global_conversion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createSharedMemoryGlobalConversionPass());
  });
  m.def("add_masked_ops_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertMaskedOpsToLLVM());
  });
}

void init_triton_npu_passes_tenstorrent(py::module &&m) {
  m.def("add_core_specialize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createCoreSpecialize());
  });
  m.def("add_materialize_multicasts", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::npu::createTritonTenstorrentMaterializeMulticasts());
  });
  m.def("add_register_allocation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentRegAlloc());
  });
  m.def("remove_redundant_masks", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::npu::createTritonTenstorrentRemoveRedundantMasks());
  });
  m.def("add_convert_compute_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentConvertComputeOps());
  });
  m.def("add_tag_ios", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentTagInputOutputs());
  });
  m.def("add_accelerate_matmul", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentAccelerateMatmul());
  });
  m.def("add_remove_dot_load_layout_conversions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::
                   createTritonTenstorrentRemoveDotLoadLayoutConversions());
  });
  m.def("add_convert_tensor_desc", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::
                   createTritonTenstorrentConvertTensorDescToLoadStore());
  });
  m.def("add_canonicalize_matmul_loops", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::npu::createTritonTenstorrentCanonicalizeMatmulLoops());
  });
  m.def("add_to_ttkernel_dialect", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createConvertTritonNPUToTTKernel());
  });
  m.def("add_to_d2m_dialect", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createConvertTritonNPUToD2M());
  });

  m.def("add_drop_function",
        [](mlir::PassManager &pm, const std::string &funcName) {
          pm.addPass(mlir::triton::npu::createDropFunction({funcName}));
        });

  m.def("add_tensorize_scalars", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTensorizeScalars());
  });

  // tt-mlir specific passes
  m.def("add_ttkernel_control_dst_selection", [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::tt::ttkernel::createTTKernelControlDstSection());
  });
  m.def("add_ttkernel_device_zone_scopes", [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::tt::ttkernel::createTTKernelInsertDeviceZoneScopes());
  });

  // emit-c -- TODO should this be part of a different namespace?
  m.def("add_ttkernel_to_emitc", [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::tt::createConvertTTKernelToEmitC());
  });
  m.def("add_form_expressions_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::emitc::createFormExpressionsPass());
  });

  // tt-core
  m.def(
      "add_ttcore_register_device_pass",
      [](mlir::PassManager &pm, const std::string &systemDescPath) {
        mlir::tt::ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
        if (!systemDescPath.empty()) {
          registerDeviceOptions.systemDescPath = systemDescPath;
        }
        pm.addPass(mlir::tt::ttcore::createTTCoreRegisterDevicePass(
            registerDeviceOptions));
      });

  // ttkernel
  m.def("add_ttkernel_hoist_inits", [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::tt::ttkernel::createTTKernelHoistInits());
  });
}

void init_tenstorrent_d2m_passes(py::module &&m) {
  using namespace mlir::tt;

  // essentially a copy of
  // D2MPipelines::createD2MFrontendPipeline starting from the middle

  m.def("add_generic_fusion", [](mlir::PassManager &pm) {
    d2m::D2MGenericFusionOptions fusionOptions;
    pm.addPass(d2m::createD2MGenericFusion(fusionOptions));
  });
  m.def("add_allocate", [](mlir::PassManager &pm) {
    d2m::D2MAllocateOptions allocateOptions; // defaults
    pm.addPass(d2m::createD2MAllocate(allocateOptions));
  });
  m.def("add_lower_multicast_loads", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MLowerMulticastLoads());
  });
  m.def("add_lower_to_explicit_form", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MLowerToExplicitForm());
  });
  m.def("add_decompose_masking", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MDecomposeMasking());
  });
  m.def("add_decompose_arange", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MDecomposeArange());
  });
  m.def("add_generic_tile_compute_loops", [](mlir::PassManager &pm) {
    d2m::D2MGenericTileComputeLoopsOptions tileComputeLoopsOptions;
    {
      tileComputeLoopsOptions.maxDstPhysicalSizeTiles = 0; // unset
    }
    pm.addPass(d2m::createD2MGenericTileComputeLoops(tileComputeLoopsOptions));
  });
  m.def("add_linalg_to_affine", [](mlir::PassManager &pm) {
    d2m::D2MLinalgToAffineOptions linalgToAffineOptions;
    {
      linalgToAffineOptions.markRootLoops = true;
    }
    pm.addPass(d2m::createD2MLinalgToAffine(linalgToAffineOptions));
  });
  m.def("add_op_scheduler", [](mlir::PassManager &pm) {
    d2m::D2MOpSchedulerOptions opSchedulerOptions;
    {
      opSchedulerOptions.enableOpScheduler = true;
    }
    pm.addPass(d2m::createD2MOpScheduler(opSchedulerOptions));
  });
  m.def("add_insert_spill_and_scratch", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MInsertSpillAndScratch());
  });
  m.def("add_lower_scratch_allocate", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MLowerScratchAllocate());
  });
  m.def("add_insert_dst_register_access", [](mlir::PassManager &pm) {
    d2m::D2MInsertDstRegisterAccessUnscheduledOptions unschedDstOpts;
    {
      unschedDstOpts.maxDstPhysicalSizeTiles = 0; // unset
      unschedDstOpts.disableL1Acc = false;        // default from tt-mlir
    }
    pm.addPass(
        d2m::createD2MInsertDstRegisterAccessUnscheduled(unschedDstOpts));
    d2m::D2MInsertDstRegisterAccessScheduledOptions schedDstOpts;
    {
      schedDstOpts.maxDstPhysicalSizeTiles = 0; // unset
      schedDstOpts.disableL1Acc = false;        // default from tt-mlir
    }
    pm.addPass(d2m::createD2MInsertDstRegisterAccessScheduled(schedDstOpts));
  });
  m.def("add_sfpu_tile_loop_fission", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MSFPUTileLoopFission());
  });
  m.def("add_linearize_memref", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MGenericLinearizeMemref());
  });

  // Frontend of DMA lowering pipeline; insert compute-side CB
  // sync ops and split the unified thread into separate compute
  // and datamovement threads.
  m.def("add_hoist_cb_allocs", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MHoistCBAllocs());
  });
  m.def("add_split_unified_thread", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MSplitUnifiedThread());
  });

  // Backend of DMA lowering pipeline. generic ops are now
  // in split compute-dma form. All remote loads and stores
  // are lowered to concrete dma ops (dma_read, dma_write,
  // dma_wait, etc.) implementable by D2MToTTKernel lowering
  // pass.
  m.def("add_preallocate_mcast_semaphores", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MPreallocateMcastSemaphores());
  });
  m.def("add_schedule_dma",
        [](mlir::PassManager &pm) { pm.addPass(d2m::createD2MScheduleDMA()); });
  m.def("add_lower_load_store_ops_to_dma", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MLowerLoadStoreOpsToDMA());
  });
  m.def("add_optimize_dma",
        [](mlir::PassManager &pm) { pm.addPass(d2m::createD2MOptimizeDMA()); });
  m.def("add_expand_dma_read_composite_view", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MExpandDMAReadCompositeView());
  });
  m.def("add_lower_dma_to_fully_indexed_form", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MLowerDMAToFullyIndexedForm());
  });

  m.def("add_normalize_thread_args", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MNormalizeThreadArgs());
  });

  m.def("add_d2m_generic_regions_to_funcs", [](mlir::PassManager &pm) {
    pm.addPass(d2m::createD2MGenericRegionsToFuncs());
  });

  m.def("add_convert_d2m_to_ttkernel",
        [](mlir::PassManager &pm, bool ttnnMode) {
          d2m::ConvertD2MToTTKernelOptions D2MToTTKernelOptions;
          D2MToTTKernelOptions.ttnnMode = ttnnMode;

          mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
          funcPm.addPass(createConvertD2MToTTKernelPass(D2MToTTKernelOptions));
        });

  m.def("add_convert_d2m_to_ttmetal", [](mlir::PassManager &pm) {
    d2m::ConvertD2MToTTMetalOptions d2mToTTMetalOptions;
    // TODO: mathfidelity?
    pm.addPass(createConvertD2MToTTMetalPass(d2mToTTMetalOptions));
  });
  m.def("add_convert_d2m_to_ttnn", [](mlir::PassManager &pm) {
    d2m::ConvertD2MToTTNNOptions d2mToTTNNOptions;
    // TODO: mathfidelity?
    pm.addPass(createConvertD2MToTTNNPass(d2mToTTNNOptions));
  });
}

void init_triton_npu_passes_common(py::module &&m) {
  m.def("add_arith_int_range_opts", [](mlir::PassManager &pm) {
    pm.addPass(mlir::arith::createArithIntRangeOpts());
  });
  m.def("add_arith_expand", [](mlir::PassManager &pm) {
    pm.addPass(mlir::arith::createArithExpandOpsPass());
  });
  m.def("add_affine_licm", [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
  });
  m.def("add_lower_affine", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
  });
  m.def("add_fold_memref_alias_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  });
}

void init_triton_cpu_passes_ttgpuir(py::module &&m) {
  m.def("add_coalesce", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createTritonCPUCoalesce());
  });
  m.def("add_make_persistent_kernel", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createMakePersistentKernelPass());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // Triton to TritonGPU passes specific to the Triton CPU plugin
  init_triton_cpu_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  // TritonGPU to LLVM passes specific to the Triton CPU plugin
  init_triton_cpu_passes(passes.def_submodule("ttcpuir"));

  init_triton_npu_passes_tenstorrent(passes.def_submodule("tenstorrent"));
  init_tenstorrent_d2m_passes(passes.def_submodule("d2m"));
  init_triton_npu_passes_common(passes.def_submodule("common"));

  m.def(
      "translate_to_cpp",
      [](mlir::ModuleOp moduleOp, const std::string &symbolName) -> py::object {
        mlir::SymbolTable symbolTable(moduleOp);
        auto entry = symbolTable.lookup<mlir::func::FuncOp>(symbolName);
        assert(entry && "expected kernel func with given symbol name");
        assert(entry->hasAttr(mlir::tt::ttkernel::ThreadTypeAttr::name) &&
               "expected thread type attr on kernel func");

        std::string cppCode;
        llvm::raw_string_ostream os(cppCode);
        assert(mlir::succeeded(
                   mlir::tt::ttkernel::translateKernelFuncToCpp(entry, os)) &&
               "failed to translate kernel func to C++");
        return py::str(cppCode);
      });

  m.def("serialize_ttnn_to_flatbuffer", [](mlir::ModuleOp module) -> py::bytes {
    std::unordered_map<
        std::string, std::unordered_map<std::uint32_t, mlir::tt::GoldenTensor>>
        goldenMap;
    std::vector<std::pair<std::string, std::string>> moduleCache;

    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    if (mlir::failed(mlir::tt::ttnn::translateTTNNToFlatbuffer(
            module.getOperation(), os, goldenMap, moduleCache))) {
      throw std::runtime_error("Failed to serialize TTNN flatbuffer");
    }
    return py::bytes(buffer);
  });

  m.def("load_dialects",
        [](mlir::MLIRContext &context, const std::string &device) {
          mlir::DialectRegistry registry;
          registry.insert<mlir::triton::cpu::TritonCPUDialect>();

          if (device == "Tenstorrent") {
            // register tenstorrent dialects
            registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
          }

          context.appendDialectRegistry(registry);
          context.loadAllAvailableDialects();
        });

  m.def("get_default_target_triple",
        []() { return getDefaultTargerOrProcessTriple(); });

  m.def("get_processor_name",
        []() { return llvm::sys::getHostCPUName().str(); });

  m.def("attach_target_triple",
        [](llvm::Module *module, const std::string &triple) {
          module->setTargetTriple(llvm::Triple(triple));
        });
}
