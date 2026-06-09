#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"

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

static unsigned getMaxVectorWidthBits(llvm::StringRef featureStr) {
  llvm::SubtargetFeatures features(featureStr);
  const auto &fv = features.getFeatures();
  auto has = [&](llvm::StringRef f) { return llvm::is_contained(fv, f); };

  if (has("+avx512f"))
    return 512; // AVX-512 foundation
  if (has("+avx"))
    return 256; // AVX/AVX2 (both set +avx)

  // safe minimum for any modern target
  return 128;
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

void init_triton_cpu_passes_ttgpuir(py::module &&m) {
  m.def(
      "add_accelerate_matmul",
      [](mlir::PassManager &pm, bool optimizeBlockLayout,
         bool canonicalizeKLoop) {
        mlir::triton::cpu::TritonCPUAccelerateMatmulOptions opts;
        opts.optimizeBlockLayout = optimizeBlockLayout;
        opts.canonicalizeKLoop = canonicalizeKLoop;
        pm.addPass(mlir::triton::cpu::createTritonCPUAccelerateMatmul(opts));
      },
      py::arg("pm"), py::arg("optimize_block_layout") = false,
      py::arg("canonicalize_k_loop") = false);
  m.def(
      "add_coalesce",
      [](mlir::PassManager &pm, int maxVectorWidth) {
        mlir::triton::cpu::TritonCPUCoalesceOptions opts;
        opts.MaxVectorWidth = maxVectorWidth;
        pm.addPass(mlir::triton::cpu::createTritonCPUCoalesce(opts));
      },
      py::arg("pm"), py::arg("max_vector_width"));
  m.def("add_make_persistent_kernel", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createMakePersistentKernelPass());
  });
  m.def("add_tile_and_fuse", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createTritonCPUTileAndFuse());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // Triton to TritonGPU passes specific to the Triton CPU plugin
  init_triton_cpu_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  // TritonGPU to LLVM passes specific to the Triton CPU plugin
  init_triton_cpu_passes(passes.def_submodule("ttcpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect>();

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("assemble_cpu", [](const std::string &assembly,
                           const std::string &tripleStr,
                           const std::string &arch,
                           const std::string &features) {
    std::string error;

    llvm::Triple triple(tripleStr);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
      throw std::runtime_error("target lookup error: " + error);

    llvm::SourceMgr srcMgr;
    srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                              llvm::SMLoc());

    const llvm::MCTargetOptions mcOptions;
    std::unique_ptr<llvm::MCRegisterInfo> mri(target->createMCRegInfo(triple));
    if (!mri)
      throw std::runtime_error(
          "assembler initialization error: failed to create MCRegisterInfo");
    std::unique_ptr<llvm::MCAsmInfo> mai(
        target->createMCAsmInfo(*mri, triple, mcOptions));
    if (!mai)
      throw std::runtime_error(
          "assembler initialization error: failed to create MCAsmInfo");
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(triple, arch, features));
    if (!sti)
      throw std::runtime_error(
          "assembler initialization error: failed to create MCSubtargetInfo");

    llvm::MCContext ctx(triple, *mai, *mri, *sti, &srcMgr);
    std::unique_ptr<llvm::MCObjectFileInfo> mofi(
        target->createMCObjectFileInfo(ctx, /*PIC=*/true,
                                       /*LargeCodeModel=*/false));
    ctx.setObjectFileInfo(mofi.get());

    llvm::SmallString<128> cwd;
    if (!llvm::sys::fs::current_path(cwd))
      ctx.setCompilationDir(cwd);

    llvm::SmallVector<char, 0> result;
    llvm::raw_svector_ostream svos(result);

    std::unique_ptr<llvm::MCStreamer> mcStreamer;
    std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

    std::unique_ptr<llvm::MCCodeEmitter> ce(
        target->createMCCodeEmitter(*mcii, ctx));
    std::unique_ptr<llvm::MCAsmBackend> mab(
        target->createMCAsmBackend(*sti, *mri, mcOptions));
    std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
    mcStreamer.reset(target->createMCObjectStreamer(
        triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

    std::unique_ptr<llvm::MCAsmParser> parser(
        createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
    std::unique_ptr<llvm::MCTargetAsmParser> tap(
        target->createMCAsmParser(*sti, *parser, *mcii));
    if (!tap)
      throw std::runtime_error("assembler initialization error");

    parser->setTargetParser(*tap);
    if (parser->Run(/*NoInitialTextSection=*/false))
      throw std::runtime_error("assembly failed");

    return py::bytes(std::string(result.begin(), result.end()));
  });

  m.def("get_default_target_triple",
        []() { return getDefaultTargerOrProcessTriple(); });

  m.def("get_processor_name",
        []() { return llvm::sys::getHostCPUName().str(); });

  m.def("get_processor_features", []() {
    std::string features;
    for (auto [i, F] : llvm::enumerate(llvm::sys::getHostCPUFeatures())) {
      if (i > 0)
        features += ",";
      features += (F.second ? "+" : "-") + F.first().str();
    }
    return features;
  });

  m.def("attach_target_triple",
        [](llvm::Module *module, const std::string &triple) {
          module->setTargetTriple(llvm::Triple(triple));
        });

  m.def("get_max_vector_width_bits", [](const std::string &featureStr) {
    return getMaxVectorWidthBits(featureStr);
  });
}
