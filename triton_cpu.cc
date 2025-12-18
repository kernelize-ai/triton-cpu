#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"

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

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect>();

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
