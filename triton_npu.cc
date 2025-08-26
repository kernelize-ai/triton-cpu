// #include "TritonToTritonCPU/Passes.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "npu/include/TritonNPUToLLVM/Passes.h"

#include "mlir/Pass/PassManager.h"
// #include "mlir/Dialect/Vector/IR/VectorOps.h"
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

void init_triton_npu_passes_ttgpuir(py::module &&m) {
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createConvertTritonNPUToLLVMPass());
  });
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createAllocateSharedMemoryPass());
  });
}

void init_triton_npu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_npu_passes_ttgpuir(passes.def_submodule("ttnpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect>();
    // registry.insert<mlir::vector::VectorDialect>();
    // mlir::triton::cpu::registerTritonOpScalarizeExternalModels(registry);
    // mlir::registerAMXDialectTranslation(registry);
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
