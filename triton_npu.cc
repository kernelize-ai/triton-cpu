// #include "TritonToTritonCPU/Passes.h"

// #include "triton/Dialect/TritonCPU/IR/Dialect.h" // TODO: do we need this?
#include "TritonNPUToLLVM/Passes.h"

#include "mlir/Pass/PassManager.h"
// #include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

void init_triton_npu_passes_ttgpuir(py::module &&m) {
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm) {
          pm.addPass(mlir::triton::createConvertTritonNPUToLLVMPass());
        });
}

void init_triton_npu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_npu_passes_ttgpuir(passes.def_submodule("ttnpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    // registry.insert<mlir::vector::VectorDialect>();
    // mlir::triton::cpu::registerTritonOpScalarizeExternalModels(registry);
    // mlir::registerAMXDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
