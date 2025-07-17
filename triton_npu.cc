// #include "TritonToTritonCPU/Passes.h"

// #include "triton/Dialect/TritonCPU/IR/Dialect.h" // TODO: do we need this? 

#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

void init_triton_npu_passes_ttgpuir(py::module &&m) {
#if 0
    m.def("add_to_llvmir",
    [](mlir::PassManager &pm, int32_t capability, int32_t ptxVersion) {
        pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass());
    });
#endif
}

void init_triton_npu(py::module &&m) {
    auto passes = m.def_submodule("passes");

    m.def("load_dialects", [](mlir::MLIRContext &context) {
        mlir::DialectRegistry registry;
        registry.insert<mlir::vector::VectorDialect>();
        // mlir::triton::cpu::registerTritonOpScalarizeExternalModels(registry);
        // mlir::registerAMXDialectTranslation(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    });

}
