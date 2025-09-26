#include "npu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Utility.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool _isKernel(LLVM::LLVMFuncOp f) {
  // TODO: maybe we should set an attribute on the function during FuncOpToLLVM
  // lowering instead?
  return f.getLinkage() == LLVM::Linkage::External;
}

} // namespace

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_GENERATEKERNELWRAPPER
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"

class GenerateKernelWrapper
    : public impl::GenerateKernelWrapperBase<GenerateKernelWrapper> {

public:
  using impl::GenerateKernelWrapperBase<
      GenerateKernelWrapper>::GenerateKernelWrapperBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SymbolTable symTab(mod);
    LLVM::LLVMFuncOp kernel = nullptr;
    for (auto f : mod.getOps<LLVM::LLVMFuncOp>()) {
      // llvm::errs() << "f = " << f.getName() << ", isPublic " << f.isPublic()
      // << ", linkage " << f.getLinkage() << "\n";
      if (_isKernel(f)) {
        kernel = f;
        break;
      }
    }
    assert(kernel && "expected kernel function in module");

    OpBuilder b(mod.getContext());

    // rename original kernel
    StringRef oldName = kernel.getName();
    std::string implName = (oldName + ".impl").str();
    SymbolTable::setSymbolName(kernel, implName);
    kernel.setLinkage(LLVM::Linkage::Internal);

    // create wrapper function
    auto oldFnTy = kernel.getFunctionType();
    SmallVector<Type> argTys(oldFnTy.getParams().begin(),
                             oldFnTy.getParams().end());
    argTys.push_back(b.getIntegerType(32)); // start block
    argTys.push_back(b.getIntegerType(32)); // end block

    auto wrapperFnTy =
        LLVM::LLVMFunctionType::get(oldFnTy.getReturnType(), argTys,
                                    /*isVarArg=*/false);

    b.setInsertionPoint(kernel);
    auto wrapperFunc =
        b.create<LLVM::LLVMFuncOp>(mod.getLoc(), oldName, wrapperFnTy);
    wrapperFunc.setLinkage(LLVM::Linkage::External);

    // create the wrapper function body
    Block *entry = wrapperFunc.addEntryBlock(b);

    auto args = entry->getArguments();
    SmallVector<Value> origArgs(args.begin(), args.end() - 2);

    auto *header = b.createBlock(&wrapperFunc.getBody());
    auto *body = b.createBlock(&wrapperFunc.getBody());
    body->addArgument(b.getIntegerType(32), wrapperFunc.getLoc());
    auto *exit = b.createBlock(&wrapperFunc.getBody());

    b.setInsertionPointToStart(entry);

    Value start = args[args.size() - 2];
    Value end = args.back();
    b.create<cf::BranchOp>(wrapperFunc.getLoc(), header, ValueRange{start});

    // header: initialize loop index
    b.setInsertionPointToStart(header);
    Value i = header->addArgument(b.getIntegerType(32), wrapperFunc.getLoc());
    Value cond = b.create<LLVM::ICmpOp>(wrapperFunc.getLoc(),
                                        LLVM::ICmpPredicate::slt, i, end);
    b.create<cf::CondBranchOp>(wrapperFunc.getLoc(), cond, body, ValueRange{i},
                               exit, ValueRange{});

    // body: call impl, increment loop index
    b.setInsertionPointToStart(body);
    // replace the block.x, block.y, and block.z values with the loop index. for
    // now just the first one, later all 3
    auto &blockX = origArgs[origArgs.size() + kProgramIdArgsOffset];
    blockX = body->getArgument(0);
    b.create<LLVM::CallOp>(wrapperFunc.getLoc(), TypeRange{},
                           FlatSymbolRefAttr::get(mod.getContext(), implName),
                           origArgs);
    Value one = b.create<LLVM::ConstantOp>(wrapperFunc.getLoc(),
                                           b.getIntegerType(32), 1);
    Value next =
        b.create<LLVM::AddOp>(wrapperFunc.getLoc(), body->getArgument(0), one);
    b.create<cf::BranchOp>(wrapperFunc.getLoc(), header, ValueRange{next});

    // exit: return void
    b.setInsertionPointToStart(exit);
    if (isa<LLVM::LLVMVoidType>(oldFnTy.getReturnType()))
      b.create<LLVM::ReturnOp>(wrapperFunc.getLoc(), ValueRange{});
    else
      wrapperFunc.emitError("Wrapper for non-void kernels not implemented.");
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
