#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritoncpu-outline-dot-microkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_OUTLINEDOTMICROKERNEL
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

static bool hasNestedDotOp(cpu::GenericOp genericOp) {
  WalkResult result =
      genericOp.walk([&](triton::DotOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

static std::string getMicrokernelFuncName(ModuleOp mod,
                                          cpu::GenericOp genericOp) {
  std::string base = "triton_cpu_dot_microkernel";
  if (auto parentFunc = genericOp->getParentOfType<FunctionOpInterface>())
    base = (parentFunc.getName() + "_dot_microkernel").str();

  if (!mod.lookupSymbol(base))
    return base;

  unsigned counter = 0;
  std::string name;
  do {
    name = (base + "_" + Twine(counter++)).str();
  } while (mod.lookupSymbol(name));
  return name;
}

struct OutlineDotMicrokernelPass
    : public impl::OutlineDotMicrokernelBase<OutlineDotMicrokernelPass> {
  using OutlineDotMicrokernelBase::OutlineDotMicrokernelBase;

  void runOnOperation() override {
    // TODO:
    /**
     *
     *

Compute the live-in set as every SSA operand of the generic op — its init, all
its ins, and its blocks sizes. Do this by walking the op's operands, not by
reading a known ins order. This is the genericity point made concrete: a naive
"read the ins list" would miss two live-ins here — %cst (the init) and %24 (the
padded-K block extent in blocks[%24, %c64, %c64]). Live-in-of-the-op analysis
catches all three categories uniformly, and it's the same analysis whether the
kernel passes (ptr, stride) pairs, a tensor descriptor, or a group index. Build
the function signature from the types of those live-ins; the result type is the
generic's result type, tensor<64x64xf32, #blocked>. Move the generic op into the
new function body, remapping its operands to the corresponding block arguments,
then func.return its result. The generic's interior — tile loop, addressing,
masks, loads, tt.dot — moves wholesale, unchanged. At the original site, replace
the generic with a func.call passing the live-ins, and RAUW the generic's result
(%25) with the call result so %26 consumes it transparently. Mark the function
noinline.
     *
     */

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    OpBuilder moduleBuilder(context);
    moduleBuilder.setInsertionPointToStart(mod.getBody());

    mod.walk([&](cpu::GenericOp genericOp) {
      if (!hasNestedDotOp(genericOp))
        return;

      auto parentFunc = genericOp->getParentOfType<triton::FuncOp>();
      assert(parentFunc && "expected generic op to have parent function");

      SmallVector<Type> argumentTypes;
      DenseMap<unsigned, Attribute> divisibilityMap;
      for (auto [i, operand] : llvm::enumerate(genericOp->getOperands())) {
        if (auto constantOp =
                dyn_cast_or_null<arith::ConstantOp>(operand.getDefiningOp()))
          continue;
        if (isa<PointerType>(operand.getType())) {
          auto blockArg = dyn_cast<BlockArgument>(operand);
          if (blockArg) {
            // get the divisibility from the parent func
            divisibilityMap[argumentTypes.size()] =
                parentFunc.getArgAttr(i, "tt.divisibility");
          } else {
            genericOp->emitWarning("Outlining pointer operand without known "
                                   "divisibility, performance may suffer");
          }
        }
        argumentTypes.push_back(operand.getType());
      }

      auto uKernelFuncTy = moduleBuilder.getFunctionType(
          argumentTypes, genericOp.getResultTypes());

      auto uKernelFunc = triton::FuncOp::create(
          moduleBuilder, genericOp.getLoc(),
          getMicrokernelFuncName(mod, genericOp), uKernelFuncTy);
      uKernelFunc.setVisibility(SymbolTable::Visibility::Private);
      uKernelFunc->setAttr("noinline", moduleBuilder.getBoolAttr(true));

      for (auto [index, attr] : divisibilityMap) {
        uKernelFunc.setArgAttr(index, "tt.divisibility", attr);
      }

      Block *entryBlock = uKernelFunc.addEntryBlock();
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);

      IRMapping mapping;
      // map generic arguments and clone constants first
      unsigned numConstantsSeen = 0;
      for (auto [i, operand] : llvm::enumerate(genericOp->getOperands())) {
        if (auto constantOp =
                dyn_cast_or_null<arith::ConstantOp>(operand.getDefiningOp())) {
          bodyBuilder.clone(*constantOp, mapping);
          numConstantsSeen++;
        } else {
          // subtract the number of cloned constants seen so far when converting
          // generic operand index to func arg index
          mapping.map(operand, entryBlock->getArgument(i - numConstantsSeen));
        }
      }

      auto newGeneric = bodyBuilder.clone(*genericOp, mapping);

      triton::ReturnOp::create(bodyBuilder, uKernelFunc.getLoc(),
                               newGeneric->getResults());
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineDotMicrokernelPass() {
  return std::make_unique<OutlineDotMicrokernelPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
