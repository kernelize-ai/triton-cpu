#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TRITONCPUREWRITETENSORDESCARGS
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

// The frontend follows each tensor descriptor kernel argument with its shape
// (i32) and strides (i64). Pass the descriptor as its base pointer and rebuild
// it from those arguments, so that lowering the descriptor does not add a
// second set of shape and stride arguments.
void rewriteTensorDescArgs(triton::FuncOp funcOp) {
  Block &entry = funcOp.getBody().front();
  OpBuilder b(&entry, entry.begin());
  bool changed = false;
  for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
    BlockArgument arg = entry.getArgument(i);
    auto descTy = dyn_cast<triton::TensorDescType>(arg.getType());
    if (!descTy)
      continue;
    unsigned rank = descTy.getBlockType().getRank();
    if (i + 2 * rank >= e)
      continue;
    ValueRange shape = entry.getArguments().slice(i + 1, rank);
    ValueRange strides = entry.getArguments().slice(i + 1 + rank, rank);
    if (!llvm::all_of(shape.getTypes(),
                      [](Type ty) { return ty.isInteger(32); }) ||
        !llvm::all_of(strides.getTypes(),
                      [](Type ty) { return ty.isInteger(64); }))
      continue;

    arg.setType(
        triton::getPointerType(descTy.getSignlessBlockType().getElementType()));
    auto desc = triton::MakeTensorDescOp::create(
        b, arg.getLoc(), descTy, arg, shape, strides,
        triton::PaddingOption::PAD_ZERO);
    arg.replaceAllUsesExcept(desc, desc);
    changed = true;
  }
  if (changed)
    funcOp.setType(FunctionType::get(funcOp.getContext(),
                                     entry.getArgumentTypes(),
                                     funcOp.getResultTypes()));
}

} // namespace

struct TritonCPURewriteTensorDescArgs
    : public impl::TritonCPURewriteTensorDescArgsBase<
          TritonCPURewriteTensorDescArgs> {
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<triton::FuncOp>()) {
      if (triton::isKernel(funcOp))
        rewriteTensorDescArgs(funcOp);
    }
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
