#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/Debug.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "tritoncpu-make-persistent-kernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_MAKEPERSISTENTKERNELPASS
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

static LogicalResult addPidSentinel(triton::FuncOp funcOp,
                                    unsigned blockIdxArgPos) {
  Block &entry = funcOp.getBody().front();
  Value blockIdx = entry.getArgument(blockIdxArgPos);

  OpBuilder b(&entry, entry.begin());
  Value blockIdOp = triton::cpu::CurrentBlockOp::create(
      b, funcOp.getLoc(), blockIdx.getType(), blockIdx);

  SmallVector<triton::GetProgramIdOp> pidOps;
  for (auto pidOp : entry.getOps<triton::GetProgramIdOp>()) {
    pidOps.push_back(pidOp);
  }
  for (auto pidOp : pidOps) {
    pidOp.replaceAllUsesWith(blockIdOp);
    pidOp.erase();
  }
  return success();
}

static constexpr StringLiteral kAttrSymName("sym_name");
static constexpr StringLiteral kAttrFuncType("function_type");
static constexpr StringLiteral kAttrSymVisibility("sym_visibility");
static constexpr StringLiteral kAttrArgAttrs("arg_attrs");
static constexpr StringLiteral kAttrResAttrs("res_attrs");
static constexpr StringLiteral kAttrNoinline("noinline");

// Collect op-level attributes to copy (exclude structural ones set by builder).
static SmallVector<NamedAttribute>
collectClonableOpAttrs(Operation *op, bool excludeNoInline = false) {
  SmallVector<NamedAttribute> out;
  for (NamedAttribute na : op->getAttrs()) {
    StringRef name = na.getName();
    if (name == kAttrSymName || name == kAttrFuncType ||
        name == kAttrSymVisibility || name == kAttrArgAttrs ||
        name == kAttrResAttrs)
      continue;
    if (excludeNoInline && name == kAttrNoinline)
      continue;
    out.push_back(na);
  }
  return out;
}

// Get (old) arg attrs as vector<DictionaryAttr>.
static SmallVector<DictionaryAttr> getArgAttrArray(Operation *op) {
  SmallVector<DictionaryAttr> v;
  if (auto arr = op->getAttrOfType<ArrayAttr>(kAttrArgAttrs)) {
    for (Attribute a : arr)
      v.push_back(a ? cast<DictionaryAttr>(a) : DictionaryAttr());
  } else {
    unsigned numArgs = cast<triton::FuncOp>(op).getNumArguments();
    v.resize(numArgs, DictionaryAttr());
  }
  return v;
}

// Clone tt.func -> tt.func(newName) with extra trailing i32 arg,
// preserving op attrs, arg attrs, res attrs, and body.
static triton::FuncOp cloneTTFuncWithExtraI32Arg(ModuleOp mod,
                                                 triton::FuncOp src,
                                                 StringRef newName) {
  MLIRContext *ctx = mod.getContext();
  OpBuilder b(mod.getBodyRegion());

  // New function type = old inputs + i32, same results.
  auto oldFTy = src.getFunctionType();
  SmallVector<Type> newInputs(oldFTy.getInputs().begin(),
                              oldFTy.getInputs().end());
  Type i32Ty = IntegerType::get(ctx, 32);
  newInputs.push_back(i32Ty);
  auto newFTy = FunctionType::get(ctx, newInputs, oldFTy.getResults());

  // Copy user attrs (excluding structural), arg attrs (+ empty for new arg),
  // res attrs.
  SmallVector<NamedAttribute> userAttrs =
      collectClonableOpAttrs(src, /*excludeNoInline=*/true);
  SmallVector<DictionaryAttr> argDicts = getArgAttrArray(src);
  argDicts.push_back(DictionaryAttr::get(ctx, {})); // placeholder for new arg

  // Create the new func with attrs/argAttrs passed via the builder.
  auto newFunc = triton::FuncOp::create(b, src.getLoc(), newName, newFTy,
                                        /*attrs=*/userAttrs,
                                        /*argAttrs=*/argDicts);
  newFunc.setPrivate();

  // Preserve result attrs (builder signature didn’t include res_attrs).
  if (auto resArr = src->getAttr(kAttrResAttrs))
    newFunc->setAttr(kAttrResAttrs, resArr);

  // Ensure new func has an entry block with the right arg count.
  newFunc.addEntryBlock(); // no-op if already present
  Block &oldEntry = src.getBody().front();
  Block &newEntry = newFunc.getBody().front();

  IRMapping map;
  // Map old BB args -> new BB args (1:1 for the original args).
  for (auto it : llvm::zip(
           oldEntry.getArguments(),
           newEntry.getArguments().take_front(oldEntry.getNumArguments())))
    map.map(std::get<0>(it), std::get<1>(it));

  // Clone ops.
  OpBuilder bodyBuilder(&newEntry, newEntry.begin());
  for (Operation &op : oldEntry.getOperations())
    bodyBuilder.clone(op, map);

  return newFunc;
}

static triton::FuncOp buildWrapper(ModuleOp mod, triton::FuncOp kernel,
                                   triton::FuncOp impl, StringRef name) {
  MLIRContext *ctx = mod.getContext();
  OpBuilder b(mod.getBodyRegion());

  Type i32Ty = IntegerType::get(ctx, 32);
  // use the original kernel to avoid pulling in the extra block param
  SmallVector<Type> wrapInputs(kernel.getArgumentTypes().begin(),
                               kernel.getArgumentTypes().end());

  auto wrapTy = FunctionType::get(ctx, wrapInputs, {});
  // Copy function-level user attrs from kernel
  SmallVector<NamedAttribute> userAttrs = collectClonableOpAttrs(kernel);
  SmallVector<DictionaryAttr> argDicts = getArgAttrArray(kernel);

  auto wrap = triton::FuncOp::create(b, kernel.getLoc(), name, wrapTy,
                                     userAttrs, argDicts);
  wrap->setAttr(SymbolTable::getVisibilityAttrName(),
                b.getStringAttr("public"));

  Block *entry = wrap.addEntryBlock();
  OpBuilder wb(entry, entry->end());

  Value bEnd = triton::cpu::BlockEndOp::create(wb, wrap.getLoc(), i32Ty);
  Value bStart = triton::cpu::BlockStartOp::create(wb, wrap.getLoc(), i32Ty);
  Value bStep = arith::ConstantOp::create(wb, wrap.getLoc(), i32Ty,
                                          wb.getIntegerAttr(i32Ty, 1));

  scf::ForOp forOp =
      scf::ForOp::create(wb, wrap.getLoc(), bStart, bEnd, bStep, ValueRange{});
  {
    Block *body = forOp.getBody();
    OpBuilder fb(body, body->begin());

    SmallVector<Value> callArgs;
    for (BlockArgument arg : wrap.getArguments()) {
      callArgs.push_back(arg);
    }
    callArgs.push_back(forOp.getInductionVar()); // add the block index offset

    // tt::CallOp can call tt.func by symbol (has FunctionType).
    triton::CallOp::create(fb, wrap.getLoc(), impl.getSymName(), TypeRange{},
                           callArgs);
  }

  triton::ReturnOp::create(wb, wrap.getLoc());
  return wrap;
}

} // namespace

struct MakePersistentKernelPass
    : public impl::MakePersistentKernelPassBase<MakePersistentKernelPass> {
  using MakePersistentKernelPassBase::MakePersistentKernelPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();
    OpBuilder b(ctx);

    SmallVector<triton::FuncOp, 4> kernels;
    for (auto funcOp : moduleOp.getOps<triton::FuncOp>()) {
      if (triton::isKernel(funcOp))
        kernels.push_back(funcOp);
    };
    assert(kernels.size() == 1 && "there should only be one kernel");
    LDBG("Adding kernel stream function wrapping " << kernels[0].getName());
    auto kernel = kernels[0];

    // 1. Clone the existing kernel, rename to `kernel`_impl, and add an i32
    // parameter which is the block index offset
    StringRef oldName = kernel.getName();
    std::string implName = (oldName + ".impl").str();
    triton::FuncOp implFunc =
        cloneTTFuncWithExtraI32Arg(moduleOp, kernel, implName);

    // 2. Rewrite the tt.get_program_id operation to add the block index offset
    // to the return value (for the impl kernel)
    unsigned blockIdxOffset = implFunc.getNumArguments() - 1;
    if (failed(addPidSentinel(implFunc, blockIdxOffset)))
      return signalPassFailure();

    // 3. Add the wrapper function calling kernel_impl in a loop over
    // block_start to block_end offsets (kernel function parameters)
    buildWrapper(moduleOp, kernel, implFunc, oldName);

    // 4. Erase the original kernel
    kernel.erase();
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
