#include "npu/include/Dialect/TritonCPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritoncpu-add-kernel-stream"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_ADDKERNELSTREAMPASS
#include "npu/include/Dialect/TritonCPU/Transforms/Passes.h.inc"

namespace {

/// Replace `tt.get_program_id x` result with (pid + block_index_offset).
/// Assumes the cloned kernel has an extra last i32 argument.
static LogicalResult offsetPidByBlockIndex(triton::FuncOp funcOp,
                                           unsigned blockIdxArgPos) {
  Block &entry = funcOp.getBody().front();
  Value blockIdx = entry.getArgument(blockIdxArgPos);

  auto programIdOps = llvm::to_vector(funcOp.getOps<triton::GetProgramIdOp>());
  assert(programIdOps.size() == 1 &&
         "expected exactly one tt.get_program_id op");

  OpBuilder b(programIdOps[0]->getNextNode());
  Value blockOffset = b.create<arith::AddIOp>(
      programIdOps[0]->getLoc(), programIdOps[0].getResult(), blockIdx);
  programIdOps[0].getResult().replaceUsesWithIf(
      blockOffset, [&](OpOperand &operand) {
        return operand.getOwner() != blockOffset.getDefiningOp();
      });

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
  auto newFunc = b.create<triton::FuncOp>(src.getLoc(), newName, newFTy,
                                          /*attrs=*/userAttrs,
                                          /*argAttrs=*/argDicts);

  // Preserve visibility (or make it private if you want to hide the impl).
  if (auto vis = src->getAttrOfType<StringAttr>(kAttrSymVisibility))
    newFunc->setAttr(kAttrSymVisibility, vis);
  else
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
  // use the original kernel to avoid pulling in the extra block param (TODO we
  // should just embed that in the getProgramId op)
  SmallVector<Type> wrapInputs(kernel.getArgumentTypes().begin(),
                               kernel.getArgumentTypes().end());
  wrapInputs.push_back(i32Ty); // bStart
  wrapInputs.push_back(i32Ty); // bEnd

  auto wrapTy = FunctionType::get(ctx, wrapInputs, {});
  // Copy function-level user attrs from kernel (optional).
  SmallVector<NamedAttribute> userAttrs = collectClonableOpAttrs(kernel);

  // Arg attrs: copy 1:1 for the first original args; add empty for the 3 new
  // ones.
  SmallVector<DictionaryAttr> argDicts = getArgAttrArray(kernel);
  argDicts.resize(wrapInputs.size(), DictionaryAttr::get(ctx, {}));

  auto wrap =
      b.create<triton::FuncOp>(kernel.getLoc(), name, wrapTy,
                               /*attrs=*/userAttrs, /*argAttrs=*/argDicts);
  wrap.setPublic();

  Block *entry = wrap.addEntryBlock();
  OpBuilder wb(entry, entry->end());

  Value bEnd = entry->getArgument(wrap.getNumArguments() - 1);
  Value bStart = entry->getArgument(wrap.getNumArguments() - 2);

  Value bStartIdx =
      wb.create<arith::IndexCastOp>(wrap.getLoc(), wb.getIndexType(), bStart);
  Value bEndIdx =
      wb.create<arith::IndexCastOp>(wrap.getLoc(), wb.getIndexType(), bEnd);

  Value bStepIdx = wb.create<arith::ConstantIndexOp>(
      wrap.getLoc(), 1); // TODO: should we make this a param too?

  scf::ForOp forOp = wb.create<scf::ForOp>(wrap.getLoc(), bStartIdx, bEndIdx,
                                           bStepIdx, ValueRange{});
  {
    Block *body = forOp.getBody();
    OpBuilder fb(body, body->begin());

    Value bI32 = fb.create<arith::IndexCastOp>(wrap.getLoc(), i32Ty,
                                               forOp.getInductionVar());

    SmallVector<Value> callArgs;
    for (BlockArgument arg : wrap.getArguments().drop_back(2)) {
      callArgs.push_back(arg);
    }
    callArgs.push_back(bI32); // add the block index offset

    // tt::CallOp can call tt.func by symbol (has FunctionType).
    fb.create<triton::CallOp>(wrap.getLoc(), impl.getSymName(), TypeRange{},
                              callArgs);
  }

  wb.create<triton::ReturnOp>(wrap.getLoc());
  return wrap;
}

} // namespace

struct AddKernelStreamPass
    : public impl::AddKernelStreamPassBase<AddKernelStreamPass> {
  using AddKernelStreamPassBase::AddKernelStreamPassBase;

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
#if 1
    StringRef oldName = kernel.getName();
    std::string implName = (oldName + ".impl").str();
    triton::FuncOp implFunc =
        cloneTTFuncWithExtraI32Arg(moduleOp, kernel, implName);
#else
    SmallVector<Type> argTys(kernel.getArgumentTypes().begin(),
                             kernel.getArgumentTypes().end());
    Type i32Ty = IntegerType::get(ctx, 32);
    argTys.push_back(i32Ty); // block index

    auto kernelImpl = b.create<triton::FuncOp>(
        kernel.getLoc(), StringRef(implName),
        FunctionType::get(ctx, argTys, kernel.getResultTypes()));
    kernelImpl.setPrivate();

#if 0
    if (kernel.getArgAttrs())
        kernelImpl.setArgAttrs(*kernel.getArgAttrs());
    if (kernel.getResAttrs())
        kernelImpl.setResAttrs(*kernel.getResAttrs());

    auto argAttrs = kernel.getArgAttrs();
    if (argAttrs) {
        for (auto it : *argAttrs) {
            kernelImpl.setArgAttr(it.first, it.second);
        }
    }
    auto resAttrs = kernel.getResAttrs();
    if (resAttrs) {
        for (auto it : *resAttrs) {
            kernelImpl.setResAttr(it.first, it.second);
        }
    }
#endif

    moduleOp.push_back(kernelImpl);

    // Build entry block and clone the body.
    kernelImpl.addEntryBlock();
    IRMapping mapper;
    // Map original args → new args (excluding the extra last one).
    for (auto it :
         llvm::zip(kernel.getArguments(),
                   kernelImpl.getBody().front().getArguments().take_front(
                       kernel.getNumArguments())))
      mapper.map(std::get<0>(it), std::get<1>(it));

    b.setInsertionPointToStart(&kernelImpl.getBody().front());
    for (Operation &op : kernel.getBody().front())
      b.clone(op, mapper);
#endif

    // 2. Rewrite the tt.get_program_id operation to add the block index offset
    // to the return value
    unsigned blockIdxOffset = implFunc.getNumArguments() - 1;
    if (failed(offsetPidByBlockIndex(implFunc, blockIdxOffset)))
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
