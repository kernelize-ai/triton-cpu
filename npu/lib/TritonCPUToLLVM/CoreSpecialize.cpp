#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Analysis/Utility.h"

#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {
#define GEN_PASS_DEF_CORESPECIALIZE
#include "npu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace npu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class Specializer {
  triton::FuncOp func;
  SmallVector<triton::gpu::LocalAllocOp> allocs;
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::StoreOp> stores;
  DenseMap<Operation *, triton::gpu::LocalAllocOp> allocMap;
  public:
  Specializer(ModuleOp m, triton::FuncOp _func) : func(_func) {
    // collect all tile sizes and create multi-buffers
    for (auto load : func.getOps<triton::LoadOp>()) {
      auto alloc = createSharedBuffer(load, cast<RankedTensorType>(load.getType()));
      allocMap[load] = alloc;
      loads.push_back(load);
    }
    for (auto store : func.getOps<triton::StoreOp>()) {
      auto alloc = createSharedBuffer(store, cast<RankedTensorType>(store.getValue().getType()));
      allocMap[store] = alloc;
      stores.push_back(store);
    }

    // Create Reader Program
    auto readerFunc = makeReader(func);
    m.insert(func, readerFunc);

    // Create Compute Program
    auto computeFunc = makeCompute(func);
    m.insert(func, computeFunc);

    // Create Writer Program
    auto writerFunc = makeWriter(func);
    m.insert(func, writerFunc);
  }
  ~Specializer() {
    allocs.clear();
    loads.clear();
    stores.clear();
    allocMap.clear();
    func.erase();
  }

  
// func entry:
//  %57 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #ttg.shared_memory, mutable>

//      %90 = ttg.memdesc_index %57[%81] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x128xf16, #ttg.shared_memory, mutable, 2x128x128>
//      %91 = ttg.async_copy_global_to_local %arg11, %90 mask %89 other %cst_1 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>> -> <128x128xf16, #ttg.shared_memory, mutable, 2x128x128>
//      %92 = ttg.async_commit_group tokens %91 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
//      %93 = ttg.async_wait %92 {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}


triton::gpu::LocalAllocOp createSharedBuffer(Operation *op, RankedTensorType rtType) {
  int idx = allocs.size();
  auto ctx = func.getContext();
  OpBuilder b(func.getBody());
  auto shape = rtType.getShape();
  int32_t size = shape[0]; // tile size?
  SmallVector<std::pair<unsigned, unsigned>> intervalPads{{1, 1}};
  SmallVector<unsigned, 4> order{0};
  auto ctaLayout = triton::gpu::CTALayoutAttr::getDefault(ctx, 1);
  auto sharedEncoding = triton::gpu::PaddedSharedEncodingAttr::get(ctx, intervalPads, order, shape, ctaLayout);
  auto memdesc = triton::gpu::MemDescType::get(shape, rtType.getElementType(),
                                          sharedEncoding, triton::gpu::SharedMemorySpaceAttr::get(ctx), true);
  auto alloc = b.create<triton::gpu::LocalAllocOp>(op->getLoc(), memdesc);
  alloc->setAttr("alloc_idx", b.getI32IntegerAttr(idx));

  allocs.push_back(alloc);
  return alloc;
}



// Create a global tensor shared-buffer that is externally visible
// 
  triton::FuncOp makeReader(triton::FuncOp func) {
    IRMapping map;
    auto readerFunc = cast<triton::FuncOp>(func->clone(map));
    readerFunc.setName(func.getName().str() + "__reader");
    
    // Replace all loads with DMA to SRAM
    // TODO: handle address loads for subsequent loads
    // TODO: preload first N tiles (needs address logic..)
    for (auto load : loads) {
      auto alloc = allocMap[load];
      auto cload = cast<triton::LoadOp>(map.lookup(load).getDefiningOp());
      auto calloc = cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cload);
      auto loc = cload.getLoc();
      // replace uses with null value
      auto nullValue = b.create<arith::ConstantOp>(loc, cload.getType(), b.getZeroAttr(cload.getType()));
      cload.replaceAllUsesWith(nullValue.getResult());
      #if 0
      // create async copy, commit group, and wait
      auto asyncCopy = b.create<triton::gpu::AsyncCopyGlobalToLocalOp>(loc, cload.getPtr(), calloc, cload.getMask(), cload.getOther());
      auto commitGroup = b.create<triton::gpu::AsyncCommitGroupOp>(loc, asyncCopy.getToken());
      auto wait = b.create<triton::gpu::AsyncWaitOp>(loc, commitGroup.getAsyncToken(), 1);
      #else
      // Use local store until async copy is supported
      b.setInsertionPointAfter(cload);
      auto lstore = b.create<triton::gpu::LocalStoreOp>(loc, cload.getResult(), calloc);
      #endif
    }
    // Erase all stores
    for (auto store : stores) {
      auto cstore = cast<triton::StoreOp>(map.lookup(store));
      cstore.erase();
    }
    return readerFunc;
  }

  triton::FuncOp makeCompute(triton::FuncOp func) {
    IRMapping map;
    auto computeFunc = cast<triton::FuncOp>(func->clone(map));
    computeFunc.setName(func.getName().str() + "__compute");

    // Replace all loads with local loads
    for (auto load : loads) {
      auto alloc = allocMap[load];
      auto cload = cast<triton::LoadOp>(map.lookup(load).getDefiningOp());
      auto calloc = cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cload);
      auto loc = cload.getLoc();
      auto lload = b.create<triton::gpu::LocalLoadOp>(loc, cload.getType(), calloc);
      cload.replaceAllUsesWith(lload.getResult());
    }
    // Erase all stores
    for (auto store : stores) {
      auto alloc = allocMap[store];
      auto cstore = cast<triton::StoreOp>(map.lookup(store));
      auto calloc = cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cstore);
      auto loc = cstore.getLoc();
      auto lstore = b.create<triton::gpu::LocalStoreOp>(loc, cstore.getValue(), calloc);
      cstore.erase();
    }
    return computeFunc;
  }

  triton::FuncOp makeWriter(triton::FuncOp func) {
    IRMapping map;
    auto writerFunc = cast<triton::FuncOp>(func->clone(map));
    writerFunc.setName(func.getName().str() + "__writer");
    
    // Replace all stores with local stores
    for (auto store : stores) {
      auto alloc = allocMap[store];
      auto cstore = cast<triton::StoreOp>(map.lookup(store));
      auto calloc = cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cstore);
      auto loc = cstore.getLoc();
      #if 1
      // Use local load until async store is supported
      auto lload = b.create<triton::gpu::LocalLoadOp>(loc, cstore.getValue().getType(), calloc);
      cstore.getValueMutable().assign(lload.getResult());
      #else
      // create async copy, commit group, and wait
      auto asyncCopy = b.create<triton::gpu::AsyncCopyLocalToGlobalOp>(loc, calloc, cstore.getPtr(), cstore.getMask());
      auto commitGroup = b.create<triton::gpu::AsyncCommitGroupOp>(loc, asyncCopy.getToken());
      auto wait = b.create<triton::gpu::AsyncWaitOp>(loc, commitGroup.getAsyncToken(), 1);
      cstore.erase();
      #endif
    }
    return writerFunc;
  }
};

class CoreSpecializePass : public triton::npu::impl::CoreSpecializeBase<CoreSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<triton::FuncOp> funcOps;
    for (auto func : m.getOps<triton::FuncOp>()) {
      funcOps.push_back(func);
    }

    for (auto func : funcOps) {
      Specializer(m, func);
    }
    return;
  }
};

} // namespace
