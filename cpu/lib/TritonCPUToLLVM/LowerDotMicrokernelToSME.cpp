#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_LOWERDOTMICROKERNELTOSME
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

namespace {

struct DotDescriptor {
  cpu::GenericOp generic; // anchor: read geometry, delete in phase 3
  triton::DotOp dot;

  // Type elemTy;            // f32
  int64_t blockM, blockN; // 64, 64   <- generic.blocks[1..2]
  int64_t blockK;         // 32       <- tileShape[0]; slab depth + pack height
  Value kFull;            //          <- generic.blocks[0]; slab loop bound

  // Value aTile, bTile, acc; // dot.getA()/getB()/getC() — one hop, no chain
  // walk

  static std::optional<DotDescriptor> tryMatch(cpu::GenericOp genericOp) {
    triton::DotOp dotOp;

    WalkResult result = genericOp.walk([&](triton::DotOp dot) {
      if (!dotOp) {
        dotOp = dot;
        return WalkResult::advance();
      }
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      return std::nullopt;

    if (genericOp.getBlockShape().size() != 3)
      return std::nullopt;
    DotDescriptor ret;
    bool failed = false;
    // we expect the generic block shape to be K, M, N
    for (auto [i, dim] : llvm::enumerate(genericOp.getBlockShape())) {
      if (i == 0) {
        ret.kFull = dim;
        continue;
      }
      APInt val;
      if (!matchPattern(dim, m_ConstantInt(&val))) {
        failed = true;
        break;
      }
      if (i == 1)
        ret.blockM = val.getSExtValue();
      else
        ret.blockN = val.getSExtValue();
    }
    if (failed)
      return std::nullopt;

    ret.blockK = genericOp.getTileShape()[0];
    ret.generic = genericOp;
    ret.dot = dotOp;
    return ret;
  }
};

// Load one vector<[4]xf32> slice out of a packed A/B buffer (row-major
// [KC, stride]) at logical position [k, col]. offset = k * stride + col.
static Value loadPackedSlice(OpBuilder &rewriter, Location loc, Type sveTy,
                             Value basePtr, Value k, Value col,
                             int64_t stride) {
  TritonLLVMOpBuilder b(loc, rewriter);

  Value row = b.mul(k, b.i64_val(stride)).getResult();
  Value off = b.add(row, col);
  Value gep = b.gep(basePtr.getType(), f32_ty, basePtr, ValueRange{off});
  return b.load(sveTy, gep);
}

// TODO: claude generated, check this
// Wrap a bare !llvm.ptr as a contiguous, row-major rank-2 memref<?x?xf32> so
// arm_sme.tile_load / tile_store (which require a memref base) can address it.
// Builds a MemRef descriptor struct from the pointer and casts it to the memref
// type. `rows` x `cols` f32 elements, row stride = cols.
static Value wrapPtrAsMemref(OpBuilder &rewriter, Location loc, Value basePtr,
                             int64_t rows, int64_t cols) {
  MLIRContext *context = rewriter.getContext();
  // { allocatedPtr, alignedPtr, offset, sizes[2], strides[2] }
  auto descTy = LLVM::LLVMStructType::getLiteral(
      context, {ptr_ty(context), ptr_ty(context), i64_ty, array_ty(i64_ty, 2),
                array_ty(i64_ty, 2)});

  TritonLLVMOpBuilder b(loc, rewriter);

  MemRefDescriptor desc = MemRefDescriptor::poison(rewriter, loc, descTy);
  desc.setAllocatedPtr(rewriter, loc, basePtr);
  desc.setAlignedPtr(rewriter, loc, basePtr);
  desc.setOffset(rewriter, loc, b.i64_val(0));
  desc.setSize(rewriter, loc, 0, b.i64_val(rows));
  desc.setSize(rewriter, loc, 1, b.i64_val(cols));
  desc.setStride(rewriter, loc, 0, b.i64_val(cols)); // row-major, contiguous
  desc.setStride(rewriter, loc, 1, b.i64_val(1));

  Value descVal = desc;
  auto memrefTy =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32_ty);
  return UnrealizedConversionCastOp::create(rewriter, loc, TypeRange{memrefTy},
                                            ValueRange{descVal})
      .getResult(0);
}

static triton::FuncOp createStreamingSMEKernel(DotDescriptor &desc,
                                               ModuleOp mod) {
  std::string base = "triton_cpu_sme_microkernel_";
  base += std::to_string(desc.blockM) + "_" + std::to_string(desc.blockN) +
          "_" + std::to_string(desc.blockK);

  if (auto existing = mod.lookupSymbol(base))
    return cast<triton::FuncOp>(existing);

  MLIRContext *context = mod.getContext();
  OpBuilder rewriter(context);
  rewriter.setInsertionPointToStart(mod.getBody());

  SmallVector<Type> argumentTypes{/*a=*/ptr_ty(context), /*b=*/ptr_ty(context),
                                  /*c=*/ptr_ty(context), /*kStep=*/i64_ty};

  auto smeFuncTy = rewriter.getFunctionType(argumentTypes, {});
  auto smeFunc =
      triton::FuncOp::create(rewriter, desc.generic.getLoc(), base, smeFuncTy);
  smeFunc.setVisibility(SymbolTable::Visibility::Private);
  smeFunc->setAttr("noinline", rewriter.getBoolAttr(true));
  smeFunc->setAttr("arm_locally_streaming", rewriter.getUnitAttr());
  smeFunc->setAttr("arm_new_za", rewriter.getUnitAttr());

  Block *entryBlock = smeFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  Location loc = desc.generic.getLoc();
  TritonLLVMOpBuilder b(loc, rewriter);

  // Kernel arguments
  Value aPtr = entryBlock->getArgument(0); // !llvm.ptr, packed A [KC, blockM]
  Value bPtr = entryBlock->getArgument(1); // !llvm.ptr, packed B [KC, blockN]
  Value cPtr =
      entryBlock->getArgument(2); // !llvm.ptr, C           [blockM, blockN]
  Value kBlocks = entryBlock->getArgument(3); // i64, number of K slabs (KC)

  // One Accumulator ZA tile: vector<[4]x[4]xf32> (both dims scalable).
  auto tileTy = VectorType::get({4, 4}, f32_ty, /*scalableDims=*/{true, true});
  // One SVE column/row: vector<[4]xf32>.
  auto sveTy = VectorType::get({4}, f32_ty, /*scalableDims=*/{true});

  // --- Tiling geometry ----------------------------------------------------
  // tw = 4 * vscale  == lanes in one vector<[4]xf32> == side of one ZA tile.
  // step = 2 * tw    == 2x2 ZA tile arrangement per (m, n) iteration.
  Value vscale = vector::VectorScaleOp::create(
      rewriter, loc); // index - TODO need index cast?
  Value tw = b.mul(vscale, b.i64_val(4));
  Value step = b.add(tw, tw);

  // arm_sme.tile_load / tile_store require a rank-2 memref base, not an
  // !llvm.ptr. Wrap C as a contiguous row-major memref<?x?xf32> (blockM x
  // blockN, row stride = blockN).
  Value cMemref =
      wrapPtrAsMemref(rewriter, loc, cPtr, desc.blockM, desc.blockN);

  // ==== for m = 0 .. blockM step step ====================================
  scf::ForOp::create(
      rewriter, loc, b.i64_val(0), b.i64_val(desc.blockM), step, ValueRange{},
      [&](OpBuilder &mRewriter, Location loc, Value m, ValueRange) {
        // ==== for n = 0 .. blockN step step ==============================
        scf::ForOp::create(
            mRewriter, loc, b.i64_val(0), b.i64_val(desc.blockN), step,
            ValueRange{},
            [&](OpBuilder &nRewriter, Location loc, Value n, ValueRange) {
              TritonLLVMOpBuilder nB(loc, nRewriter);
              Value mTw = nB.add(m, tw);
              Value nTw = nB.add(n, tw);

              // --- Load the four C sub-tiles into ZA -------------------
              // 2x2 arrangement: (m,n) (m,n+tw) / (m+tw,n) (m+tw,n+tw).
              auto loadTile = [&](Value row, Value col) -> Value {
                return arm_sme::TileLoadOp::create(
                    nRewriter, loc, tileTy, cMemref, ValueRange{row, col});
              };
              Value t0 = loadTile(m, n);
              Value t1 = loadTile(m, nTw);
              Value t2 = loadTile(mTw, n);
              Value t3 = loadTile(mTw, nTw);

              // --- Accumulate over K slabs -----------------------------
              auto kLoop = scf::ForOp::create(
                  nRewriter, loc, b.i64_val(0), kBlocks, b.i64_val(1),
                  ValueRange{t0, t1, t2, t3},
                  [&](OpBuilder &kRewriter, Location loc, Value k,
                      ValueRange acc) {
                    Value a0 = loadPackedSlice(kRewriter, loc, sveTy, aPtr, k,
                                               m, desc.blockM);
                    Value a1 = loadPackedSlice(kRewriter, loc, sveTy, aPtr, k,
                                               mTw, desc.blockM);
                    Value b0 = loadPackedSlice(kRewriter, loc, sveTy, bPtr, k,
                                               n, desc.blockN);
                    Value b1 = loadPackedSlice(kRewriter, loc, sveTy, bPtr, k,
                                               nTw, desc.blockN);

                    auto op = [&](Value lhs, Value rhs, Value z) -> Value {
                      return arm_sme::OuterProductOp::create(
                          kRewriter, loc, tileTy, lhs, rhs, /*lhsMask=*/Value(),
                          /*rhsMask=*/Value(), /*acc=*/z,
                          arm_sme::CombiningKind::Add);
                    };
                    Value u0 = op(a0, b0, acc[0]);
                    Value u1 = op(a0, b1, acc[1]);
                    Value u2 = op(a1, b0, acc[2]);
                    Value u3 = op(a1, b1, acc[3]);
                    scf::YieldOp::create(kRewriter, loc,
                                         ValueRange{u0, u1, u2, u3});
                  });

              // --- Drain the accumulated tiles back to C ---------------
              auto storeTile = [&](Value tile, Value row, Value col) {
                arm_sme::TileStoreOp::create(nRewriter, loc, tile, cMemref,
                                             ValueRange{row, col});
              };
              storeTile(kLoop.getResult(0), m, n);
              storeTile(kLoop.getResult(1), m, nTw);
              storeTile(kLoop.getResult(2), mTw, n);
              storeTile(kLoop.getResult(3), mTw, nTw);

              scf::YieldOp::create(nRewriter, loc, ValueRange{});
            });
        scf::YieldOp::create(mRewriter, loc, ValueRange{});
      });

  triton::ReturnOp::create(rewriter, loc, ValueRange{});

  return smeFunc;
}

} // namespace
struct LowerDotMicrokernelToSMEPass
    : public impl::LowerDotMicrokernelToSMEBase<LowerDotMicrokernelToSMEPass> {
  using LowerDotMicrokernelToSMEBase::LowerDotMicrokernelToSMEBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mod.walk([&](triton::FuncOp funcOp) {
      if (!funcOp.getName().ends_with("matmul_kernel_dot_microkernel"))
        return;

      funcOp.walk([&](cpu::GenericOp genericOp) {
        // 1. see if the current dot descriptor pattern matches
        // (if not, should we drop no-inline and run the inliner? probably. we
        // will likely run the inliner anyway)
        auto desc = DotDescriptor::tryMatch(genericOp);
        if (!desc)
          return;

        // 2. Get-or-create the leaf at module scope
        auto smeKernel = createStreamingSMEKernel(*desc, mod);
        llvm::errs() << "created sme kernel: " << smeKernel << "\n";

        // 3. In-body rewrite, at the generic's location: entry-block allocas →
        // zero C → slab loop { pack A, pack B, call } → drain.

        // 4. Delete the generic.

        llvm::errs() << "lower microkernel " << funcOp.getName() << "\n";
        assert(false && "TODO");
      });
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerDotMicrokernelToSMEPass() {
  return std::make_unique<LowerDotMicrokernelToSMEPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
