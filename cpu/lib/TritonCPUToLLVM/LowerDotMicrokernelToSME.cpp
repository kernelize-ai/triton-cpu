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

#include "mlir/Analysis/SliceAnalysis.h"
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

  Value row = b.mul(arith::IndexCastOp::create(rewriter, loc, i64_ty, k),
                    b.i64_val(stride))
                  .getResult();
  Value off =
      b.add(row, arith::IndexCastOp::create(rewriter, loc, i64_ty, col));
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
  // TritonLLVMOpBuilder b(loc, rewriter);

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
  // All loop induction (m, n, k) and tile offsets are `index`; we drop to i64
  // only for llvm pointer math inside loadPackedSlice.
  Value vscale = vector::VectorScaleOp::create(rewriter, loc); // index
  Value tw = arith::MulIOp::create(
      rewriter, loc, vscale, arith::ConstantIndexOp::create(rewriter, loc, 4));
  Value step = arith::AddIOp::create(rewriter, loc, tw, tw);

  // arm_sme.tile_load / tile_store require a rank-2 memref base, not an
  // !llvm.ptr. Wrap C as a contiguous row-major memref<?x?xf32> (blockM x
  // blockN, row stride = blockN).
  Value cMemref =
      wrapPtrAsMemref(rewriter, loc, cPtr, desc.blockM, desc.blockN);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value cBlockM = arith::ConstantIndexOp::create(rewriter, loc, desc.blockM);
  Value cBlockN = arith::ConstantIndexOp::create(rewriter, loc, desc.blockN);
  // KC slab count arrives as i64; the k loop runs in `index`.
  Value kUb = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                         kBlocks);

  // ==== for m = 0 .. blockM step step ====================================
  scf::ForOp::create(
      rewriter, loc, c0, cBlockM, step, ValueRange{},
      [&](OpBuilder &mRewriter, Location loc, Value m, ValueRange) {
        // ==== for n = 0 .. blockN step step ==============================
        scf::ForOp::create(
            mRewriter, loc, c0, cBlockN, step, ValueRange{},
            [&](OpBuilder &nRewriter, Location loc, Value n, ValueRange) {
              Value mTw = arith::AddIOp::create(nRewriter, loc, m, tw);
              Value nTw = arith::AddIOp::create(nRewriter, loc, n, tw);

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
                  nRewriter, loc, c0, kUb, c1, ValueRange{t0, t1, t2, t3},
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

// Build a 2D `tensor<rows x cols x !tt.ptr<elemTy>, enc>` addressing a
// contiguous, row-major buffer based at the scalar pointer `basePtr`
// (!tt.ptr<elemTy>). offset[r, c] = r * cols + c. Shape/encoding are taken from
// `likeTy` so the result can feed a `tt.store` whose value has type `likeTy`.
//
// `rowStride`/`colStride` are the buffer's per-dimension strides in elements
// (row-major contiguous => colStride == 1, rowStride == leading dim, which may
// exceed `cols` when the tile is a window into a larger buffer).
// `rowOffset`/`colOffset` are absolute i32 positions of this tile's top-left
// element within that buffer. offset[r, c] = (rowOffset + r) * rowStride +
// (colOffset + c) * colStride.
//
// Each dimension is folded in via its own tt.addptr (rather than summing the
// offsets first) so the stride-1 dimension stays visible to the backend's
// contiguity/divisibility analysis, which is what lets it vectorize the access.
static Value buildContiguousPtrTensor(OpBuilder &rewriter, Location loc,
                                      Value basePtr, Value rowOffset,
                                      Value colOffset, int64_t rowStride,
                                      int64_t colStride,
                                      RankedTensorType likeTy) {
  MLIRContext *context = rewriter.getContext();
  assert(likeTy.getRank() == 2 && "expected a 2D value tensor");
  int64_t rows = likeTy.getShape()[0];
  int64_t cols = likeTy.getShape()[1];
  auto enc = cast<triton::gpu::DistributedEncodingTrait>(likeTy.getEncoding());
  Type i32Ty = rewriter.getI32Type();

  // Row indices: tensor<rows x i32, slice<dim=1, enc>> spanning
  //   [rowOffset, rowOffset + rows) --expand axis=1--> tensor<rows x 1 x i32,
  //   enc>
  auto rowSlice = triton::gpu::SliceEncodingAttr::get(context, /*dim=*/1, enc);
  Value rowRange = cpu::MakeDynamicRangeOp::create(
      rewriter, loc, RankedTensorType::get({rows}, i32Ty, rowSlice), rowOffset);
  Value rows2d = triton::ExpandDimsOp::create(
      rewriter, loc, RankedTensorType::get({rows, 1}, i32Ty, enc), rowRange,
      /*axis=*/1);

  // Col indices: tensor<cols x i32, slice<dim=0, enc>> spanning
  //   [colOffset, colOffset + cols) --expand axis=0--> tensor<1 x cols x i32,
  //   enc>
  auto colSlice = triton::gpu::SliceEncodingAttr::get(context, /*dim=*/0, enc);
  Value colRange = cpu::MakeDynamicRangeOp::create(
      rewriter, loc, RankedTensorType::get({cols}, i32Ty, colSlice), colOffset);
  Value cols2d = triton::ExpandDimsOp::create(
      rewriter, loc, RankedTensorType::get({1, cols}, i32Ty, enc), colRange,
      /*axis=*/0);

  // Scale each dimension's offset by its stride, splatting the stride into the
  // expanded (rows x 1 / 1 x cols) shape so per-dimension divisibility
  // survives.
  auto scaleByStride = [&](Value expanded, int64_t stride) -> Value {
    Value strideSplat = triton::SplatOp::create(
        rewriter, loc, cast<RankedTensorType>(expanded.getType()),
        arith::ConstantOp::create(rewriter, loc,
                                  rewriter.getI32IntegerAttr(stride)));
    return arith::MulIOp::create(rewriter, loc, expanded, strideSplat);
  };
  Value rowOff = scaleByStride(rows2d, rowStride);
  Value colOff = scaleByStride(cols2d, colStride);

  // Broadcast each to rows x cols and fold in via its own addptr, keeping the
  // stride-1 dimension distinct for the vectorizer.
  auto offTy = RankedTensorType::get({rows, cols}, i32Ty, enc);
  auto ptrTensorTy = RankedTensorType::get(
      {rows, cols}, PointerType::get(likeTy.getElementType(), /*addrSpace=*/0),
      enc);
  Value ptr = triton::SplatOp::create(rewriter, loc, ptrTensorTy, basePtr);
  ptr = triton::AddPtrOp::create(
      rewriter, loc, ptrTensorTy, ptr,
      triton::BroadcastOp::create(rewriter, loc, offTy, rowOff));
  ptr = triton::AddPtrOp::create(
      rewriter, loc, ptrTensorTy, ptr,
      triton::BroadcastOp::create(rewriter, loc, offTy, colOff));
  return ptr;
}

void rewriteExistingFunctionBody(DotDescriptor &desc, triton::FuncOp funcOp,
                                 MLIRContext *context) {
  Location loc = funcOp.getLoc();

  OpBuilder rewriter(context);
  // insert new ops after the existing generic so we can re-use its inputs. the
  // existing generic is deleted after this rewrite completes.
  rewriter.setInsertionPointAfter(desc.generic);
  TritonLLVMOpBuilder b(loc, rewriter);

  Value aBuf = cpu::LocalAllocOp::create(
      rewriter, loc, PointerType::get(f32_ty, 0), desc.blockK * desc.blockM);
  Value bBuf = cpu::LocalAllocOp::create(
      rewriter, loc, PointerType::get(f32_ty, 0), desc.blockK * desc.blockN);
  Value cBuf = cpu::LocalAllocOp::create(
      rewriter, loc, PointerType::get(f32_ty, 0), desc.blockM * desc.blockN);

  // kFull is i32
  auto kLoop = scf::ForOp::create(rewriter, loc, b.i32_val(0), desc.kFull,
                                  b.i32_val(desc.blockK), ValueRange{});
  rewriter.setInsertionPointToStart(kLoop.getBody());

  // TODO: a weird place for this - maybe this should be part of the matcher for
  // the descriptor?
  auto nonTerminatorOps = desc.generic.getBody().front().without_terminator();
  assert(
      !nonTerminatorOps.empty() &&
      &*std::prev(nonTerminatorOps.end()) == desc.dot.getOperation() &&
      "expected dot op to be the last op in the microkernel function generic");

  auto existingTileShapes = desc.generic.getTileShape(); // K, M, N

  // pack A
  {
    Value a = desc.dot.getA();
    Operation *aOp = a.getDefiningOp();
    // The dot (and hence its dot_op operand layout) is being replaced by the
    // SME microkernel, so the dot-operand encoding is irrelevant here. Peel the
    // convert_layout feeding the dot and pack the raw blocked load tile
    // instead, which transposes cleanly (a dot_op layout may not).
    if (auto cvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(aOp))
      if (Operation *src = cvt.getSrc().getDefiningOp())
        aOp = src;
    llvm::errs() << "aOp: " << *aOp << "\n";
    assert(aOp &&
           "expected dot op a operand to be an operation, not block argument");
    auto aType = cast<RankedTensorType>(aOp->getResult(0).getType());
    assert(isa<BlockedEncodingAttr>(aType.getEncoding()) &&
           "only blocked encoding tensors are supported");

    SetVector<Operation *> slice;
    (void)mlir::getBackwardSlice(aOp, &slice);

    IRMapping mapping;

    SmallVector<Value> newIns(desc.generic.getIns());
    newIns.push_back(aBuf);

    auto newGeneric = cpu::GenericOp::create(
        rewriter, loc, /*results=*/TypeRange{}, newIns,
        {b.i32_val(desc.blockK), b.i32_val(desc.blockM)},
        {static_cast<int32_t>(desc.blockK), existingTileShapes[1]});
    Block *body = rewriter.createBlock(&newGeneric.getBody());
    for (unsigned i = 0; i < newGeneric.getTileShape().size(); i++)
      body->addArgument(
          rewriter.getI32Type(),
          newGeneric.getLoc()); // tile offset per vector shape dim

    for (auto existingArg : desc.generic.getBody().getArguments().drop_front(
             desc.generic.getNumInductionVars() +
             desc.generic.getNumIterArgs())) {
      mapping.map(existingArg, body->addArgument(existingArg.getType(),
                                                 existingArg.getLoc()));
    }
    // TODO: map induction vars
    // K, M, N --> K, M
    mapping.map(desc.generic.getBody().getArgument(0), body->getArgument(0));
    mapping.map(desc.generic.getBody().getArgument(1), body->getArgument(1));

    auto aBufBlockArg = body->addArgument(aBuf.getType(), aBuf.getLoc());
    rewriter.setInsertionPointToStart(body);

    for (auto *op : slice) {
      rewriter.clone(*op, mapping);
    }

    Value newA = rewriter.clone(*aOp, mapping)->getResult(0); // [M, K] blocked

    // The SME kernel reads A packed as [K, M] (M contiguous, off = k*blockM+m).
    // Transpose the data to [K, M] so the store is contiguous (M is the fast,
    // stride-1 axis) and vectorizes, rather than scattering.
    Value newAT =
        triton::TransOp::create(rewriter, loc, newA, ArrayRef<int32_t>{1, 0});

    // Contiguous [K, M] pointer tensor matching newAT: rows=K (stride blockM),
    // cols=M (stride 1). tt.store needs shape/encoding to match the value.
    Value aPtrs =
        buildContiguousPtrTensor(rewriter, loc, aBufBlockArg,
                                 /*rowOffset=*/body->getArgument(0), // K offset
                                 /*colOffset=*/body->getArgument(1), // M offset
                                 /*rowStride=*/desc.blockM, /*colStride=*/1,
                                 cast<RankedTensorType>(newAT.getType()));
    triton::StoreOp::create(rewriter, loc, aPtrs, newAT,
                            triton::CacheModifier::NONE,
                            triton::EvictionPolicy::NORMAL);
    cpu::YieldOp::create(rewriter, loc, /*values=*/ValueRange{});
  }

  // pack B

  // call streaming ukernel

  rewriter.setInsertionPointAfter(kLoop);
  // drain C
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

      // NOTE: this assumes only one streaming microkernel generic per module
      funcOp.walk([&](cpu::GenericOp genericOp) {
        // 1. see if the current dot descriptor pattern matches
        // (if not, should we drop no-inline and run the inliner? probably. we
        // will likely run the inliner anyway)
        auto desc = DotDescriptor::tryMatch(genericOp);
        if (!desc)
          return WalkResult::advance();

        // 2. Get-or-create the leaf at module scope
        auto smeKernel = createStreamingSMEKernel(*desc, mod);

        // 3. In-body rewrite, at the generic's location: entry-block allocas →
        // zero C → slab loop { pack A, pack B, call } → drain.
        rewriteExistingFunctionBody(*desc, funcOp, context);

        // 4. Delete the generic.

        // assert(false && "TODO");
        return WalkResult::interrupt();
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
