#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "PatternTritonNPUToTenstorrent.h"
#include "PointerInfoAnalysis.h"
#include "TypeConverter.h"
#include "Utility.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel

// Tenstorrent TTKernel includes
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOTTKERNEL
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline bool isCBOp(Operation *op) {
  if (auto compileTimeArg = dyn_cast<ttkernel::GetCompileArgValOp>(op)) {
    return isa<ttkernel::CBType>(compileTimeArg.getType());
  }
  return false;
}

inline bool requiresSFPUInit(func::FuncOp funcOp) {
  bool requiresInit = false;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<ttkernel::TTKernelBinaryOpTrait>()) {
      requiresInit = true;
      return;
    }
  });
  return requiresInit;
}

inline bool requiresMMInit(func::FuncOp funcOp) {
  bool requiresInit = false;
  funcOp.walk([&](ttkernel::MatmulTilesOp dotOp) {
    requiresInit = true;
    return;
  });
  return requiresInit;
}

class InitializationHelper {
public:
  InitializationHelper(func::FuncOp F)
      : funcOp(F), addSFPUInit(requiresSFPUInit(F)),
        addMMInit(requiresMMInit(F)) {
    // build maps of copy/pack ops to respective indices
    funcOp.walk([&](Operation *op) {
      if (auto copyTileOp = dyn_cast<ttkernel::CopyTileOp>(op)) {
        copyTileOps[copyTileOp]++;
      } else if (auto copyTileInitOp = dyn_cast<ttkernel::CopyTileInitOp>(op)) {
        copyTileInitOps.push_back(copyTileInitOp);
      } else if (auto packTileOp = dyn_cast<ttkernel::PackTileOp>(op)) {
        packTileOps.insert(packTileOp);
      } else if (auto matmulTilesOp = dyn_cast<ttkernel::MatmulTilesOp>(op)) {
        matmulTilesOps.insert(matmulTilesOp);
      }
    });
  }

  void insertTileRegsAcquireOps() {
    SmallVector<ttkernel::PackTileOp, 4> packTileOps;
    funcOp.walk([&](ttkernel::PackTileOp op) { packTileOps.push_back(op); });
    assert(!packTileOps.empty() && "expecting at least one pack tile op");
    auto firstPackTileOp = packTileOps.front();
    Block *parentBlock = firstPackTileOp->getBlock();
    OpBuilder builder(parentBlock, parentBlock->begin());
    // Put the tile regs acquire op after the last GetCompileArgValOp.
    // GetCompileArgValOps correspond to CBs. TileRegsAcquire is used for
    // locating SFPU/FPU init ops, which take CBs as input. Putting tile regs
    // acquire after the last CB guarantees CBs will exist for initialization
    // ops.
    auto getArgValOpItr = parentBlock->getOps<ttkernel::GetCompileArgValOp>();
    if (!getArgValOpItr.empty()) {
      auto argValOps = llvm::reverse(getArgValOpItr);
      builder.setInsertionPointAfter(*argValOps.begin());
    }
    ttkernel::TileRegsAcquireOp::create(builder, firstPackTileOp->getLoc());
  }

  void insertComputeInitializationOps() {
    auto getLatestValue = [](ArrayRef<Value> values) -> Value {
      Operation *latestOp = nullptr;
      Value latestValue;

      for (Value val : values) {
        Operation *defOp = val.getDefiningOp();
        if (!defOp)
          continue; // Skip block arguments

        if (!latestOp || latestOp->isBeforeInBlock(defOp)) {
          latestOp = defOp;
          latestValue = val;
        }
      }

      return latestValue;
    };

    // mminit also does sfpu init
    if (addMMInit) {
      // mm_init goes after cb initialization ops. but we need to collect the
      // circular buffers first
      auto matmulTilesOp = *matmulTilesOps.begin();

      Location loc = matmulTilesOp.getLoc();
      OpBuilder builder(matmulTilesOp);

      Value aCb = matmulTilesOp.getIn0CbId();
      Value bCb = matmulTilesOp.getIn1CbId();
      ttkernel::PackTileOp packTileOp = *packTileOps.begin();
      Value outCb = packTileOp.getOutCb();

      Value latest = getLatestValue({aCb, bCb, outCb});
      builder.setInsertionPointAfterValue(latest);

      // TODO: support transpose. we cannot grab the transpose from the matmul
      // op as transpose could be defined inside the matmul loop
      Value transpose = arith::createConstantI32(loc, builder, 0);
      ttkernel::MatmulInitOp::create(builder, loc, aCb, bCb, outCb, transpose);

      SmallVector<ttkernel::TileRegsAcquireOp, 4> tileRegsAcquireOps;
      funcOp.walk([&](ttkernel::TileRegsAcquireOp op) {
        tileRegsAcquireOps.push_back(op);
      });
      assert(!tileRegsAcquireOps.empty() && "expecting tile regs acquire op");
      Operation *acquireOp = tileRegsAcquireOps.front();

      builder.setInsertionPointAfter(acquireOp);
      ttkernel::MatmulInitShortOp::create(builder, loc, aCb, bCb, transpose);

    } else if (addSFPUInit) {
      SmallVector<ttkernel::TileRegsAcquireOp, 4> tileRegsAcquireOps;
      funcOp.walk([&](ttkernel::TileRegsAcquireOp op) {
        tileRegsAcquireOps.push_back(op);
      });
      assert(!tileRegsAcquireOps.empty() && "expecting tile regs acquire op");
      Operation *acquireOp = tileRegsAcquireOps.front();
      OpBuilder builder(acquireOp);
      ttkernel::PackTileOp packTileOp = *packTileOps.begin();
      Value inCb = copyTileOps.begin()->first.getCb0();
      Value outCb = packTileOp.getOutCb();

      Value latest = getLatestValue({inCb, outCb});
      builder.setInsertionPointAfterValue(latest);

      ttkernel::InitSFPUOp::create(builder, acquireOp->getLoc(), inCb, outCb);
    }
  }

  void insertTileRegsOps() {
    // look for groups of `ttkernel::PackTileOp` and insert sequences of
    // TileRegsCommitOp, TileRegsWaitOp, and then (later) TileRegsReleaseOp
    assert(!packTileOps.empty() && "expecting at least one pack tile op");
    ttkernel::PackTileOp firstPackTileOp = *packTileOps.begin();
    OpBuilder builder(firstPackTileOp);
    ttkernel::TileRegsCommitOp::create(builder, firstPackTileOp.getLoc());
    ttkernel::TileRegsWaitOp::create(builder, firstPackTileOp.getLoc());

    ttkernel::PackTileOp lastPackTileOp = *packTileOps.rbegin();
    builder.setInsertionPointAfter(lastPackTileOp);
    ttkernel::TileRegsReleaseOp::create(builder, lastPackTileOp.getLoc());
  }

private:
  SmallVector<ttkernel::CopyTileInitOp, 4> copyTileInitOps;
  llvm::MapVector<ttkernel::CopyTileOp, unsigned> copyTileOps;
  SetVector<ttkernel::PackTileOp> packTileOps;
  SetVector<ttkernel::MatmulTilesOp> matmulTilesOps;

  func::FuncOp funcOp;
  const bool addSFPUInit;
  const bool addMMInit;
};

} // namespace

struct ConvertTritonNPUToTTKernelPass
    : public impl::ConvertTritonNPUToTTKernelBase<
          ConvertTritonNPUToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonNPUToTenstorrentTypeConverter typeConverter(context);

    mlir::ConversionTarget funcTarget(*context);
    funcTarget.addIllegalOp<triton::FuncOp>();
    funcTarget.addIllegalOp<triton::ReturnOp>();

    funcTarget.addLegalDialect<arith::ArithDialect>();
    funcTarget.addLegalDialect<func::FuncDialect>();
    funcTarget.addLegalDialect<triton::TritonDialect>();
    funcTarget.addLegalOp<UnrealizedConversionCastOp>();
    funcTarget.addLegalOp<ttkernel::GetArgValOp>();
    funcTarget.addLegalOp<ttkernel::GetCommonArgValOp>();
    funcTarget.addLegalOp<ttkernel::TensorAccessorArgsOp>();

    mlir::RewritePatternSet funcPatterns(context);
    populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                    PatternBenefit(1));
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    LLVM_DEBUG({
      DBGS() << "After FuncOp conversion:\n";
      mod.dump();
    });

    npu::PointerInfoAnalysis pointerInfoAnalysis(mod);

    mlir::ConversionTarget target{*context};

    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::cpu::TritonCPUDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp funcOp) { return funcOp.getNumArguments() == 0; });
    target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *op) {
      // only legal if not operating on tensors
      return llvm::all_of(op->getOperands(), [](Value v) {
        return !(isa<RankedTensorType>(v.getType()));
      });
    });

    mlir::RewritePatternSet patterns(context);
    populateMemoryOpConversionPattern(typeConverter, patterns,
                                      &pointerInfoAnalysis, PatternBenefit(1));
    populateComputeOpConversionPattern(typeConverter, patterns,
                                       PatternBenefit(1));
    populateDotOpConversionPattern(typeConverter, patterns, PatternBenefit(1));
    populateElementwiseOpConversionPattern(typeConverter, patterns,
                                           PatternBenefit(1));
    populateIntrinsicOpConversionPattern(typeConverter, patterns,
                                         PatternBenefit(1));
    populateMakeRangeOpConversionPattern(typeConverter, patterns,
                                         PatternBenefit(1));
    populatePrintOpConversionPattern(typeConverter, patterns,
                                     PatternBenefit(1));
    populateSPMDOpConversionPattern(typeConverter, patterns, PatternBenefit(1));
    populateViewOpConversionPattern(typeConverter, patterns, PatternBenefit(1));
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // insert tile regs acquire before copy tile ops
    mod.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getSymName().ends_with("__compute"))
        return;

      InitializationHelper initHelper(funcOp);
      initHelper.insertTileRegsAcquireOps();
      initHelper.insertComputeInitializationOps();
      initHelper.insertTileRegsOps();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
