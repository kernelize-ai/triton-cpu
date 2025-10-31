#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "masked-ops-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {

class ConvertMaskedLoadOp : public OpRewritePattern<triton::cpu::MaskedLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto elemTy = loadOp.getType();
    LDBG("Lowering masked load " << loadOp << " to LLVM");

    auto ptr = loadOp.getPtr();
    auto mask = loadOp.getMask();
    auto other = loadOp.getFalseVal();

    // masked load
    if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
      LDBG("Vector masked load type: " << vecTy);
      auto vecElemTy = vecTy.getElementType();
      auto elemSizeInBytes = vecElemTy.getIntOrFloatBitWidth() / 8;
      unsigned alignment = loadOp.getAlignment()
                               ? *loadOp.getAlignment()
                               : elemSizeInBytes * vecTy.getNumElements();
      LDBG("ptrType = " << ptr.getType() << ", maskType = " << mask.getType()
                        << ", otherType = " << other.getType()
                        << ", alignment = " << alignment);
      Value loadVal = LLVM::MaskedLoadOp::create(rewriter, loc, elemTy, ptr,
                                                 mask, other, alignment);
      rewriter.replaceOp(loadOp, {loadVal});
      return success();
    }

    unsigned alignment =
        loadOp.getAlignment()
            ? *loadOp.getAlignment()
            : std::min(8u,
                       getElementTypeOrSelf(elemTy).getIntOrFloatBitWidth() /
                           8u);
    // direct load
    if (matchPattern(mask, m_One())) {
      LDBG("Constant mask (" << mask << "), direct load type " << elemTy
                             << " with alignment " << alignment);
      Value loadVal =
          LLVM::LoadOp::create(rewriter, loc, elemTy, ptr, alignment);
      rewriter.replaceOp(loadOp, loadVal);
      return success();
    }

    // predicated load
    LDBG("Predicated load with alignment " << alignment);
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, ValueRange{},
                           afterLoad, ValueRange{other});
    rewriter.setInsertionPointToStart(trueBlock);
    auto load = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr, alignment);
    LLVM::BrOp::create(rewriter, loc, ValueRange{load.getResult()}, afterLoad);

    Value loadResult = afterLoad->getArgument(0);
    rewriter.replaceOp(loadOp, {loadResult});

    return success();
  }
};

class ConvertMaskedStoreOp
    : public OpRewritePattern<triton::cpu::MaskedStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::MaskedStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    LDBG("Lowering masked store " << storeOp << " to LLVM");

    auto ptr = storeOp.getPtr();
    auto mask = storeOp.getMask();
    auto val = storeOp.getValue();

    auto elemTy = val.getType();
    if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
      LDBG("Vector masked store type: " << vecTy);
      auto vecElemTy = vecTy.getElementType();
      auto elemSizeInBytes = vecElemTy.getIntOrFloatBitWidth() / 8;
      unsigned alignment = storeOp.getAlignment()
                               ? *storeOp.getAlignment()
                               : elemSizeInBytes * vecTy.getNumElements();
      LDBG("ptrType = " << ptr.getType() << ", maskType = " << mask.getType()
                        << ", alignment = " << alignment);
      rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(storeOp, val, ptr, mask,
                                                       alignment);
      return success();
    }

    unsigned alignment =
        storeOp.getAlignment()
            ? *storeOp.getAlignment()
            : std::min(8u,
                       getElementTypeOrSelf(elemTy).getIntOrFloatBitWidth() /
                           8u);
    // direct store
    if (matchPattern(mask, m_One())) {
      LDBG("Constant mask (" << mask << "), direct load type " << elemTy
                             << " with alignment " << alignment);
      LLVM::StoreOp::create(rewriter, loc, val, ptr, alignment);
    } else {
      LDBG("Predicated store with alignment " << alignment);
      // default to predicated load with conditional branching
      Block *currentBlock = rewriter.getInsertionBlock();
      Block *afterStore =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      Block *trueBlock = rewriter.createBlock(afterStore);
      rewriter.setInsertionPointToEnd(currentBlock);
      LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, afterStore);
      rewriter.setInsertionPointToStart(trueBlock);
      LLVM::StoreOp::create(rewriter, loc, val, ptr, alignment);
      LLVM::BrOp::create(rewriter, loc, afterStore);
      rewriter.setInsertionPointToStart(afterStore);
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_CONVERTMASKEDOPSTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

class ConvertMaskedOpsToLLVM
    : public impl::ConvertMaskedOpsToLLVMBase<ConvertMaskedOpsToLLVM> {
public:
  using impl::ConvertMaskedOpsToLLVMBase<
      ConvertMaskedOpsToLLVM>::ConvertMaskedOpsToLLVMBase;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<ConvertMaskedLoadOp>(patterns.getContext());
    patterns.add<ConvertMaskedStoreOp>(patterns.getContext());

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace cpu
} // namespace triton
} // namespace mlir
