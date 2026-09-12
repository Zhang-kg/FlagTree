#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";

  /// Return `annotation` with its unroll component replaced by an explicit
  /// `llvm.loop.unroll.disable`.  Preserve every other loop hint, in
  /// particular the LICM disable emitted by `tl.range(disable_licm=True)`.
  LLVM::LoopAnnotationAttr
  withUnrollingDisabled(MLIRContext *ctx,
                        LLVM::LoopAnnotationAttr annotation) {
    auto unroll = LLVM::LoopUnrollAttr::get(
        ctx, BoolAttr::get(ctx, true), {}, {}, {}, {}, {}, {});
    if (!annotation)
      return LLVM::LoopAnnotationAttr::get(
          ctx, {}, {}, {}, unroll, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
          {});

    return LLVM::LoopAnnotationAttr::get(
        ctx, annotation.getDisableNonforced(), annotation.getVectorize(),
        annotation.getInterleave(), unroll, annotation.getUnrollAndJam(),
        annotation.getLicm(), annotation.getDistribute(),
        annotation.getPipeline(), annotation.getPeeled(),
        annotation.getUnswitch(), annotation.getMustProgress(),
        annotation.getIsVectorized(), annotation.getStartLoc(),
        annotation.getEndLoc(), annotation.getParallelAccesses());
  }

public:
  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    SmallVector<scf::ForOp, 4> loopsToDisable;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Attribute absence deliberately keeps the historical behavior: the
      // Triton pass does nothing and LLVM may use its normal unroll heuristic.
      // Only an explicitly requested factor of one means end-to-end no-unroll.
      auto factor =
          forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName);
      if (!factor)
        return;
      if (factor.getInt() == 1)
        loopsToDisable.push_back(forOp);
      else if (factor.getInt() > 1)
        loops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    for (auto loop : loopsToDisable) {
      auto annotation = loop->getAttrOfType<LLVM::LoopAnnotationAttr>(
          "llvm.loop_annotation");
      loop->setAttr("llvm.loop_annotation",
                    withUnrollingDisabled(ctx, annotation));
      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Disabling LLVM loop unrolling for loop\n" << loop);
    }

    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);
      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // namespace mlir::triton
