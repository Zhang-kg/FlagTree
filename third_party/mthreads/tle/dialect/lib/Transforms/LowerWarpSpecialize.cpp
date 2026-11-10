#ifdef __TLE__

#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLEPREPAREWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
class PrepareWarpSpecializePass
    : public impl::TritonMUSAGPUTLEPrepareWarpSpecializeBase<
          PrepareWarpSpecializePass> {
  LogicalResult prepareWarpSpecialize(ModuleOp mod, ttg::WarpSpecializeOp ws,
                                      IRRewriter &rewriter) {
    auto func = ws->getParentOfType<triton::FuncOp>();
    if (!func)
      return ws.emitOpError("requires a parent Triton function");

    Block &entry = func.getBody().front();
    if (ws->getBlock() != &entry)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize must be in the function entry "
          "block");
    if (!ws->getNextNode() || !isa<triton::ReturnOp>(ws->getNextNode()) ||
        ws->getNextNode()->getNextNode())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize must be the final operation "
          "before tt.return");
    if (ws.getNumResults() != 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize does not support results");

    auto partitions = ws.getPartitionOp();
    auto partitionRegions = partitions.getPartitionRegions();
    ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
    if (partitionRegions.empty())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize requires at least one worker "
          "partition");
    if (partitionRegions.size() != partitionNumWarps.size())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize worker region and warp-count "
          "sizes must match");

    Region &defaultRegion = ws.getDefaultRegion();
    if (!defaultRegion.hasOneBlock())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize requires single-block "
          "default and worker regions");

    Block &defaultBlock = defaultRegion.front();
    auto defaultYield =
        dyn_cast<ttg::WarpYieldOp>(defaultBlock.getTerminator());
    if (!defaultYield || defaultYield.getNumOperands() != 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize default region must yield no "
          "values");

    ValueRange captures = partitions.getExplicitCaptures();
    for (auto [index, region] : llvm::enumerate(partitionRegions)) {
      if (partitionNumWarps[index] <= 0)
        return ws.emitOpError()
               << "mthreads TLE static warp_specialize worker " << index
               << " must have a positive static warp count";
      if (!region.hasOneBlock())
        return ws.emitOpError() << "mthreads TLE static warp_specialize worker "
                                << index << " must have exactly one block";

      Block &workerBlock = region.front();
      auto workerReturn =
          dyn_cast<ttg::WarpReturnOp>(workerBlock.getTerminator());
      if (!workerReturn || workerReturn->getNumOperands() != 0)
        return ws.emitOpError()
               << "mthreads TLE static warp_specialize worker " << index
               << " must end with ttg.warp_return with no values";
      if (workerBlock.getNumArguments() != captures.size())
        return ws.emitOpError() << "mthreads TLE static warp_specialize worker "
                                << index << " capture count mismatch";
      for (auto [argument, capture] :
           llvm::zip(workerBlock.getArguments(), captures)) {
        if (argument.getType() != capture.getType())
          return ws.emitOpError()
                 << "mthreads TLE static warp_specialize worker " << index
                 << " capture type mismatch";
      }
    }

    DominanceInfo dominance(func);
    for (Value capture : captures) {
      if (!dominance.dominates(capture, ws.getOperation()))
        return ws.emitOpError(
            "mthreads TLE static warp_specialize capture must dominate the "
            "partition split");
    }

    int32_t baseNumWarps = ttg::lookupNumWarps(ws);
    if (baseNumWarps <= 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize default warp count must be "
          "positive");
    int32_t warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    if (warpSize <= 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize warp size must be positive");

    int64_t totalNumWarps64 = baseNumWarps;
    SmallVector<int32_t> expectedStartIds;
    expectedStartIds.reserve(partitionNumWarps.size());
    for (int32_t workerWarps : partitionNumWarps) {
      if (totalNumWarps64 > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "mthreads TLE static warp_specialize warp count overflow");
      expectedStartIds.push_back(static_cast<int32_t>(totalNumWarps64));
      totalNumWarps64 += workerWarps;
    }
    int64_t totalThreads64 = totalNumWarps64 * warpSize;
    if (totalNumWarps64 > std::numeric_limits<int32_t>::max() ||
        totalThreads64 > std::numeric_limits<int32_t>::max())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize thread count overflow");

    int32_t totalNumWarps = static_cast<int32_t>(totalNumWarps64);
    if (auto existingStartIds = ws.getWarpGroupStartIds()) {
      if (!llvm::equal(*existingStartIds, expectedStartIds))
        return ws.emitOpError(
            "mthreads TLE static worker start ids must form contiguous "
            "ranges after the default partition");
    } else {
      ws.setWarpGroupStartIds(expectedStartIds);
    }
    if (auto existing =
            mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
      if (existing.getInt() != totalNumWarps)
        return ws.emitOpError(
            "mthreads TLE static warp_specialize conflicts with existing "
            "ttg.total-num-warps");
    } else {
      mod->setAttr("ttg.total-num-warps",
                   rewriter.getI32IntegerAttr(totalNumWarps));
    }

    return success();
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<ttg::WarpSpecializeOp> marked;
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty())
      return;
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "mthreads TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(prepareWarpSpecialize(mod, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
