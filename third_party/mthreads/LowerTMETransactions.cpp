#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERTMETRANSACTIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static FailureOr<int32_t>
resolveIssueThread(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  auto ws = copy->getParentOfType<ttg::WarpSpecializeOp>();
  if (!ws)
    return 0;

  Region *copyRegion = copy->getParentRegion();
  bool defaultProducer = ws->hasAttr("musa_tle.default_producer");
  if (defaultProducer) {
    if (!ws.getDefaultRegion().isAncestor(copyRegion)) {
      copy.emitOpError(
          "mthreads TLE default producer TME copy must be in the default "
          "partition");
      return failure();
    }
    return 0;
  }
  if (ws.getDefaultRegion().isAncestor(copyRegion)) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the producer partition");
    return failure();
  }

  std::optional<unsigned> partitionIndex;
  auto partitionRegions = ws.getPartitionRegions();
  for (auto [index, region] : llvm::enumerate(partitionRegions)) {
    if (region->isAncestor(copyRegion)) {
      partitionIndex = index;
      break;
    }
  }
  if (!partitionIndex || *partitionIndex + 1 != partitionRegions.size()) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the final producer "
        "partition");
    return failure();
  }

  ModuleOp module = copy->getParentOfType<ModuleOp>();
  auto numWarps = module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarp =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!numWarps || !threadsPerWarp || numWarps.getInt() <= 0 ||
      threadsPerWarp.getInt() <= 0) {
    copy.emitOpError("mthreads TLE producer issue thread requires "
                     "ttg.num-warps and ttg.threads-per-warp");
    return failure();
  }

  int64_t producerStartWarp = numWarps.getInt();
  ArrayRef<int32_t> workerNumWarps = ws.getPartitionNumWarps();
  if (workerNumWarps.size() != partitionRegions.size()) {
    copy.emitOpError(
        "mthreads TLE producer issue thread requires matching worker warp "
        "counts");
    return failure();
  }
  for (unsigned index = 0; index < *partitionIndex; ++index)
    producerStartWarp += workerNumWarps[index];
  int64_t issueThread = producerStartWarp * threadsPerWarp.getInt();
  if (issueThread > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("mthreads TLE producer issue thread exceeds int32 range");
    return failure();
  }
  return static_cast<int32_t>(issueThread);
}

class LowerTMETransactionsPass
    : public impl::TritonMUSAGPUTLELowerTMETransactionsBase<
          LowerTMETransactionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> copies;
    module.walk([&](ttmg::AsyncTMECopyGlobalToLocalOp copy) {
      if (copy->hasAttr(ttmg::kTLEExpectBytesAttr) ||
          copy->hasAttr(ttmg::kTLEGroupedCompletionAttr))
        copies.push_back(copy);
    });

    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : copies) {
      auto expectBytes =
          copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
      bool groupedCompletion =
          copy->hasAttr(ttmg::kTLEGroupedCompletionAttr);
      if (expectBytes && (!expectBytes.getType().isInteger(32) ||
                          expectBytes.getInt() <= 0)) {
        copy.emitOpError(
            "mthreads TLE completion TME copy requires positive expect_bytes");
        signalPassFailure();
        return;
      }
      if (!expectBytes && !groupedCompletion) {
        copy.emitOpError("mthreads TLE completion TME copy requires transaction "
                         "bytes or a grouped completion marker");
        signalPassFailure();
        return;
      }

      SmallVector<int32_t> transactionParts;
      if (auto parts = copy->getAttrOfType<DenseI32ArrayAttr>(
              ttmg::kTLEExpectBytesPartsAttr)) {
        if (!expectBytes || !groupedCompletion || parts.empty()) {
          copy.emitOpError(
              "mthreads TLE transaction byte parts require grouped completion "
              "with a positive expect_bytes leader");
          signalPassFailure();
          return;
        }
        int64_t totalParts = 0;
        for (int32_t part : parts.asArrayRef()) {
          if (part <= 0 || totalParts > std::numeric_limits<int32_t>::max() - part) {
            copy.emitOpError(
                "mthreads TLE transaction byte parts must be positive and fit "
                "in i32");
            signalPassFailure();
            return;
          }
          totalParts += part;
          transactionParts.push_back(part);
        }
        if (totalParts != expectBytes.getInt()) {
          copy.emitOpError(
              "mthreads TLE transaction byte parts must sum to expect_bytes");
          signalPassFailure();
          return;
        }
      } else if (expectBytes) {
        transactionParts.push_back(expectBytes.getInt());
      }

      FailureOr<int32_t> issueThread = resolveIssueThread(copy);
      if (failed(issueThread)) {
        signalPassFailure();
        return;
      }

      Location loc = copy.getLoc();
      auto issueThreadAttr = rewriter.getI32IntegerAttr(*issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();

      copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
      copy->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
      if (expectBytes) {
        Operation *lastAddTrans = nullptr;
        rewriter.setInsertionPoint(copy);
        for (int32_t part : transactionParts) {
          Value bytes = arith::ConstantIntOp::create(rewriter, loc, part, 32);
          auto addTrans = ttmg::BarrierAddTransOp::create(
              rewriter, loc, copy.getBarId(), bytes, copy.getPred());
          addTrans->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
          addTrans->setAttr(ttmg::kTMEExplicitCompletionAttr,
                            explicitCompletionAttr);
          lastAddTrans = addTrans.getOperation();
        }
        if (groupedCompletion)
          rewriter.setInsertionPointAfter(lastAddTrans);
        else
          rewriter.setInsertionPointAfter(copy);
        auto arrive = ttmg::ArriveBarrierNoRetOp::create(
            rewriter, loc, copy.getBarId(), copy.getPred());
        arrive->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        arrive->setAttr(ttmg::kTMEExplicitCompletionAttr,
                        explicitCompletionAttr);
      }
      copy->removeAttr(ttmg::kTLEExpectBytesAttr);
      copy->removeAttr(ttmg::kTLEExpectBytesPartsAttr);
      copy->removeAttr(ttmg::kTLEGroupedCompletionAttr);
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
