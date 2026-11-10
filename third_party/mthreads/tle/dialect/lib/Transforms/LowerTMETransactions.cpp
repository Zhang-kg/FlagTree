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
  if (ws.getDefaultRegion().isAncestor(copyRegion))
    return 0;

  std::optional<unsigned> partitionIndex;
  for (auto [index, region] : llvm::enumerate(ws.getPartitionRegions())) {
    if (region->isAncestor(copyRegion)) {
      partitionIndex = index;
      break;
    }
  }
  if (!partitionIndex) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the default or a worker "
        "partition");
    return failure();
  }

  ModuleOp module = copy->getParentOfType<ModuleOp>();
  auto threadsPerWarp =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!threadsPerWarp || threadsPerWarp.getInt() <= 0) {
    copy.emitOpError("mthreads TLE worker issue thread requires positive "
                     "ttg.threads-per-warp");
    return failure();
  }

  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (partitionNumWarps.size() != ws.getPartitionRegions().size()) {
    copy.emitOpError(
        "mthreads TLE worker issue thread requires one warp count per "
        "partition");
    return failure();
  }

  int64_t startWarp = 0;
  if (std::optional<ArrayRef<int32_t>> startIds =
          ws.getWarpGroupStartIds()) {
    if (startIds->size() != ws.getPartitionRegions().size()) {
      copy.emitOpError(
          "mthreads TLE worker issue thread requires one start ID per "
          "partition");
      return failure();
    }
    startWarp = (*startIds)[*partitionIndex];
  } else {
    auto numWarps =
        module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
    if (!numWarps || numWarps.getInt() <= 0) {
      copy.emitOpError("mthreads TLE worker issue thread requires positive "
                       "ttg.num-warps before warp group allocation");
      return failure();
    }
    startWarp = numWarps.getInt();
    if (startWarp > std::numeric_limits<int32_t>::max()) {
      copy.emitOpError("mthreads TLE worker issue thread exceeds int32 range");
      return failure();
    }
    for (unsigned index = 0; index < *partitionIndex; ++index) {
      if (partitionNumWarps[index] <= 0) {
        copy.emitOpError(
            "mthreads TLE worker issue thread requires positive partition "
            "warp counts");
        return failure();
      }
      if (partitionNumWarps[index] >
          std::numeric_limits<int32_t>::max() - startWarp) {
        copy.emitOpError(
            "mthreads TLE worker issue thread exceeds int32 range");
        return failure();
      }
      startWarp += partitionNumWarps[index];
    }
  }

  if (partitionNumWarps[*partitionIndex] <= 0 || startWarp < 0) {
    copy.emitOpError("mthreads TLE worker issue thread requires a positive "
                     "partition warp count and non-negative start ID");
    return failure();
  }

  if (startWarp >
      std::numeric_limits<int32_t>::max() / threadsPerWarp.getInt()) {
    copy.emitOpError("mthreads TLE worker issue thread exceeds int32 range");
    return failure();
  }
  int64_t issueThread = startWarp * threadsPerWarp.getInt();
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
          copy->hasAttr(ttmg::kTLESkipTransactionAccountingAttr))
        copies.push_back(copy);
    });

    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : copies) {
      Attribute skipAccounting =
          copy->getAttr(ttmg::kTLESkipTransactionAccountingAttr);
      if (skipAccounting && copy->hasAttr(ttmg::kTLEExpectBytesAttr)) {
        copy.emitOpError("expect_bytes and skip_transaction_accounting are "
                         "mutually exclusive");
        signalPassFailure();
        return;
      }
      if (skipAccounting && !isa<UnitAttr>(skipAccounting)) {
        copy.emitOpError(
            "skip_transaction_accounting must be a unit attribute");
        signalPassFailure();
        return;
      }
      auto expectBytes =
          copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
      if (!skipAccounting &&
          (!expectBytes || !expectBytes.getType().isInteger(32) ||
           expectBytes.getInt() <= 0)) {
        copy.emitOpError(
            "mthreads TLE completion TME copy requires positive expect_bytes");
        signalPassFailure();
        return;
      }

      FailureOr<int32_t> issueThread = resolveIssueThread(copy);
      if (failed(issueThread)) {
        signalPassFailure();
        return;
      }

      Location loc = copy.getLoc();
      auto issueThreadAttr = rewriter.getI32IntegerAttr(*issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();

      // Secondary copies still belong to the same explicit completion protocol:
      // they must use this writer's issue warp and must not insert barrier0.
      copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
      copy->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);

      if (!skipAccounting) {
        rewriter.setInsertionPoint(copy);
        Value bytes =
            arith::ConstantIntOp::create(rewriter, loc, expectBytes.getInt(), 32);
        auto addTrans = ttmg::BarrierAddTransOp::create(
            rewriter, loc, copy.getBarId(), bytes, copy.getPred());

        // Register all bytes for this writer and one producer arrival before
        // its first copy. Pending bytes keep the generation incomplete until
        // every copy in the commit finishes (MP31 arrive_and_expect_tx).
        auto arrive = ttmg::ArriveBarrierNoRetOp::create(
            rewriter, loc, copy.getBarId(), copy.getPred());

        for (Operation *op : {addTrans.getOperation(), arrive.getOperation()}) {
          op->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
          op->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
        }
      }
      copy->removeAttr(ttmg::kTLEExpectBytesAttr);
      copy->removeAttr(ttmg::kTLESkipTransactionAccountingAttr);
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
