#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
static constexpr StringLiteral kLocalSyncIntrinsic = "llvm.musa.syncthreads.lm";
static constexpr StringLiteral kCtaSyncIntrinsic = "llvm.musa.barrier0";
static constexpr StringLiteral kWarpSyncIntrinsic = "llvm.musa.syncwarp";
static constexpr StringLiteral kBarRecordIntrinsic =
    "llvm.musa.async.bar.record";

struct RoleSyncs {
  int32_t numWarps = 0;
  SmallVector<LLVM::CallIntrinsicOp> syncs;
  Value barrierId;
};

static bool isCtaSync(LLVM::CallIntrinsicOp call) {
  return call.getIntrin() == kLocalSyncIntrinsic ||
         call.getIntrin() == kCtaSyncIntrinsic;
}

static LogicalResult lowerWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                            ttg::WarpSpecializeOp ws,
                                            IRRewriter &rewriter) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto defaultWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  if (!defaultWarpsAttr || defaultWarpsAttr.getInt() <= 0 ||
      defaultWarpsAttr.getInt() > std::numeric_limits<int32_t>::max())
    return ws.emitOpError(
        "mthreads TLE role synchronization requires positive int32 "
        "ttg.num-warps");
  int32_t defaultWarps = static_cast<int32_t>(defaultWarpsAttr.getInt());

  auto partitionRegions = ws.getPartitionRegions();
  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (partitionRegions.size() != partitionNumWarps.size())
    return ws.emitOpError(
        "mthreads TLE role synchronization requires matching worker regions "
        "and warp counts");

  SmallVector<RoleSyncs> roles;
  roles.reserve(partitionRegions.size() + 1);
  auto collectRoleSyncs = [&](Region &region, int32_t numWarps) {
    roles.emplace_back();
    RoleSyncs &role = roles.back();
    role.numWarps = numWarps;
    region.walk([&](LLVM::CallIntrinsicOp call) {
      if (isCtaSync(call))
        role.syncs.push_back(call);
    });
  };
  collectRoleSyncs(ws.getDefaultRegion(), defaultWarps);
  for (auto [index, region] : llvm::enumerate(partitionRegions)) {
    if (partitionNumWarps[index] <= 0)
      return ws.emitOpError() << "mthreads TLE role synchronization worker "
                              << index << " must have a positive warp count";
    collectRoleSyncs(*region, partitionNumWarps[index]);
  }

  SmallVector<unsigned> asyncRoleIndices;
  for (auto [index, role] : llvm::enumerate(roles)) {
    if (role.syncs.empty())
      continue;
    if (role.numWarps == 1) {
      for (LLVM::CallIntrinsicOp sync : role.syncs) {
        rewriter.setInsertionPoint(sync);
        LLVM::CallIntrinsicOp::create(
            rewriter, sync.getLoc(), rewriter.getStringAttr(kWarpSyncIntrinsic),
            ValueRange{});
        rewriter.eraseOp(sync);
      }
      continue;
    }
    asyncRoleIndices.push_back(index);
  }

  if (asyncRoleIndices.empty())
    return success();
  if (asyncRoleIndices.size() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    return ws.emitOpError(
        "mthreads TLE role synchronization barrier count overflow");

  auto reserved = ttmg::reserveBarrierIdRange(
      roles[asyncRoleIndices.front()].syncs.front(),
      static_cast<int32_t>(asyncRoleIndices.size()));
  if (failed(reserved))
    return roles[asyncRoleIndices.front()].syncs.front().emitOpError(
        "mthreads TLE partition synchronization exhausted hardware barrier "
        "ids");

  LLVM::CallIntrinsicOp initializationRendezvous;
  for (Operation *op = ws->getPrevNode(); op; op = op->getPrevNode()) {
    auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
    if (call && isCtaSync(call)) {
      initializationRendezvous = call;
      break;
    }
  }

  Location loc = initializationRendezvous ? initializationRendezvous.getLoc()
                                          : ws.getLoc();
  if (initializationRendezvous)
    rewriter.setInsertionPoint(initializationRendezvous);
  else
    rewriter.setInsertionPoint(ws);
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  for (unsigned roleIndex : asyncRoleIndices) {
    RoleSyncs &role = roles[roleIndex];
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    // MUSA async barriers count converged warp arrivals, not individual lanes.
    // Every role executes one arrive instruction per participating warp.
    Value count =
        arith::ConstantIntOp::create(rewriter, loc, role.numWarps, 32);
    initializationArgs.push_back({id, count});
    role.barrierId = id;
  }

  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value issueInit =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, tid, zero);
  auto initIf = scf::IfOp::create(rewriter, loc, issueInit, false);
  rewriter.setInsertionPointToStart(&initIf.getThenRegion().front());
  for (auto [id, count] : initializationArgs)
    LLVM::CallIntrinsicOp::create(
        rewriter, loc, rewriter.getStringAttr("llvm.musa.async.init.arrival"),
        ValueRange{id, count, zero});

  rewriter.setInsertionPointAfter(initIf);
  if (!initializationRendezvous)
    LLVM::CallIntrinsicOp::create(rewriter, loc,
                                  rewriter.getStringAttr(kLocalSyncIntrinsic),
                                  ValueRange{});

  // A multi-warp role owns one reusable barrier. Every participating thread
  // arrives, and the returned phase selects that dynamic barrier generation.
  for (unsigned roleIndex : asyncRoleIndices) {
    RoleSyncs &role = roles[roleIndex];
    for (LLVM::CallIntrinsicOp sync : role.syncs) {
      rewriter.setInsertionPoint(sync);
      // Preserve an explicit warp convergence point before the role's
      // collective arrival, including after lane-predicated scalar work.
      LLVM::CallIntrinsicOp::create(
          rewriter, sync.getLoc(), rewriter.getStringAttr(kWarpSyncIntrinsic),
          ValueRange{});
      Value phase = LLVM::CallIntrinsicOp::create(
                        rewriter, sync.getLoc(), rewriter.getI32Type(),
                        rewriter.getStringAttr("llvm.musa.async.arrive"),
                        ValueRange{role.barrierId})
                        .getResult(0);
      LLVM::CallIntrinsicOp::create(
          rewriter, sync.getLoc(),
          rewriter.getStringAttr("llvm.musa.async.wait"),
          ValueRange{role.barrierId, phase});
      rewriter.eraseOp(sync);
    }
  }

  SmallVector<LLVM::CallIntrinsicOp> oldBarRecords;
  func.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() == kBarRecordIntrinsic)
      oldBarRecords.push_back(call);
  });
  for (LLVM::CallIntrinsicOp record : oldBarRecords)
    rewriter.eraseOp(record);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  Value barCount = arith::ConstantIntOp::create(
      rewriter, func.getLoc(), ttmg::getReservedBarrierCount(func), 32);
  LLVM::CallIntrinsicOp::create(rewriter, func.getLoc(),
                                rewriter.getStringAttr(kBarRecordIntrinsic),
                                ValueRange{barCount});
  return success();
}

static LogicalResult lowerStaticWarpSpecialize(LLVM::LLVMFuncOp func,
                                               ttg::WarpSpecializeOp ws,
                                               IRRewriter &rewriter) {
  auto partitionRegions = ws.getPartitionRegions();
  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (ws.getNumResults() != 0 || partitionRegions.empty() ||
      partitionRegions.size() != partitionNumWarps.size())
    return ws.emitOpError(
        "mthreads TLE late lowering requires one default region, at least one "
        "worker region with a matching warp count, and no results");

  Region &defaultRegion = ws.getDefaultRegion();
  if (defaultRegion.empty())
    return ws.emitOpError("mthreads TLE static partitions must not be empty");
  for (auto [index, region] : llvm::enumerate(partitionRegions)) {
    if (region->empty())
      return ws.emitOpError()
             << "mthreads TLE static worker " << index << " must not be empty";
  }

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  for (auto [index, region] : llvm::enumerate(partitionRegions)) {
    Block &workerEntry = region->front();
    if (workerEntry.getNumArguments() != captures.size())
      return ws.emitOpError()
             << "worker " << index << " capture count changed during lowering";
    for (auto [argument, capture] :
         llvm::zip(workerEntry.getArguments(), captures)) {
      if (argument.getType() != capture.getType())
        return ws.emitOpError()
               << "worker " << index
               << " capture types were not converted consistently";
    }
  }
  for (Region *region : partitionRegions) {
    Block &workerEntry = region->front();
    for (auto [argument, capture] :
         llvm::zip(workerEntry.getArguments(), captures))
      argument.replaceAllUsesWith(capture);
    workerEntry.eraseArguments([](BlockArgument) { return true; });
  }

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto defaultWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!defaultWarpsAttr || !threadsPerWarpAttr ||
      defaultWarpsAttr.getInt() <= 0 ||
      defaultWarpsAttr.getInt() > std::numeric_limits<int32_t>::max() ||
      threadsPerWarpAttr.getInt() <= 0 ||
      threadsPerWarpAttr.getInt() > std::numeric_limits<int32_t>::max())
    return ws.emitOpError("late lowering requires positive ttg.num-warps and "
                          "ttg.threads-per-warp");
  int32_t defaultWarps = static_cast<int32_t>(defaultWarpsAttr.getInt());
  int32_t threadsPerWarp = static_cast<int32_t>(threadsPerWarpAttr.getInt());

  auto startIdsAttr = ws.getWarpGroupStartIds();
  if (!startIdsAttr || startIdsAttr->size() != partitionRegions.size())
    return ws.emitOpError(
        "late lowering requires one start warp id for each worker region");
  ArrayRef<int32_t> startIds = *startIdsAttr;

  int64_t expectedStartWarp = defaultWarps;
  int64_t defaultEndThreads64 = expectedStartWarp * threadsPerWarp;
  if (defaultEndThreads64 > std::numeric_limits<int32_t>::max())
    return ws.emitOpError(
        "static default partition boundary exceeds int32 range");
  int32_t defaultEndThreads = static_cast<int32_t>(defaultEndThreads64);
  SmallVector<int32_t> workerEndThreads;
  workerEndThreads.reserve(partitionRegions.size());
  for (size_t index = 0; index < partitionRegions.size(); ++index) {
    int32_t workerWarps = partitionNumWarps[index];
    if (workerWarps <= 0 || startIds[index] != expectedStartWarp)
      return ws.emitOpError()
             << "worker " << index
             << " must occupy a positive contiguous warp range";
    expectedStartWarp += workerWarps;
    int64_t endThreads64 = expectedStartWarp * threadsPerWarp;
    if (expectedStartWarp > std::numeric_limits<int32_t>::max() ||
        endThreads64 > std::numeric_limits<int32_t>::max())
      return ws.emitOpError()
             << "worker " << index << " partition boundary exceeds int32 range";
    workerEndThreads.push_back(static_cast<int32_t>(endThreads64));
  }

  SmallVector<ttg::WarpReturnOp> workerReturns;
  for (auto [index, region] : llvm::enumerate(partitionRegions)) {
    size_t previousSize = workerReturns.size();
    region->walk([&](ttg::WarpReturnOp op) { workerReturns.push_back(op); });
    if (workerReturns.size() == previousSize)
      return ws.emitOpError()
             << "static worker " << index << " lost its terminator";
  }
  SmallVector<ttg::WarpYieldOp> defaultYields;
  defaultRegion.walk([&](ttg::WarpYieldOp op) { defaultYields.push_back(op); });
  if (defaultYields.empty())
    return ws.emitOpError("static default region lost its terminator");
  for (ttg::WarpReturnOp op : workerReturns) {
    if (op->getNumOperands() != 0)
      return op.emitOpError(
          "mthreads TLE static worker must not return values");
  }
  for (ttg::WarpYieldOp op : defaultYields) {
    if (op.getNumOperands() != 0)
      return op.emitOpError(
          "mthreads TLE static default region must not yield values");
  }

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  SmallVector<Block *> workerRoots;
  workerRoots.reserve(partitionRegions.size());
  for (Region *region : partitionRegions) {
    workerRoots.push_back(&region->front());
    funcBlocks.splice(continuation->getIterator(), region->getBlocks());
  }
  Block *defaultRoot = &defaultRegion.front();
  funcBlocks.splice(continuation->getIterator(), defaultRegion.getBlocks());
  SmallVector<Block *> workerDispatches;
  workerDispatches.reserve(partitionRegions.size());
  for (size_t index = 0; index < partitionRegions.size(); ++index)
    workerDispatches.push_back(
        rewriter.createBlock(&funcBody, continuation->getIterator()));

  for (ttg::WarpReturnOp op : workerReturns) {
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }
  for (ttg::WarpYieldOp op : defaultYields) {
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  Location loc = ws.getLoc();
  // Keep every launched warp alive until all roles have finished. The join
  // also drains cross-role use of CTA-local resources before kernel return.
  // This is outside the role regions and must remain a full-CTA rendezvous.
  rewriter.setInsertionPointToStart(continuation);
  LLVM::CallIntrinsicOp::create(rewriter, loc,
                                rewriter.getStringAttr(kLocalSyncIntrinsic),
                                ValueRange{});
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value defaultEnd =
      arith::ConstantIntOp::create(rewriter, loc, defaultEndThreads, 32);
  Value isDefault = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, tid, defaultEnd);
  cf::CondBranchOp::create(rewriter, loc, isDefault, defaultRoot, ValueRange{},
                           workerDispatches.front(), ValueRange{});

  // The ranges are contiguous, so testing successive upper bounds forms an
  // N-way dispatch without materializing a worker-id lookup table.
  for (size_t index = 0; index < workerDispatches.size(); ++index) {
    Block *workerDispatch = workerDispatches[index];
    Block *next = index + 1 < workerDispatches.size()
                      ? workerDispatches[index + 1]
                      : continuation;
    rewriter.setInsertionPointToEnd(workerDispatch);
    Value workerEnd = arith::ConstantIntOp::create(rewriter, loc,
                                                   workerEndThreads[index], 32);
    Value isWorker = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, tid, workerEnd);
    cf::CondBranchOp::create(rewriter, loc, isWorker, workerRoots[index],
                             ValueRange{}, next, ValueRange{});
  }
  return success();
}

class LowerWarpSpecializePass
    : public impl::TritonMUSAGPUTLELowerWarpSpecializeBase<
          LowerWarpSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ttg::WarpSpecializeOp> marked;
    module.walk([&](ttg::WarpSpecializeOp ws) {
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

    auto func = marked.front()->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      marked.front().emitOpError(
          "mthreads TLE late lowering requires an LLVM function");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(lowerWarpGroupBarriers(func, marked.front(), rewriter)) ||
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
