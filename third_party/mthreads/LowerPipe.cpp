#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERPIPE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace tle = triton::tle;
namespace musa_tle = triton::musa_tle;

static int64_t getCapacity(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("capacity").getInt();
}

static OperandRange getFields(Operation *op) {
  if (auto pipe = dyn_cast<tle::PipeCreateOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterAcquireOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterCommitOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterCloseOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeReaderWaitOp>(op))
    return pipe.getFields();
  return cast<tle::PipeReaderReleaseOp>(op).getFields();
}

static bool isPipeOp(Operation *op) {
  return isa<tle::PipeCreateOp, tle::PipeWriterAcquireOp,
             tle::PipeWriterCommitOp, tle::PipeWriterCloseOp,
             tle::PipeReaderWaitOp, tle::PipeReaderReleaseOp>(op);
}

static Value canonicalizePipeField(Value field) {
  while (auto blockArg = dyn_cast<BlockArgument>(field)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!partitions)
      break;
    auto ws = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!ws ||
        blockArg.getArgNumber() >= partitions.getExplicitCaptures().size())
      break;
    field = partitions.getExplicitCaptures()[blockArg.getArgNumber()];
  }
  return field;
}

static Value getMemDescRoot(Value value) {
  Value current = canonicalizePipeField(value);
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = canonicalizePipeField(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = canonicalizePipeField(subslice.getSrc());
      continue;
    }
    return current;
  }
}

static std::string getPipeKey(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << getCapacity(op) << "|";
  op->getAttr("scope").print(os);
  os << "|";
  if (Attribute name = op->getAttr("pipe_name"))
    name.print(os);
  os << "|";
  op->getAttr("field_names").print(os);
  os << "|";
  for (Value field : getFields(op))
    os << getMemDescRoot(field).getAsOpaquePointer() << ",";
  return key;
}

static bool sameIndex(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  APInt lhsValue;
  APInt rhsValue;
  return matchPattern(lhs, m_ConstantInt(&lhsValue)) &&
         matchPattern(rhs, m_ConstantInt(&rhsValue)) && lhsValue == rhsValue;
}

enum class PipeRole { None, Consumer, Producer };

static bool hasDefaultProducer(ttg::WarpSpecializeOp ws) {
  return ws && ws->hasAttr("musa_tle.default_producer");
}

static PipeRole getPipeRole(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &ws.getDefaultRegion())
        return hasDefaultProducer(ws) ? PipeRole::Producer
                                      : PipeRole::Consumer;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      auto ws = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
      for (auto [index, partition] : llvm::enumerate(partitions.getRegions())) {
        if (region == partition)
          return hasDefaultProducer(ws)
                     ? PipeRole::Consumer
                     : (index + 1 == partitions.getNumRegions()
                            ? PipeRole::Producer
                            : PipeRole::Consumer);
      }
    }
    region = parent->getParentRegion();
  }
  return PipeRole::None;
}

static std::optional<std::pair<ttg::WarpSpecializeOp, Region *>>
getEnclosingPartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
      return std::make_pair(
          cast<ttg::WarpSpecializeOp>(partitions->getParentOp()), region);
    region = parent->getParentRegion();
  }
  return std::nullopt;
}

static bool isDefinedInside(Value value, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return region->isAncestor(blockArg.getOwner()->getParent());
  Operation *def = value.getDefiningOp();
  return def && region->isAncestor(def->getParentRegion());
}

static Value captureForUse(Operation *use, Value value) {
  auto partition = getEnclosingPartition(use);
  if (!partition || isDefinedInside(value, partition->second))
    return value;

  ttg::WarpSpecializeOp ws = partition->first;
  ttg::WarpSpecializePartitionsOp partitions = ws.getPartitionOp();
  Region *region = partition->second;
  for (auto [index, capture] :
       llvm::enumerate(partitions.getExplicitCaptures())) {
    if (capture == value)
      return region->getArgument(index);
  }

  partitions->insertOperands(partitions->getNumOperands(), value);
  unsigned captureIndex = partitions->getNumOperands() - 1;
  for (Region *partitionRegion : ws.getPartitionRegions())
    partitionRegion->addArgument(value.getType(), value.getLoc());
  return region->getArgument(captureIndex);
}

static LogicalResult verifyCommonContract(Operation *op) {
  if (getCapacity(op) <= 0)
    return op->emitOpError("requires positive capacity");
  auto scope = op->getAttrOfType<StringAttr>("scope");
  if (!scope || scope.getValue() != "cta")
    return op->emitOpError(
        "initial mthreads tle.pipe supports only scope='cta'");
  if (getFields(op).empty())
    return op->emitOpError("requires at least one payload field");
  if (getFields(op).size() > 3)
    return op->emitOpError(
        "mthreads tle.pipe supports at most three payload fields");
  auto fieldNames = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNames || fieldNames.size() != getFields(op).size())
    return op->emitOpError("requires one field name per payload field");
  if (op->getAttr("reader_name"))
    return op->emitOpError(
        "initial mthreads tle.pipe supports only the default SPSC reader");
  return success();
}

static FailureOr<int32_t> getConsumerWarps(Operation *op) {
  int warps = ttg::lookupNumWarps(op);
  if (warps <= 0 || warps > std::numeric_limits<int32_t>::max()) {
    op->emitOpError("requires a positive consumer warp count");
    return failure();
  }
  return static_cast<int32_t>(warps);
}

static Region *getConsumerExecutionRegion(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &ws.getDefaultRegion())
        return region;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      for (Region &partition : partitions.getPartitionRegions()) {
        if (region == &partition)
          return region;
      }
    }
    region = parent->getParentRegion();
  }
  if (auto func = op->getParentOfType<tt::FuncOp>())
    return &func.getBody();
  return nullptr;
}

static FailureOr<int32_t> getTransactionBytes(ttg::TMACopyOp copy) {
  auto descTy = dyn_cast<tt::TensorDescType>(copy.getSrc().getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(copy.getDst().getType());
  if (!descTy || !memDescTy) {
    copy.emitOpError("initial mthreads tle.pipe requires a tensor-descriptor "
                     "to shared-memory TME copy");
    return failure();
  }
  auto blockTy = descTy.getSignlessBlockType();
  if (blockTy.getShape() != memDescTy.getShape() ||
      blockTy.getElementType() != memDescTy.getElementType()) {
    copy.emitOpError("pipe TME descriptor block must match the destination "
                     "slot shape and element type");
    return failure();
  }

  int64_t elements = 1;
  for (int64_t dim : blockTy.getShape()) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim) {
      copy.emitOpError("cannot infer a positive static TME transaction size");
      return failure();
    }
    elements *= dim;
  }
  unsigned bitWidth = blockTy.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 ||
      elements > std::numeric_limits<int64_t>::max() / bitWidth ||
      (elements * bitWidth) % 8 != 0) {
    copy.emitOpError("TME transaction size must be a whole number of bytes");
    return failure();
  }
  int64_t bytes = elements * bitWidth / 8;
  if (bytes <= 0 || bytes > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("TME transaction bytes exceed the positive i32 range");
    return failure();
  }
  return static_cast<int32_t>(bytes);
}

static bool isExactSlot(Value destination, Value fieldRoot, Value stage) {
  auto index = destination.getDefiningOp<ttg::MemDescIndexOp>();
  return index && getMemDescRoot(index.getSrc()) == fieldRoot &&
         sameIndex(index.getIndex(), stage);
}

struct PipeState {
  tle::PipeCreateOp create;
  SmallVector<Value> fieldRoots;
  int32_t capacity = 0;
  int32_t transactionBytes = -1;
  int32_t consumerWarps = 0;
  DenseMap<Region *, int32_t> consumerPartitionWarps;
  std::optional<bool> warpSpecialized;
  Value fullBase;
  Value emptyBase;
  SmallVector<tle::PipeWriterAcquireOp> acquires;
  SmallVector<tle::PipeWriterCommitOp> commits;
  SmallVector<tle::PipeReaderWaitOp> waits;
  SmallVector<tle::PipeReaderReleaseOp> releases;
};

static LogicalResult recordConsumerWarps(PipeState &state, Operation *op) {
  FailureOr<int32_t> warps = getConsumerWarps(op);
  if (failed(warps))
    return failure();
  Region *executionRegion = getConsumerExecutionRegion(op);
  if (!executionRegion)
    return op->emitOpError("cannot identify the reader execution region");
  auto [it, inserted] =
      state.consumerPartitionWarps.try_emplace(executionRegion, *warps);
  if (!inserted) {
    if (it->second != *warps)
      return op->emitOpError(
          "reader warp count changed within one execution partition");
    return success();
  }
  if (state.consumerWarps > std::numeric_limits<int32_t>::max() - *warps)
    return op->emitOpError("total consumer warp count exceeds int32 range");
  state.consumerWarps += *warps;
  return success();
}

static LogicalResult recordExecutionMode(PipeState &state, Operation *op,
                                         PipeRole actualRole,
                                         PipeRole warpSpecializedRole,
                                         StringRef endpoint,
                                         bool allowProducerConsumer = false) {
  bool fusedProducerConsumer =
      allowProducerConsumer && warpSpecializedRole == PipeRole::Consumer &&
      actualRole == PipeRole::Producer;
  if (actualRole != PipeRole::None && actualRole != warpSpecializedRole &&
      !fusedProducerConsumer)
    return op->emitOpError()
           << "requires " << endpoint
           << " operations either outside warp_specialize or in the "
           << (warpSpecializedRole == PipeRole::Producer
                   ? "final worker partition"
                   : "default or consumer worker partition");

  bool usesWarpSpecialize = actualRole != PipeRole::None;
  if (state.warpSpecialized && *state.warpSpecialized != usesWarpSpecialize)
    return op->emitOpError(
        "cannot mix warp-specialized and non-warp-specialized endpoints on "
        "one pipe");
  state.warpSpecialized = usesWarpSpecialize;
  return success();
}

class LowerPipePass
    : public impl::TritonMUSAGPUTLELowerPipeBase<LowerPipePass> {
  LogicalResult analyze(ModuleOp module,
                        std::map<std::string, PipeState> &pipes,
                        DenseMap<Operation *, SmallVector<ttg::TMACopyOp>>
                            &commitCopies) {
    SmallVector<Operation *> pipeOps;
    module.walk([&](Operation *op) {
      if (isPipeOp(op))
        pipeOps.push_back(op);
    });

    for (Operation *op : pipeOps) {
      if (failed(verifyCommonContract(op)))
        return failure();
      std::string key = getPipeKey(op);

      if (auto create = dyn_cast<tle::PipeCreateOp>(op)) {
        if (create->getAttrOfType<ArrayAttr>("readers"))
          return create.emitOpError(
              "initial mthreads tle.pipe does not support named readers");
        if (auto oneShot = create->getAttrOfType<BoolAttr>("one_shot");
            oneShot && oneShot.getValue())
          return create.emitOpError(
              "initial mthreads tle.pipe does not support one_shot=True");
        if (!create->getParentOfType<tt::FuncOp>() ||
            create->getParentOfType<ttg::WarpSpecializeOp>())
          return create.emitOpError(
              "requires pipe.create outside warp_specialize");
        if (pipes.find(key) != pipes.end())
          return create.emitOpError("duplicates an existing pipe identity");
        PipeState state;
        state.create = create;
        for (Value field : create.getFields()) {
          Value root = getMemDescRoot(field);
          if (llvm::is_contained(state.fieldRoots, root))
            return create.emitOpError(
                "requires payload fields with distinct shared-memory roots");
          state.fieldRoots.push_back(root);
        }
        state.capacity = static_cast<int32_t>(getCapacity(op));
        pipes.emplace(key, std::move(state));
        continue;
      }

      auto it = pipes.find(key);
      if (it == pipes.end())
        return op->emitOpError("requires a preceding matching pipe.create");
      PipeState &state = it->second;

      if (auto close = dyn_cast<tle::PipeWriterCloseOp>(op))
        return close.emitOpError(
            "initial mthreads tle.pipe does not support writer.close");
      if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op)) {
        if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                       PipeRole::Producer, "writer")))
          return failure();
        state.acquires.push_back(acquire);
        continue;
      }
      if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
        if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                       PipeRole::Producer, "writer")))
          return failure();

        tle::PipeWriterAcquireOp matchingAcquire;
        SmallVector<SmallVector<ttg::TMACopyOp>> matchingFieldCopies(
            state.fieldRoots.size());
        for (Operation *previous = commit->getPrevNode(); previous;
             previous = previous->getPrevNode()) {
          if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(previous)) {
            if (getPipeKey(previous) == key &&
                sameIndex(acquire.getStage(), commit.getStage())) {
              matchingAcquire = acquire;
              break;
            }
            continue;
          }
          if (auto copy = dyn_cast<ttg::TMACopyOp>(previous)) {
            for (auto [fieldIndex, fieldRoot] :
                 llvm::enumerate(state.fieldRoots)) {
              if (isExactSlot(copy.getDst(), fieldRoot, commit.getStage()))
                matchingFieldCopies[fieldIndex].push_back(copy);
            }
          }
        }
        if (!matchingAcquire)
          return commit.emitOpError(
              "requires a same-block, same-stage matching writer.acquire");
        SmallVector<ttg::TMACopyOp> matchingCopies;
        int64_t totalBytes = 0;
        for (auto [fieldIndex, fieldCopies] :
             llvm::enumerate(matchingFieldCopies)) {
          if (fieldCopies.size() != 1)
            // Preserve the stable diagnostic prefix consumed by existing
            // frontend tests while retaining the field index for debugging
            // multi-payload producer protocols.
            return commit.emitOpError()
                   << "requires exactly one TME copy between acquire and commit; found "
                   << fieldCopies.size() << " (payload field " << fieldIndex
                   << ")";
          ttg::TMACopyOp copy = fieldCopies.front();
          if (copy.getCompletionBarrier())
            return copy.emitOpError(
                "pipe-managed TME copy must not provide an explicit barrier");
          FailureOr<int32_t> bytes = getTransactionBytes(copy);
          if (failed(bytes))
            return failure();
          totalBytes += *bytes;
          if (totalBytes > std::numeric_limits<int32_t>::max())
            return commit.emitOpError(
                "total TME transaction bytes exceed positive i32 range");
          matchingCopies.push_back(copy);
        }
        if (state.transactionBytes >= 0 &&
            state.transactionBytes != totalBytes)
          return commit.emitOpError(
              "all commits on one pipe must use identical transaction bytes");
        state.transactionBytes = static_cast<int32_t>(totalBytes);
        state.commits.push_back(commit);
        commitCopies[commit.getOperation()] = std::move(matchingCopies);
        continue;
      }
      if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                       PipeRole::Consumer, "reader",
                                       /*allowProducerConsumer=*/true)))
          return failure();
        if (failed(recordConsumerWarps(state, op)))
          return failure();
        state.waits.push_back(wait);
        continue;
      }

      auto release = cast<tle::PipeReaderReleaseOp>(op);
      if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                     PipeRole::Consumer, "reader",
                                     /*allowProducerConsumer=*/true)))
        return failure();
      bool matchingWait = false;
      for (Operation *previous = release->getPrevNode(); previous;
           previous = previous->getPrevNode()) {
        if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(previous)) {
          if (getPipeKey(previous) == key &&
              sameIndex(wait.getStage(), release.getStage())) {
            matchingWait = true;
            break;
          }
        }
      }
      if (!matchingWait)
        return release.emitOpError(
            "requires a same-block, same-stage matching reader.wait");
      if (failed(recordConsumerWarps(state, op)))
        return failure();
      state.releases.push_back(release);
    }

    for (auto &[key, state] : pipes) {
      if (state.acquires.empty() || state.commits.empty())
        return state.create.emitOpError(
            "requires at least one writer acquire/commit pair");
      if (state.waits.empty() || state.releases.empty())
        return state.create.emitOpError(
            "requires at least one reader wait/release pair");
      if (state.transactionBytes <= 0 || state.consumerWarps <= 0)
        return state.create.emitOpError(
            "could not infer transaction bytes or consumer warp count");
    }
    return success();
  }

  static Value toI32Phase(OpBuilder &builder, Location loc, Value phase,
                          bool invert) {
    Value value = phase;
    if (invert) {
      Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
      value = arith::XOrIOp::create(builder, loc, value, one);
    }
    return arith::ExtUIOp::create(builder, loc, builder.getI32Type(), value);
  }

  static Value createIndex(OpBuilder &builder, Location loc, Operation *use,
                           Value base, Value stage) {
    Value capturedBase = captureForUse(use, base);
    return musa_tle::BarrierIndexOp::create(builder, loc, capturedBase, stage);
  }

  LogicalResult rewrite(ModuleOp module,
                        std::map<std::string, PipeState> &pipes,
                        DenseMap<Operation *, SmallVector<ttg::TMACopyOp>>
                            &commitCopies) {
    SmallVector<Operation *> pipeOps;
    module.walk([&](Operation *op) {
      if (isPipeOp(op))
        pipeOps.push_back(op);
    });

    for (Operation *op : pipeOps) {
      PipeState &state = pipes.at(getPipeKey(op));
      OpBuilder builder(op);
      Location loc = op->getLoc();

      if (auto create = dyn_cast<tle::PipeCreateOp>(op)) {
        auto capacity = builder.getI32IntegerAttr(state.capacity);
        auto one = builder.getI32IntegerAttr(1);
        auto pending = builder.getI32IntegerAttr(0);
        auto ready = builder.getI32IntegerAttr(1);
        auto bytes = builder.getI32IntegerAttr(state.transactionBytes);
        state.fullBase = musa_tle::BarrierAllocOp::create(
            builder, loc, capacity, one, pending, bytes);
        state.emptyBase = musa_tle::BarrierAllocOp::create(
            builder, loc, capacity,
            builder.getI32IntegerAttr(state.consumerWarps), ready,
            IntegerAttr());
        create.erase();
        continue;
      }

      if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op)) {
        Value barrier =
            createIndex(builder, loc, op, state.emptyBase, acquire.getStage());
        Value phase = toI32Phase(builder, loc, acquire.getPhase(), true);
        musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        acquire.erase();
        continue;
      }

      if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
        auto copies = commitCopies.lookup(op);
        if (copies.size() != state.fieldRoots.size())
          return commit.emitOpError("lost the analyzed pipe TME copies");
        for (auto [index, copy] : llvm::enumerate(copies)) {
          OpBuilder copyBuilder(copy);
          Value barrier = createIndex(copyBuilder, copy.getLoc(), copy,
                                      state.fullBase, commit.getStage());
          auto replacement = ttg::TMACopyOp::create(
              copyBuilder, copy.getLoc(), copy.getSrc(), copy.getDst(),
              copy.getIndices(), barrier);
          replacement->setDiscardableAttrs(
              copy->getDiscardableAttrDictionary());
          replacement->setAttr(triton::musa::kTLEGroupedCompletionAttr,
                               copyBuilder.getUnitAttr());
          if (index == 0)
            replacement->setAttr(
                "expect_bytes",
                copyBuilder.getI32IntegerAttr(state.transactionBytes));
          if (index == 0 && copies.size() > 1) {
            SmallVector<int32_t> partBytes;
            partBytes.reserve(copies.size());
            for (ttg::TMACopyOp partCopy : copies) {
              FailureOr<int32_t> bytes = getTransactionBytes(partCopy);
              if (failed(bytes))
                return failure();
              partBytes.push_back(*bytes);
            }
            replacement->setAttr(
                triton::musa::kTLEExpectBytesPartsAttr,
                DenseI32ArrayAttr::get(replacement->getContext(), partBytes));
          }
          copy.erase();
        }
        commit.erase();
        continue;
      }

      if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        Value barrier =
            createIndex(builder, loc, op, state.fullBase, wait.getStage());
        Value phase = toI32Phase(builder, loc, wait.getPhase(), false);
        musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        if (!wait.getIsClosed().use_empty()) {
          Value notClosed = arith::ConstantIntOp::create(builder, loc, 0, 1);
          wait.getIsClosed().replaceAllUsesWith(notClosed);
        }
        wait.erase();
        continue;
      }

      if (isa<tle::PipeWriterCloseOp>(op))
        return op->emitOpError("writer.close must have failed analysis");

      auto release = cast<tle::PipeReaderReleaseOp>(op);
      Value barrier =
          createIndex(builder, loc, op, state.emptyBase, release.getStage());
      Value phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
      musa_tle::BarrierArriveOp::create(builder, loc, barrier, phase,
                                        builder.getI32IntegerAttr(1));
      release.erase();
    }

    bool hasPipeOps = false;
    module.walk([&](Operation *op) { hasPipeOps |= isPipeOp(op); });
    if (hasPipeOps)
      return module.emitError("mthreads TLE pipe lowering left lifecycle ops");
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::map<std::string, PipeState> pipes;
    DenseMap<Operation *, SmallVector<ttg::TMACopyOp>> commitCopies;
    if (failed(analyze(module, pipes, commitCopies)) ||
        failed(rewrite(module, pipes, commitCopies)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
