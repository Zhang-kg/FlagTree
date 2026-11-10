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

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
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
namespace ttmg = triton::musa;
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

struct PipeRole {
  Operation *warpSpecialize = nullptr;
  // -2 is non-warp-specialized code, -1 is the default partition, and worker
  // partitions use their zero-based region index.
  int32_t partition = -2;

  bool operator==(const PipeRole &other) const {
    return warpSpecialize == other.warpSpecialize &&
           partition == other.partition;
  }
  bool operator!=(const PipeRole &other) const { return !(*this == other); }
};

struct PipeRoleLess {
  bool operator()(const PipeRole &lhs, const PipeRole &rhs) const {
    if (lhs.warpSpecialize != rhs.warpSpecialize)
      return std::less<Operation *>{}(lhs.warpSpecialize, rhs.warpSpecialize);
    return lhs.partition < rhs.partition;
  }
};

static PipeRole getPipeRole(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &ws.getDefaultRegion())
        return PipeRole{ws.getOperation(), -1};
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      for (auto [index, partition] : llvm::enumerate(partitions.getRegions())) {
        if (region == partition)
          return PipeRole{partitions->getParentOp(),
                          static_cast<int32_t>(index)};
      }
    }
    region = parent->getParentRegion();
  }
  return PipeRole{};
}

static std::string describePipeRole(const PipeRole &role) {
  if (!role.warpSpecialize)
    return "outside warp_specialize";
  if (role.partition < 0)
    return "default partition";
  return "worker partition " + std::to_string(role.partition);
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
  auto fieldNames = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNames || fieldNames.size() != getFields(op).size())
    return op->emitOpError("requires one field name for every payload field");
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

static FailureOr<int32_t>
resolveReaderIssueThread(Operation *op, const PipeRole &role,
                         int32_t consumerWarps) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto threadsPerWarp =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!threadsPerWarp || threadsPerWarp.getInt() != 32) {
    op->emitOpError(
        "mthreads pipe empty publication requires ttg.threads-per-warp = 32");
    return failure();
  }
  if (!role.warpSpecialize)
    return 0;

  auto ws = dyn_cast<ttg::WarpSpecializeOp>(role.warpSpecialize);
  if (!ws) {
    op->emitOpError("mthreads pipe reader must belong to warp_specialize");
    return failure();
  }
  auto defaultWarps =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  constexpr int64_t maxWarps = std::numeric_limits<int32_t>::max() / 32;
  if (!defaultWarps || defaultWarps.getInt() <= 0 ||
      defaultWarps.getInt() > maxWarps) {
    op->emitOpError("mthreads pipe reader issue thread requires positive "
                    "ttg.num-warps with an int32 thread boundary");
    return failure();
  }
  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (partitionNumWarps.size() != ws.getPartitionRegions().size()) {
    op->emitOpError(
        "mthreads pipe reader issue thread requires one warp count per "
        "partition");
    return failure();
  }
  std::optional<ArrayRef<int32_t>> startIds = ws.getWarpGroupStartIds();
  if (startIds && startIds->size() != partitionNumWarps.size()) {
    op->emitOpError(
        "mthreads pipe reader issue thread requires one start ID per "
        "partition");
    return failure();
  }

  int64_t nextWarp = defaultWarps.getInt();
  int64_t readerStart = role.partition == -1 ? 0 : -1;
  int64_t readerWarps = role.partition == -1 ? nextWarp : -1;
  for (auto [index, warps] : llvm::enumerate(partitionNumWarps)) {
    if (warps <= 0 || (startIds && (*startIds)[index] != nextWarp)) {
      op->emitOpError() << "mthreads pipe reader issue thread requires worker "
                        << index << " to occupy a positive contiguous warp "
                                    "range";
      return failure();
    }
    if (warps > maxWarps - nextWarp) {
      op->emitOpError(
          "mthreads pipe reader partition boundary exceeds int32 range");
      return failure();
    }
    if (role.partition == static_cast<int32_t>(index)) {
      readerStart = nextWarp;
      readerWarps = warps;
    }
    nextWarp += warps;
  }
  if (readerStart < 0 || readerWarps != consumerWarps) {
    op->emitOpError("mthreads pipe reader issue thread requires the declared "
                    "reader role and its consumer warp count to agree");
    return failure();
  }
  return static_cast<int32_t>(readerStart * 32);
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
  Value slot = canonicalizePipeField(destination);
  while (auto subslice = slot.getDefiningOp<ttg::MemDescSubsliceOp>())
    slot = canonicalizePipeField(subslice.getSrc());
  auto index = slot.getDefiningOp<ttg::MemDescIndexOp>();
  return index && getMemDescRoot(index.getSrc()) == fieldRoot &&
         sameIndex(index.getIndex(), stage);
}

struct PipeState {
  tle::PipeCreateOp create;
  SmallVector<Value> fieldRoots;
  SmallVector<std::string> fieldNames;
  int32_t capacity = 0;
  llvm::BitVector fieldsWithCopies;
  int32_t consumerWarps = -1;
  int32_t readerIssueThread = -1;
  // An engaged null pointer records a non-warp-specialized pipe. Otherwise all
  // lifecycle endpoints must belong to the same warp_specialize operation.
  std::optional<Operation *> executionWarpSpecialize;
  std::optional<std::string> readerName;
  std::optional<PipeRole> readerRole;
  SmallVector<std::optional<PipeRole>> fieldOwners;
  std::map<PipeRole, Value, PipeRoleLess> fullBases;
  Value emptyBase;
  SmallVector<tle::PipeReaderWaitOp> waits;
  SmallVector<tle::PipeReaderReleaseOp> releases;

  struct WriterState {
    llvm::BitVector ownedFields;
    bool hasOwnership = false;
    SmallVector<tle::PipeWriterAcquireOp> acquires;
    SmallVector<tle::PipeWriterCommitOp> commits;
  };
  std::map<PipeRole, WriterState, PipeRoleLess> writers;
};

struct FieldCopy {
  unsigned fieldIndex;
  ttg::TMACopyOp copy;
  int32_t transactionBytes;
};

struct CommitPayload {
  SmallVector<FieldCopy> fields;
  int32_t transactionBytes;
  Value stage;
};

using CommitCopies = DenseMap<Operation *, CommitPayload>;

static LogicalResult recordExecutionDomain(PipeState &state, Operation *op,
                                           const PipeRole &role) {
  if (state.executionWarpSpecialize &&
      *state.executionWarpSpecialize != role.warpSpecialize)
    return op->emitOpError(
        "cannot mix lifecycle endpoints from different warp_specialize "
        "operations or from warp-specialized and non-warp-specialized code");
  state.executionWarpSpecialize = role.warpSpecialize;
  return success();
}

static LogicalResult verifyReaderEndpoint(PipeState &state, Operation *op,
                                          const PipeRole &role) {
  auto actualName = op->getAttrOfType<StringAttr>("reader_name");
  if (state.readerName) {
    if (!actualName || actualName.getValue() != *state.readerName)
      return op->emitOpError() << "requires the sole declared reader name '"
                               << *state.readerName << "'";
  } else if (actualName) {
    return op->emitOpError(
        "uses a named reader on a pipe with the default reader");
  }
  if (state.readerRole && *state.readerRole != role)
    return op->emitOpError() << "moves the sole reader from "
                             << describePipeRole(*state.readerRole) << " to "
                             << describePipeRole(role);
  state.readerRole = role;
  return success();
}

class LowerPipePass
    : public impl::TritonMUSAGPUTLELowerPipeBase<LowerPipePass> {
  LogicalResult analyze(ModuleOp module,
                        std::map<std::string, PipeState> &pipes,
                        CommitCopies &commitCopies) {
    SmallVector<Operation *> pipeOps;
    // Track static operation pairs, not dynamic loop iterations: a pair in a
    // loop body executes repeatedly, but must have exactly one partner in IR.
    llvm::DenseSet<Operation *> consumedAcquires;
    llvm::DenseSet<Operation *> consumedWaits;
    llvm::DenseSet<Operation *> consumedCopies;
    module.walk([&](Operation *op) {
      if (isPipeOp(op))
        pipeOps.push_back(op);
    });

    for (Operation *op : pipeOps) {
      if (failed(verifyCommonContract(op)))
        return failure();
      std::string key = getPipeKey(op);

      if (auto create = dyn_cast<tle::PipeCreateOp>(op)) {
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
        state.capacity = static_cast<int32_t>(getCapacity(op));
        auto fieldNames = create->getAttrOfType<ArrayAttr>("field_names");
        llvm::DenseSet<Value> seenRoots;
        for (auto [index, field] : llvm::enumerate(create.getFields())) {
          Value root = getMemDescRoot(field);
          if (!seenRoots.insert(root).second)
            return create.emitOpError()
                   << "requires distinct backing allocations for field '"
                   << cast<StringAttr>(fieldNames[index]).getValue() << "'";
          state.fieldRoots.push_back(root);
          state.fieldNames.push_back(
              cast<StringAttr>(fieldNames[index]).getValue().str());
        }
        state.fieldsWithCopies.resize(state.fieldRoots.size());
        state.fieldOwners.resize(state.fieldRoots.size());

        if (auto readers = create->getAttrOfType<ArrayAttr>("readers")) {
          if (readers.size() != 1)
            return create.emitOpError(
                "supports either the default reader or exactly one named "
                "reader");
          state.readerName = cast<StringAttr>(readers[0]).getValue().str();
        }
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
        PipeRole role = getPipeRole(op);
        if (failed(recordExecutionDomain(state, op, role)))
          return failure();
        state.writers[role].acquires.push_back(acquire);
        continue;
      }
      if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
        PipeRole role = getPipeRole(op);
        if (failed(recordExecutionDomain(state, op, role)))
          return failure();

        tle::PipeWriterAcquireOp matchingAcquire;
        SmallVector<ttg::TMACopyOp> copiesByField(state.fieldRoots.size());
        for (Operation *previous = commit->getPrevNode(); previous;
             previous = previous->getPrevNode()) {
          if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(previous)) {
            if (getPipeKey(previous) == key &&
                sameIndex(acquire.getStage(), commit.getStage()) &&
                getPipeRole(previous) == role) {
              matchingAcquire = acquire;
              break;
            }
            continue;
          }
          if (auto copy = dyn_cast<ttg::TMACopyOp>(previous)) {
            for (auto [fieldIndex, root] : llvm::enumerate(state.fieldRoots)) {
              if (!isExactSlot(copy.getDst(), root, commit.getStage()))
                continue;
              if (copiesByField[fieldIndex])
                return commit.emitOpError()
                       << "has multiple TME copies for pipe field '"
                       << state.fieldNames[fieldIndex]
                       << "' between acquire and commit";
              copiesByField[fieldIndex] = copy;
            }
          }
        }
        if (!matchingAcquire)
          return commit.emitOpError(
              "requires a same-block, same-stage, same-role matching "
              "writer.acquire");
        if (!consumedAcquires.insert(matchingAcquire.getOperation()).second)
          return commit.emitOpError(
              "writer.acquire must be paired with exactly one writer.commit");

        llvm::BitVector ownedFields(state.fieldRoots.size());
        SmallVector<FieldCopy> payload;
        int64_t totalBytes = 0;
        for (auto [fieldIndex, copy] : llvm::enumerate(copiesByField)) {
          if (!copy)
            continue;
          if (copy.getCompletionBarrier())
            return copy.emitOpError(
                "pipe-managed TME copy must not provide an explicit barrier");
          if (copy->hasAttr("expect_bytes") ||
              copy->hasAttr(ttmg::kTLESkipTransactionAccountingAttr))
            return copy.emitOpError(
                "pipe-managed TME copy must not provide transaction accounting "
                "attributes");
          if (!consumedCopies.insert(copy.getOperation()).second)
            return copy.emitOpError(
                "pipe TME copy must belong to exactly one writer.commit");
          FailureOr<int32_t> bytes = getTransactionBytes(copy);
          if (failed(bytes))
            return failure();
          if (*bytes > std::numeric_limits<int32_t>::max() - totalBytes)
            return commit.emitOpError(
                "pipe writer transaction bytes exceed the positive i32 range");
          totalBytes += *bytes;
          ownedFields.set(fieldIndex);
          state.fieldsWithCopies.set(fieldIndex);
          payload.push_back(
              FieldCopy{static_cast<unsigned>(fieldIndex), copy, *bytes});
        }
        if (payload.empty())
          return commit.emitOpError(
              "must produce at least one pipe field with a TME copy");
        // Field declaration order need not match execution order. Register all
        // bytes and the writer's single arrival before its first actual copy.
        // The acquire/commit scan above guarantees that these share a block.
        llvm::sort(payload, [](FieldCopy lhs, FieldCopy rhs) {
          return lhs.copy->isBeforeInBlock(rhs.copy.getOperation());
        });

        PipeState::WriterState &writer = state.writers[role];
        if (writer.hasOwnership && writer.ownedFields != ownedFields)
          return commit.emitOpError("changes field ownership for ")
                 << describePipeRole(role)
                 << "; every commit from one role must cover the same "
                    "non-empty "
                    "field subset";
        if (!writer.hasOwnership) {
          for (int fieldIndex : ownedFields.set_bits()) {
            if (state.fieldOwners[fieldIndex] &&
                *state.fieldOwners[fieldIndex] != role)
              return commit.emitOpError()
                     << "overlaps ownership of pipe field '"
                     << state.fieldNames[fieldIndex] << "' between "
                     << describePipeRole(*state.fieldOwners[fieldIndex])
                     << " and " << describePipeRole(role);
          }
          writer.ownedFields = ownedFields;
          writer.hasOwnership = true;
          for (int fieldIndex : ownedFields.set_bits())
            state.fieldOwners[fieldIndex] = role;
        }

        writer.commits.push_back(commit);
        // The acquire's stage dominates every copy. An equivalent constant
        // used by commit may have been defined only after the copies.
        commitCopies[commit.getOperation()] = CommitPayload{
            std::move(payload), static_cast<int32_t>(totalBytes),
            matchingAcquire.getStage()};
        continue;
      }
      if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        PipeRole role = getPipeRole(op);
        if (failed(recordExecutionDomain(state, op, role)) ||
            failed(verifyReaderEndpoint(state, op, role)))
          return failure();
        FailureOr<int32_t> warps = getConsumerWarps(op);
        if (failed(warps))
          return failure();
        if (state.consumerWarps >= 0 && state.consumerWarps != *warps)
          return wait.emitOpError(
              "all reader operations must use one consumer warp count");
        state.consumerWarps = *warps;
        state.waits.push_back(wait);
        continue;
      }

      auto release = cast<tle::PipeReaderReleaseOp>(op);
      PipeRole role = getPipeRole(op);
      if (failed(recordExecutionDomain(state, op, role)) ||
          failed(verifyReaderEndpoint(state, op, role)))
        return failure();
      tle::PipeReaderWaitOp matchingWait;
      for (Operation *previous = release->getPrevNode(); previous;
           previous = previous->getPrevNode()) {
        if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(previous)) {
          if (getPipeKey(previous) == key &&
              sameIndex(wait.getStage(), release.getStage()) &&
              getPipeRole(previous) == role) {
            matchingWait = wait;
            break;
          }
        }
      }
      if (!matchingWait)
        return release.emitOpError(
            "requires a same-block, same-stage matching reader.wait");
      if (!consumedWaits.insert(matchingWait.getOperation()).second)
        return release.emitOpError(
            "reader.wait must be paired with exactly one reader.release");
      FailureOr<int32_t> warps = getConsumerWarps(op);
      if (failed(warps))
        return failure();
      if (state.consumerWarps >= 0 && state.consumerWarps != *warps)
        return release.emitOpError(
            "all reader operations must use one consumer warp count");
      state.consumerWarps = *warps;
      state.releases.push_back(release);
    }

    for (auto &[key, state] : pipes) {
      if (state.writers.empty())
        return state.create.emitOpError("requires at least one writer role");
      // Single-stage multi-writer reuse is outside the validated contract.
      // Keep it explicitly unsupported; single-writer capacity=1 is supported.
      if (state.writers.size() > 1 && state.capacity < 2)
        return state.create.emitOpError(
            "mthreads multi-writer TME pipes require capacity >= 2; "
            "single-stage multi-writer reuse is not supported");
      for (const auto &[role, writer] : state.writers) {
        if (writer.acquires.empty() || writer.commits.empty() ||
            !writer.hasOwnership)
          return state.create.emitOpError()
                 << describePipeRole(role)
                 << " requires at least one writer acquire/commit pair with a "
                    "non-empty, stable field subset";
        for (tle::PipeWriterAcquireOp acquire : writer.acquires) {
          if (!consumedAcquires.contains(acquire.getOperation()))
            return acquire.emitOpError(
                "writer.acquire has no matching writer.commit");
        }
      }
      for (auto [fieldIndex, owner] : llvm::enumerate(state.fieldOwners)) {
        if (!owner)
          return state.create.emitOpError()
                 << "writer ownership does not cover pipe field '"
                 << state.fieldNames[fieldIndex] << "'";
      }
      if (state.waits.empty() || state.releases.empty())
        return state.create.emitOpError(
            "requires at least one reader wait/release pair");
      for (tle::PipeReaderWaitOp wait : state.waits) {
        if (!consumedWaits.contains(wait.getOperation()))
          return wait.emitOpError("reader.wait has no matching reader.release");
      }
      if (state.consumerWarps <= 0)
        return state.create.emitOpError("could not infer consumer warp count");
      FailureOr<int32_t> readerIssueThread = resolveReaderIssueThread(
          state.releases.front(), *state.readerRole, state.consumerWarps);
      if (failed(readerIssueThread))
        return failure();
      state.readerIssueThread = *readerIssueThread;
      for (unsigned fieldIndex = 0; fieldIndex < state.fieldsWithCopies.size();
           ++fieldIndex) {
        if (!state.fieldsWithCopies.test(fieldIndex))
          return state.create.emitOpError()
                 << "did not find a TME copy for pipe field '"
                 << state.fieldNames[fieldIndex] << "'";
      }
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
                        CommitCopies &commitCopies) {
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
        auto pending = builder.getI32IntegerAttr(0);
        auto ready = builder.getI32IntegerAttr(1);
        // Give each writer its own transaction counter. The reader waits for
        // every writer's generation before consuming or releasing the slot.
        // Fields owned by one writer still share one byte registration and
        // one arrival, but independent issue warps never update the same
        // transaction counter.
        for (const auto &[role, writer] : state.writers)
          state.fullBases[role] = musa_tle::BarrierAllocOp::create(
              builder, loc, capacity, builder.getI32IntegerAttr(1), pending,
              IntegerAttr());
        // The collective reader rendezvous is separate; only its elected
        // first warp contributes to each empty generation.
        state.emptyBase = musa_tle::BarrierAllocOp::create(
            builder, loc, capacity, builder.getI32IntegerAttr(1), ready,
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
        auto payloadIt = commitCopies.find(op);
        if (payloadIt == commitCopies.end() || payloadIt->second.fields.empty())
          return commit.emitOpError("lost the analyzed pipe TME copies");
        const CommitPayload &payload = payloadIt->second;
        for (auto [copyIndex, fieldCopy] : llvm::enumerate(payload.fields)) {
          ttg::TMACopyOp copy = fieldCopy.copy;
          OpBuilder copyBuilder(copy);
          Value barrier = createIndex(copyBuilder, copy.getLoc(), copy,
                                      state.fullBases.at(getPipeRole(op)),
                                      payload.stage);
          auto replacement =
              ttg::TMACopyOp::create(copyBuilder, copy.getLoc(), copy.getSrc(),
                                     copy.getDst(), copy.getIndices(), barrier);
          replacement->setDiscardableAttrs(
              copy->getDiscardableAttrDictionary());
          if (copyIndex == 0) {
            replacement->setAttr(
                "expect_bytes",
                copyBuilder.getI32IntegerAttr(payload.transactionBytes));
          } else {
            replacement->setAttr(ttmg::kTLESkipTransactionAccountingAttr,
                                 copyBuilder.getUnitAttr());
          }
          copy.erase();
        }
        commit.erase();
        continue;
      }

      if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        // Enter a collective wait with the complete reader role, including
        // after scalar/predicated work in the preceding iteration.
        ttg::BarrierOp::create(builder, loc, ttg::AddrSpace::Local);
        Value phase = toI32Phase(builder, loc, wait.getPhase(), false);
        for (const auto &[role, fullBase] : state.fullBases) {
          Value barrier =
              createIndex(builder, loc, op, fullBase, wait.getStage());
          musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        }
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
      // All reader warps must finish their shared-memory accesses before the
      // elected publication warp can release this slot. Static WS lowering
      // makes this role-local, never a cross-role CTA wait. Async dot users
      // must still complete their dot reads before calling reader.release.
      ttg::BarrierOp::create(builder, loc, ttg::AddrSpace::Local);
      Value barrier =
          createIndex(builder, loc, op, state.emptyBase, release.getStage());
      Value phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
      // All readers have rendezvoused above. Their first physical warp alone
      // publishes the empty generation on behalf of the complete reader role.
      auto arrive = musa_tle::BarrierArriveOp::create(
          builder, loc, barrier, phase, builder.getI32IntegerAttr(1));
      arrive->setAttr(ttmg::kTLEPipeEmptyArrivalAttr, builder.getUnitAttr());
      arrive->setAttr(ttmg::kTMEIssueThreadAttr,
                      builder.getI32IntegerAttr(state.readerIssueThread));
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
    CommitCopies commitCopies;
    if (failed(analyze(module, pipes, commitCopies)) ||
        failed(rewrite(module, pipes, commitCopies)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
