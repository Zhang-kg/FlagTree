"""Compile and runtime coverage for the mthreads TLE pipe contract."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _pipe_consumer(reader, out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tl.store(out + iteration, iteration + tl.where(wait.is_closed, 1000, 0))
        reader.release(iteration)


@triton.jit
def _pipe_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _multi_field_consumer(reader, out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tl.store(out + iteration, iteration + tl.where(wait.is_closed, 1000, 0))
        reader.release(iteration)


@triton.jit
def _multi_field_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        offset = iteration * BLOCK
        tle.gpu.copy(desc, slot.first, (BLOCK, ), (offset, ))
        tle.gpu.copy(desc, slot.second, (BLOCK, ), (offset, ))
        writer.commit(iteration)


@triton.jit
def _reverse_multi_field_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        offset = iteration * BLOCK
        # Field declaration order must not choose the accounting copy: this
        # first-issued copy must register both fields before either TME runs.
        tle.gpu.copy(desc, slot.second, (BLOCK, ), (offset, ))
        tle.gpu.copy(desc, slot.first, (BLOCK, ), (offset, ))
        writer.commit(iteration)


@triton.jit
def _first_field_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.first, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _second_field_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.second, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _a_sfa_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        offset = iteration * BLOCK
        tle.gpu.copy(desc, slot.a, (BLOCK, ), (offset, ))
        tle.gpu.copy(desc, slot.sfa, (BLOCK, ), (offset, ))
        writer.commit(iteration)


@triton.jit
def _b_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.b, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _megamoe_math_reader(reader, out, payload, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    sfa_row0 = tl.full((BLOCK, ), 0, tl.int32)
    sfa_row1 = tl.full((BLOCK, ), 1, tl.int32)
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        # Check the actual shared payload, not just completion markers.  In a
        # narrow-copy iteration SFA row one is intentionally not initialized.
        a = tl.load(tle.gpu.local_ptr(wait.slot.a, (offsets, )))
        b = tl.load(tle.gpu.local_ptr(wait.slot.b, (offsets, )))
        sfa0 = tl.load(tle.gpu.local_ptr(wait.slot.sfa, (sfa_row0, offsets)))
        if iteration % 2 == 1:
            sfa1 = tl.load(tle.gpu.local_ptr(wait.slot.sfa, (sfa_row1, offsets)))
        else:
            sfa1 = tl.full((BLOCK, ), -777, tl.float16)
        base = payload + iteration * 4 * BLOCK + offsets
        tl.store(base, a)
        tl.store(base + BLOCK, b)
        tl.store(base + 2 * BLOCK, sfa0)
        tl.store(base + 3 * BLOCK, sfa1)
        # The producer must not reuse this stage before all payload loads.
        reader.release(iteration)
        marker = iteration + tl.where(wait.is_closed, 1000, 0)
        tl.store(out + iteration, marker)


@triton.jit
def _megamoe_a_sfa_producer(writer, a_desc, narrow_sfa_desc, wide_sfa_desc, BLOCK: tl.constexpr,
                           ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(a_desc, slot.a, (BLOCK, ), (iteration * BLOCK, ))
        if iteration % 2 == 0:
            tle.gpu.copy(narrow_sfa_desc, slot.sfa.subslice(0, 1, 0), (1, BLOCK), (2 * iteration, 0))
        else:
            tle.gpu.copy(wide_sfa_desc, slot.sfa, (2, BLOCK), (2 * iteration, 0))
        writer.commit(iteration)


@triton.jit
def _megamoe_dispatch_no_pipe(dispatch_marker):
    tl.store(dispatch_marker, 1701)


@triton.jit
def _variable_bytes_producer(writer, narrow_desc, wide_desc, BLOCK: tl.constexpr):
    narrow_slot = writer.acquire(0)
    tle.gpu.copy(narrow_desc, narrow_slot.data.subslice(0, 1, 0), (1, BLOCK), (0, 0))
    writer.commit(0)
    wide_slot = writer.acquire(1)
    tle.gpu.copy(wide_desc, wide_slot.data, (2, BLOCK), (0, 0))
    writer.commit(1)


@triton.jit
def _dual_pipe_consumer(first, second, out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        first_wait = first.wait(iteration)
        second_wait = second.wait(iteration)
        tl.store(out + iteration,
                 iteration + tl.where(first_wait.is_closed, 1000, 0) + tl.where(second_wait.is_closed, 2000, 0))
        first.release(iteration)
        second.release(iteration)


@triton.jit
def _dual_pipe_producer(first, second, first_desc, second_desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        first_slot = first.acquire(iteration)
        second_slot = second.acquire(iteration)
        tle.gpu.copy(first_desc, first_slot.data, (BLOCK, ), (iteration * BLOCK, ))
        tle.gpu.copy(second_desc, second_slot.data, (BLOCK, ), (iteration * BLOCK, ))
        first.commit(iteration)
        second.commit(iteration)


@triton.jit
def _dual_pipe_kernel(
    first_desc,
    second_desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    first_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    first = tle.pipe(capacity=STAGES, name="first", data=first_smem)
    second = tle.pipe(capacity=STAGES, name="second", data=second_smem)
    tle.gpu.warp_specialize(
        [
            (_dual_pipe_consumer, (first.reader(), second.reader(), out, ITERATIONS)),
            (
                _dual_pipe_producer,
                (first.writer(), second.writer(), first_desc, second_desc, BLOCK, ITERATIONS),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _pipe_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="data_pipe", data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
            (_pipe_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _non_ws_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(capacity=STAGES, name="runtime_ab", a=a_smem, b=b_smem)
    writer = pipe.writer()
    reader = pipe.reader()

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.static_range(0, K_TILES):
        slot = writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        writer.commit(k_iter)

        wait = reader.wait(k_iter)
        acc = tle.gpu.wgmma(wait.slot.a, wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_pipe_mm_consumer(
    reader,
    out,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        wait = reader.wait(k_iter)
        acc = tle.gpu.wgmma(wait.slot.a, wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_pipe_mm_producer(
    writer,
    a_desc,
    b_desc,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        slot = writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        writer.commit(k_iter)


@triton.jit
def _ws_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(capacity=STAGES, name="ws_runtime_ab", a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [
            (
                _ws_pipe_mm_consumer,
                (pipe.reader(), out, K_TILES, BLOCK_M, BLOCK_N),
            ),
            (
                _ws_pipe_mm_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _baseline_consumer(out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        tl.store(out + iteration, iteration)


@triton.jit
def _baseline_producer(smem, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr, STAGES: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        tle.gpu.copy(desc, smem.slot(iteration % STAGES), (BLOCK, ), (iteration * BLOCK, ))


@triton.jit
def _baseline_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.warp_specialize(
        [
            (_baseline_consumer, (out, ITERATIONS)),
            (_baseline_producer, (smem, desc, BLOCK, ITERATIONS, STAGES)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _local_store_producer(writer, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tl.store(tle.gpu.local_ptr(slot.data, (0, )), 0.0)
        writer.commit(iteration)


@triton.jit
def _double_tme_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _close_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)
        writer.close(iteration)


@triton.jit
def _invalid_pipe_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="invalid_pipe", data=smem)
    if KIND == 0:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_local_store_producer, (pipe.writer(), BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    elif KIND == 1:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_double_tme_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_close_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )


@triton.jit
def _invalid_pairing_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr, KIND: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)
    if KIND == 0:
        writer.commit(ITERATIONS - 1)
    elif KIND == 1:
        writer.acquire(ITERATIONS)


@triton.jit
def _invalid_pairing_consumer(reader, out, ITERATIONS: tl.constexpr, KIND: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tl.store(out + iteration, iteration + tl.where(wait.is_closed, 1000, 0))
        reader.release(iteration)
    if KIND == 2:
        reader.release(ITERATIONS - 1)
    elif KIND == 3:
        reader.wait(ITERATIONS)


@triton.jit
def _invalid_pairing_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr,
                            KIND: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="invalid_pairing", data=smem)
    tle.gpu.warp_specialize(
        [
            (_invalid_pairing_consumer, (pipe.reader(), out, ITERATIONS, KIND)),
            (_invalid_pairing_producer, (pipe.writer(), desc, BLOCK, ITERATIONS, KIND)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _misplaced_pipe_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="misplaced", data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _multi_field_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    first = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="multi_field", first=first, second=second)
    tle.gpu.warp_specialize(
        [
            (_multi_field_consumer, (pipe.reader(), out, ITERATIONS)),
            (_multi_field_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _reverse_multi_field_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    first = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="reverse_multi_field", first=first, second=second)
    tle.gpu.warp_specialize(
        [
            (_multi_field_consumer, (pipe.reader(), out, ITERATIONS)),
            (_reverse_multi_field_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _multi_writer_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    a = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    sfa = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="multi_writer", a=a, b=b, sfa=sfa)
    # Mirror MegaMoE ownership: the default role owns {a, sfa}, worker 0 owns
    # {b}, and worker 1 is the sole all-fields reader.
    tle.gpu.warp_specialize(
        [
            (_a_sfa_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_b_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_multi_field_consumer, (pipe.reader(), out, ITERATIONS)),
        ],
        worker_num_warps=[4, 4],
        worker_num_regs=[24, 24],
    )


@triton.jit
def _megamoe_exact_pipe_kernel(
    a_desc,
    b_desc,
    narrow_sfa_desc,
    wide_sfa_desc,
    out,
    payload,
    dispatch_marker,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    pid = tl.program_id(0)
    out = out + pid * ITERATIONS
    payload = payload + pid * ITERATIONS * 4 * BLOCK
    dispatch_marker = dispatch_marker + pid
    a = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    sfa = tle.gpu.alloc((STAGES, 2, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="megamoe_exact",
        readers=("math", ),
        a=a,
        b=b,
        sfa=sfa,
    )
    tle.gpu.warp_specialize(
        [
            (_megamoe_math_reader, (pipe.reader(name="math"), out, payload, BLOCK, ITERATIONS)),
            (_megamoe_dispatch_no_pipe, (dispatch_marker, )),
            (_megamoe_a_sfa_producer,
             (pipe.writer(), a_desc, narrow_sfa_desc, wide_sfa_desc, BLOCK, ITERATIONS)),
            (_b_producer, (pipe.writer(), b_desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[2, 1, 1],
        worker_num_regs=[24, 24, 24],
    )


@triton.jit
def _incomplete_ownership_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    first = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="incomplete", first=first, second=second)
    tle.gpu.warp_specialize(
        [
            (_multi_field_consumer, (pipe.reader(), out, ITERATIONS)),
            (_first_field_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _overlapping_ownership_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    first = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="overlap", first=first, second=second)
    tle.gpu.warp_specialize(
        [
            (_first_field_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_first_field_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_multi_field_consumer, (pipe.reader(), out, ITERATIONS)),
        ],
        worker_num_warps=[4, 4],
        worker_num_regs=[24, 24],
    )


@triton.jit
def _unstable_field_producer(writer, desc, BLOCK: tl.constexpr):
    first_slot = writer.acquire(0)
    tle.gpu.copy(desc, first_slot.first, (BLOCK, ), (0, ))
    writer.commit(0)
    second_slot = writer.acquire(1)
    tle.gpu.copy(desc, second_slot.second, (BLOCK, ), (BLOCK, ))
    writer.commit(1)


@triton.jit
def _two_iteration_consumer(reader, out):
    for iteration in tl.static_range(0, 2):
        wait = reader.wait(iteration)
        tl.store(out + iteration, tl.where(wait.is_closed, -1, iteration))
        reader.release(iteration)


@triton.jit
def _unstable_ownership_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    first = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="unstable", first=first, second=second)
    tle.gpu.warp_specialize(
        [
            (_two_iteration_consumer, (pipe.reader(), out)),
            (_unstable_field_producer, (pipe.writer(), desc, BLOCK)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _variable_bytes_kernel(
    narrow_desc,
    wide_desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    data = tle.gpu.alloc((STAGES, 2, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="variable_bytes", data=data)
    tle.gpu.warp_specialize(
        [
            (_two_iteration_consumer, (pipe.reader(), out)),
            (_variable_bytes_producer, (pipe.writer(), narrow_desc, wide_desc, BLOCK)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _named_reader_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, readers=("consumer", ), data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_consumer, (pipe.reader(name="consumer"), out, ITERATIONS)),
            (_pipe_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _multiple_named_readers_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.pipe(capacity=STAGES, readers=("left", "right"), data=smem)


@triton.jit
def _one_shot_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.pipe(capacity=STAGES, one_shot=True, data=smem)


def _compile_pipeline(fn, stages):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=fn,
        signature={
            "desc": "tensordesc<fp16[128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
        constexprs={"STAGES": stages, "BLOCK": 128, "ITERATIONS": 2 * stages},
    )
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = compiler_stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_pipe_shared_memory")
    return ttir, ttgir, module.str_nodebug()


def _compile_invalid_pipeline(fn, kind=None):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    signature = {
        "desc": "tensordesc<fp16[128]>",
        "out": "*i32",
        "STAGES": "constexpr",
        "BLOCK": "constexpr",
        "ITERATIONS": "constexpr",
    }
    constexprs = {"STAGES": 2, "BLOCK": 128, "ITERATIONS": 2}
    if kind is not None:
        signature["KIND"] = "constexpr"
        constexprs["KIND"] = kind
    source = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    return compiler_stages["ttgir"](module, metadata)


def _compile_variable_bytes_pipeline():
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_variable_bytes_kernel,
        signature={
            "narrow_desc": "tensordesc<fp16[1, 128]>",
            "wide_desc": "tensordesc<fp16[2, 128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
        constexprs={"STAGES": 2, "BLOCK": 128, "ITERATIONS": 2},
    )
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    module = compiler_stages["ttgir"](module, metadata)
    return module.str_nodebug()


def _compile_dual_pipeline(stages):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_dual_pipe_kernel,
        signature={
            "first_desc": "tensordesc<fp16[128]>",
            "second_desc": "tensordesc<fp16[128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
        constexprs={"STAGES": stages, "BLOCK": 128, "ITERATIONS": 2 * stages},
    )
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    module = compiler_stages["ttgir"](module, metadata)
    return module.str_nodebug()


def _compile_megamoe_exact_pipeline(capacity):
    iterations = 2 * capacity + 1
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_megamoe_exact_pipe_kernel,
        signature={
            "a_desc": "tensordesc<fp16[128]>",
            "b_desc": "tensordesc<fp16[128]>",
            "narrow_sfa_desc": "tensordesc<fp16[1, 128]>",
            "wide_sfa_desc": "tensordesc<fp16[2, 128]>",
            "out": "*i32",
            "payload": "*fp16",
            "dispatch_marker": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
        constexprs={
            "STAGES": capacity,
            "BLOCK": 128,
            "ITERATIONS": iterations,
        },
    )
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = compiler_stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()
    llir = compiler_stages["llir"](module, metadata)
    return ttir, ttgir, llir, metadata, iterations


def _shared_bytes(allocated):
    return int(re.search(r"ttg\.shared = (\d+) : i32", allocated).group(1))


def _i32_constants(ir_text):
    return {
        name: int(value)
        for name, value in re.findall(r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ir_text)
    }


def _tme_copy_issue_threads(ir_text):
    issue_threads = []
    for line in ir_text.splitlines():
        if "ttmg.async_tme_copy_global_to_local" not in line:
            continue
        match = re.search(
            r'"?musa\.tme\.issue_thread"?\s*=\s*(\d+)\s*:\s*i32', line
        )
        assert match, line
        issue_threads.append(int(match.group(1)))
    return issue_threads


def _assert_writer_transaction_groups(ttgir, role, terminator, transaction_bytes, copies_per_commit, issue_thread):
    """Each commit accounts before its first copy and shares one completion."""
    body = ttgir.split(role, maxsplit=1)[1].split(terminator, maxsplit=1)[0]
    pattern = r"ttmg\.(barrier_add_trans|arrive_barrier_noret|async_tme_copy_global_to_local)\b"
    operations = [(re.search(pattern, line).group(1), line) for line in body.splitlines()
                  if re.search(pattern, line)]
    expected_kinds = ["barrier_add_trans", "arrive_barrier_noret"] + [
        "async_tme_copy_global_to_local"
    ] * copies_per_commit
    assert [kind for kind, _ in operations] == expected_kinds * len(transaction_bytes), body
    constants = _i32_constants(ttgir)
    group_size = len(expected_kinds)
    for commit, expected_bytes in enumerate(transaction_bytes):
        group = operations[commit * group_size:(commit + 1) * group_size]
        add_barrier, byte_value = re.search(
            r"ttmg\.barrier_add_trans\s+(%[-\w.]+),\s*(%[-\w.]+)", group[0][1]
        ).groups()
        arrive_barrier = re.search(r"ttmg\.arrive_barrier_noret\s+(%[-\w.]+)", group[1][1]).group(1)
        assert constants[byte_value] == expected_bytes, group
        assert add_barrier == arrive_barrier, group
        for _, copy in group[2:]:
            copy_barrier = re.search(
                r"ttmg\.async_tme_copy_global_to_local\s+[^\n]*\],\s*(%[-\w.]+)", copy
            ).group(1)
            assert copy_barrier == add_barrier, group
        for _, line in group:
            assert re.search(rf'"?musa\.tme\.issue_thread"?\s*=\s*{issue_thread}\s*:\s*i32', line), line
            assert "musa.tme.explicit_completion" in line, line
    assert "musa_tle.expect_bytes" not in body, body
    assert "musa_tle.skip_transaction_accounting" not in body, body


def _assert_private_full_and_leader_empty(ttgir, stages, iterations, writer_roles, reader_role, reader_issue_thread):
    def role_body(role):
        terminator = "ttg.warp_yield" if role == "default {" else "ttg.warp_return"
        return ttgir.split(role, 1)[1].split(terminator, 1)[0]

    def constants_in(body):
        constants = _i32_constants(ttgir)
        constants.update(_i32_constants(body))
        for result, lhs, rhs in re.findall(
            r"(%[-\w.]+)\s*=\s*arith\.addi\s+(%[-\w.]+),\s*(%[-\w.]+)\s*:\s*i32", body
        ):
            if lhs in constants and rhs in constants:
                constants[result] = constants[lhs] + constants[rhs]
        return constants

    full_ids, writer_empty_waits = [], []
    for role in writer_roles:
        body = role_body(role)
        constants = constants_in(body)
        full_ids.append([constants[bar] for bar in re.findall(r"ttmg\.barrier_add_trans\s+(%[-\w.]+),", body)])
        writer_empty_waits.append([
            (constants[bar], constants[phase])
            for bar, phase in re.findall(r"ttmg\.wait_barrier\s+(%[-\w.]+),\s*(%[-\w.]+)", body)
        ])
        assert len(full_ids[-1]) == iterations and len(set(full_ids[-1])) == stages, body
    for index, ids in enumerate(full_ids):
        assert all(set(ids).isdisjoint(other) for other in full_ids[index + 1:]), full_ids

    body = role_body(reader_role)
    constants = constants_in(body)
    waits, empty_ids = [], []
    for line in body.splitlines():
        wait = re.search(r"ttmg\.wait_barrier\s+(%[-\w.]+),\s*(%[-\w.]+)", line)
        if wait:
            waits.append((constants[wait.group(1)], constants[wait.group(2)]))
        release = re.search(r"ttmg\.warp_arrive_barrier\s+(%[-\w.]+),", line)
        if release:
            iteration = len(empty_ids)
            assert waits == [(ids[iteration], (iteration // stages) % 2) for ids in full_ids], (waits, full_ids)
            assert "musa_tle.pipe_empty_arrival" in line, line
            assert f"musa.tme.issue_thread = {reader_issue_thread} : i32" in line, line
            empty_ids.append(constants[release.group(1)])
            waits = []
    assert len(empty_ids) == iterations and not waits, body
    assert len(set(empty_ids)) == stages, empty_ids
    assert all(set(empty_ids).isdisjoint(ids) for ids in full_ids), (empty_ids, full_ids)
    assert all(waits == [(empty_ids[i], 1 - (i // stages) % 2) for i in range(iterations)]
               for waits in writer_empty_waits), writer_empty_waits
    # Leader election must not replace the all-reader rendezvous.
    assert body.count("ttg.barrier local") == 2 * iterations, body
    assert ttgir.count("musa_tle.pipe_empty_arrival") == iterations, ttgir


def _assert_exact_tme_physical_role_ranges(llir, iterations):
    # LLVM can prove a one-warp issuer from WS dispatch and delete an explicit
    # (tid & -32) == role_start check. Follow the actual CFG instead: every
    # reachable TME site must be confined to its physical writer warp.
    function = re.search(r"^define\b[^\n]*\{\n(.*?)^\}", llir, re.MULTILINE | re.DOTALL)
    assert function, llir
    blocks = {"entry": []}
    block = "entry"
    for line in function.group(1).splitlines():
        label = re.match(r"^([-\w.$]+):", line)
        if label:
            block = label.group(1)
            blocks[block] = []
        else:
            blocks[block].append(line)
    if not any(line.strip() for line in blocks["entry"]):
        del blocks["entry"]
    entry = next(iter(blocks))
    definitions = dict(re.findall(r"^\s*(%[-\w.]+)\s*=\s*(.+)$", function.group(1), re.MULTILINE))
    raw_tids = {name for name, rhs in definitions.items() if "@llvm.musa.read.ptx.sreg.tid.x()" in rhs}
    assert raw_tids, llir

    def value_for_tid(value, tid):
        if value in raw_tids:
            return tid
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        rhs = definitions.get(value, "")
        compare = re.match(r"icmp (eq|ne|ult|ule|ugt|uge) i32 (%[-\w.]+|-?\d+), (%[-\w.]+|-?\d+)", rhs)
        binary = re.match(r"(and|or|xor) i(?:1|32) (%[-\w.]+|-?\d+), (%[-\w.]+|-?\d+)", rhs)
        operation = compare or binary
        if not operation:
            return None
        op, lhs, rhs = operation.groups()
        lhs, rhs = value_for_tid(lhs, tid), value_for_tid(rhs, tid)
        if lhs is None or rhs is None:
            return None
        if compare:
            lhs, rhs = lhs & 0xFFFFFFFF, rhs & 0xFFFFFFFF
            return {"eq": lhs == rhs, "ne": lhs != rhs, "ult": lhs < rhs,
                    "ule": lhs <= rhs, "ugt": lhs > rhs, "uge": lhs >= rhs}[op]
        return {"and": lhs & rhs, "or": lhs | rhs, "xor": lhs ^ rhs}[op]

    reachable_tids = {name: set() for name in blocks}
    reachable_tids[entry] = set(range(256))
    pending = [entry]
    while pending:
        block = pending.pop()
        body = "\n".join(blocks[block])
        branch = re.search(r"\bbr i1 (%[-\w.]+), label %([-\w.$]+), label %([-\w.$]+)", body)
        if branch:
            condition, on_true, on_false = branch.groups()
            choices = [(on_true, True), (on_false, False)]
        else:
            jump = re.search(r"\bbr label %([-\w.$]+)", body)
            choices = [(jump.group(1), None)] if jump else []
            assert jump or re.search(r"\b(ret|unreachable)\b", body), body
        for successor, expected in choices:
            assert successor in blocks, body
            # A non-tid predicate is conservatively traversed both ways.
            incoming = {tid for tid in reachable_tids[block]
                        if expected is None or value_for_tid(condition, tid) in (None, expected)}
            if not incoming.issubset(reachable_tids[successor]):
                reachable_tids[successor].update(incoming)
                pending.append(successor)

    copy_counts = {192: {1: 0, 2: 0}, 224: {1: 0, 2: 0}}
    transactions = {192: [], 224: []}
    for block, lines in blocks.items():
        for line in lines:
            copy = re.search(r"\bcall void @llvm\.musa\.tme\.ld\.tile\.([12])d\(", line)
            transaction = re.search(r"\bcall void @llvm\.musa\.async\.add\.trans\(i32 [^,]+, i32 (\d+)\)", line)
            if not (copy or transaction):
                continue
            tids = reachable_tids[block]
            role = next((start for start in copy_counts if tids and tids.issubset(set(range(start, start + 32)))), None)
            assert role is not None, (block, sorted(tids), line)
            if copy:
                copy_counts[role][int(copy.group(1))] += 1
            else:
                transactions[role].append(int(transaction.group(1)))
    assert copy_counts == {192: {1: iterations, 2: iterations}, 224: {1: iterations, 2: 0}}, copy_counts
    assert sorted(transactions[192]) == sorted(512 if i % 2 == 0 else 768 for i in range(iterations)), transactions
    assert transactions[224] == [256] * iterations, transactions


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_single_field_pipe_lowers_to_hardware_barriers(stages):
    ttir, ttgir, allocated = _compile_pipeline(_pipe_kernel, stages)

    assert ttir.count("tle.pipe.create") == 1, ttir
    assert ttir.count("tle.pipe.writer_acquire") == 2 * stages, ttir
    assert ttir.count("tle.pipe.writer_commit") == 2 * stages, ttir
    assert ttir.count("tle.pipe.reader_wait") == 2 * stages, ttir
    assert ttir.count("tle.pipe.reader_release") == 2 * stages, ttir
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 2 * stages, ttgir
    assert f"musa.max_bar_id = {2 * stages}" in ttgir, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 4 * stages, ttgir
    assert "ttmg.barrier_add_trans" in ttgir, ttgir
    assert "ttmg.wait_barrier" in ttgir, ttgir
    assert "ttmg.warp_arrive_barrier" in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir, ttgir
    assert "tle.pipe." not in allocated, allocated
    assert "1000" not in ttgir, ttgir

    constants = _i32_constants(ttgir)
    arrivals = re.findall(r"ttmg\.init_arrival\s+(%[-\w.]+),\s+(%[-\w.]+),\s+(%[-\w.]+)", ttgir)
    assert [constants[count] for _, count, _ in arrivals] == [1] * (2 * stages)
    assert [constants[phase] for _, _, phase in arrivals] == [0] * (2 * stages)
    assert ttgir.count("ttmg.barrier_add_trans") == 2 * stages
    assert all("%c256" in line for line in ttgir.splitlines() if "ttmg.barrier_add_trans" in line)
    _assert_private_full_and_leader_empty(ttgir, stages, 2 * stages, ["partition0("], "default {", 0)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_pipe_barrier_ids_add_no_shared_memory(stages):
    _, _, pipe_allocated = _compile_pipeline(_pipe_kernel, stages)
    _, _, baseline_allocated = _compile_pipeline(_baseline_kernel, stages)

    assert _shared_bytes(pipe_allocated) == _shared_bytes(baseline_allocated)
    assert pipe_allocated.count("ttg.warp_specialize") == 1, pipe_allocated
    ws_line = next(line for line in pipe_allocated.splitlines() if "ttg.warp_specialize" in line)
    assert "allocation.offset" not in ws_line, ws_line
    assert "musa_tle.static_ws." not in pipe_allocated, pipe_allocated


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_two_single_field_pipes_keep_independent_barrier_rings(stages):
    ttgir = _compile_dual_pipeline(stages)
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 4 * stages, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 4 * stages, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 4 * stages, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 8 * stages, ttgir
    assert f"musa.max_bar_id = {4 * stages}" in ttgir, ttgir


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_multi_field_pipe_uses_grouped_full_and_shared_empty_rings(stages):
    ttir, ttgir, _ = _compile_pipeline(_multi_field_kernel, stages)

    assert ttir.count("tle.pipe.create") == 1, ttir
    assert ttir.count("tle.pipe.writer_acquire") == 2 * stages, ttir
    assert ttir.count("tle.pipe.writer_commit") == 2 * stages, ttir
    assert ttir.count("tle.pipe.reader_wait") == 2 * stages, ttir
    assert ttir.count("tle.pipe.reader_release") == 2 * stages, ttir
    assert "field_names = [\"first\", \"second\"]" in ttir, ttir
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 2 * stages, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 2 * stages, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 2 * stages, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 4 * stages, ttgir
    # One shared-empty wait per acquire, one grouped-full wait per reader wait.
    assert ttgir.count("ttmg.wait_barrier") == 4 * stages, ttgir
    assert f"musa.max_bar_id = {2 * stages}" in ttgir, ttgir

    constants = _i32_constants(ttgir)
    arrivals = re.findall(r"ttmg\.init_arrival\s+(%[-\w.]+),\s+(%[-\w.]+),\s+(%[-\w.]+)", ttgir)
    assert [constants[count] for _, count, _ in arrivals] == [1] * (2 * stages), ttgir
    _assert_writer_transaction_groups(ttgir, "partition0(", "ttg.warp_return", [512] * (2 * stages), 2, 512)


def test_mthreads_multi_field_pipe_accounts_before_first_copy_not_first_declared_field():
    ttir, ttgir, _ = _compile_pipeline(_reverse_multi_field_kernel, 2)
    assert 'field_names = ["first", "second"]' in ttir, ttir
    # The producer executes second then first.  A declaration-order primary
    # would incorrectly place accounting between the two TME copies.
    _assert_writer_transaction_groups(ttgir, "partition0(", "ttg.warp_return", [512] * 4, 2, 512)


def test_mthreads_multi_field_pipe_rejects_aggregate_transaction_byte_overflow(tmp_path, capfd):
    # Start from valid frontend IR, then enlarge only the synthetic IR shapes.
    # Each fp16 copy still fits positive i32 (2**30 bytes), but two copies in
    # one writer commit do not.  Run LowerPipe alone: no huge allocation or
    # device execution is attempted, and the diagnostic must be about the sum.
    ttir, _, _ = _compile_pipeline(_multi_field_kernel, 1)
    fixture = ttir.replace("128", "536870912")
    assert "536870912xf16" in fixture, fixture
    # Frontend TTIR predates convert-triton-to-tritongpu.  Supply the module
    # context that LowerPipe needs when it visits the default reader role.
    assert "module {" in fixture, fixture
    fixture = fixture.replace(
        "module {",
        'module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, '
        'ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {',
        1,
    )
    fixture_path = tmp_path / "aggregate_pipe_transaction_overflow.ttir"
    fixture_path.write_text(fixture)
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_pipe(pm)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "reject_aggregate_pipe_transaction_overflow")
    diagnostics = capfd.readouterr().err
    assert "pipe writer transaction bytes exceed the positive i32 range" in diagnostics, diagnostics


def test_mthreads_pipe_uses_dominating_acquire_stage_for_equivalent_late_commit(tmp_path):
    field_type = "!ttg.memdesc<1x128xf16, #shared, #smem, mutable>"
    slot_type = "!ttg.memdesc<128xf16, #slot, #smem, mutable>"
    desc_type = "!tt.tensordesc<tensor<128xf16, #slot>>"
    pipe_attrs = 'capacity = 1 : i32, field_names = ["data"], pipe_name = "late_stage", scope = "cta"'
    fixture = f"""#shared = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
#slot = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @late_commit_stage(%desc: {desc_type}) {{
    %acquire_stage = arith.constant 0 : i32
    %false = arith.constant false
    %field = ttg.local_alloc : () -> {field_type}
    tle.pipe.create %field {{{pipe_attrs}}} : {field_type}
    tle.pipe.writer_acquire %field[%acquire_stage, %false] {{{pipe_attrs}}} : {field_type}
    %dst = ttg.memdesc_index %field[%acquire_stage] : {field_type} -> {slot_type}
    ttg.tma_copy %desc, %dst, [%acquire_stage] : {desc_type}, {slot_type}
    %commit_stage = arith.constant 0 : i32
    tle.pipe.writer_commit %field[%commit_stage] {{{pipe_attrs}}} : {field_type}
    %closed = tle.pipe.reader_wait %field[%acquire_stage, %false] {{{pipe_attrs}}} : {field_type}
    tle.pipe.reader_release %field[%acquire_stage] {{{pipe_attrs}}} : {field_type}
    tt.return
  }}
}}
"""
    fixture_path = tmp_path / "late_equivalent_commit_stage.ttir"
    fixture_path.write_text(fixture)
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_pipe(pm)
    # No canonicalization/CSE: the equivalent constant must remain after the
    # copy. Pass verification catches use-before-definition if lowering uses
    # commit.getStage() instead of the matching acquire's earlier SSA value.
    pm.run(module, "lower_pipe_with_late_equivalent_commit_stage")
    lowered = module.str_nodebug()
    assert "tle.pipe." not in lowered, lowered
    copy = next(line for line in lowered.splitlines() if "ttg.tma_copy" in line)
    copy_barrier = re.search(r"\bbarrier\s+(%[-\w.]+)", copy).group(1)
    slot_stage = re.search(r"ttg\.memdesc_index\s+%[-\w.]+\[(%[-\w.]+)\]", lowered).group(1)
    full_index = re.search(
        rf"{re.escape(copy_barrier)}\s*=\s*musa_tle\.barrier\.index\s+%[-\w.]+\[(%[-\w.]+)\]", lowered
    )
    assert full_index, lowered
    assert full_index.group(1) == slot_stage, lowered
    assert full_index.start() < lowered.index(copy), lowered
    late_zero = re.search(r"(%[-\w.]+)\s*=\s*arith\.constant\s+0\s*:\s*i32",
                          lowered[lowered.index(copy) + len(copy):])
    assert late_zero and late_zero.group(1) != slot_stage, lowered


@pytest.mark.parametrize("stages", [2, 3, 8])
def test_mthreads_multi_writer_pipe_groups_completion_and_partitions_field_ownership(stages):
    ttir, ttgir, _ = _compile_pipeline(_multi_writer_kernel, stages)

    assert ttir.count("tle.pipe.create") == 1, ttir
    assert ttir.count("tle.pipe.writer_acquire") == 4 * stages, ttir
    assert ttir.count("tle.pipe.writer_commit") == 4 * stages, ttir
    assert ttir.count("tle.pipe.reader_wait") == 2 * stages, ttir
    assert "field_names = [\"a\", \"b\", \"sfa\"]" in ttir, ttir
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 3 * stages, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 4 * stages, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 4 * stages, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 6 * stages, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 8 * stages, ttgir
    assert f"musa.max_bar_id = {3 * stages}" in ttgir, ttgir

    constants = _i32_constants(ttgir)
    arrivals = re.findall(r"ttmg\.init_arrival\s+(%[-\w.]+),\s+(%[-\w.]+),\s+(%[-\w.]+)", ttgir)
    assert [constants[count] for _, count, _ in arrivals] == [1] * (3 * stages), ttgir
    _assert_writer_transaction_groups(ttgir, "default {", "ttg.warp_yield", [512] * (2 * stages), 2, 0)
    _assert_writer_transaction_groups(ttgir, "partition0(", "ttg.warp_return", [256] * (2 * stages), 1, 512)
    _assert_private_full_and_leader_empty(ttgir, stages, 2 * stages, ["default {", "partition0("], "partition1(", 640)


@pytest.mark.parametrize("capacity", [2, 3, 8])
def test_mthreads_megamoe_exact_pipe_compile_contract(capacity):
    ttir, ttgir, llir, metadata, iterations = _compile_megamoe_exact_pipeline(capacity)

    assert iterations > 2 * capacity
    assert metadata["num_warps"] == 8
    assert 'readers = ["math"]' in ttir, ttir
    assert 'reader_name = "math"' in ttir, ttir
    assert 'field_names = ["a", "b", "sfa"]' in ttir, ttir
    assert ttir.count("tle.pipe.writer_acquire") == 2 * iterations, ttir
    assert ttir.count("tle.pipe.writer_commit") == 2 * iterations, ttir
    assert ttir.count("tle.pipe.reader_wait") == iterations, ttir
    assert ttir.count("tle.pipe.reader_release") == iterations, ttir

    assert '"ttg.total-num-warps" = 8 : i32' in ttgir, ttgir
    assert re.search(
        r"warpGroupStartIds\s*=\s*array<i32:\s*4,\s*6,\s*7>", ttgir
    ), ttgir
    assert re.findall(r"\bpartition(\d+)\(", ttgir) == ["0", "1", "2"], ttgir
    assert ttgir.count("ttg.warp_return") == 3, ttgir
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 3 * capacity, ttgir
    assert f"musa.max_bar_id = {3 * capacity}" in ttgir, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 4 * iterations, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 2 * iterations, ttgir

    constants = _i32_constants(ttgir)
    arrivals = re.findall(
        r"ttmg\.init_arrival\s+(%[-\w.]+),\s+(%[-\w.]+),\s+(%[-\w.]+)",
        ttgir,
    )
    arrival_counts = [constants[count] for _, count, _ in arrivals]
    assert arrival_counts == [1] * (3 * capacity), ttgir

    issue_threads = _tme_copy_issue_threads(ttgir)
    assert len(issue_threads) == 3 * iterations, ttgir
    assert issue_threads.count(192) == 2 * iterations, ttgir
    assert issue_threads.count(224) == iterations, ttgir
    assert set(issue_threads) == {192, 224}, ttgir

    # A+SFA registers a single 512-/768-byte transaction at each commit;
    # B independently registers 256 bytes.  Three copies need two arrivals.
    add_trans = [line for line in ttgir.splitlines() if "ttmg.barrier_add_trans" in line]
    assert len(add_trans) == 2 * iterations, ttgir
    trans_bytes = []
    for line in add_trans:
        operands = re.search(r"ttmg\.barrier_add_trans\s+(%[-\w.]+),\s+(%[-\w.]+)", line)
        assert operands, line
        trans_bytes.append(constants[operands.group(2)])
    assert trans_bytes.count(512) == (iterations + 1) // 2, ttgir
    assert trans_bytes.count(768) == iterations // 2, ttgir
    assert trans_bytes.count(256) == iterations, ttgir
    _assert_writer_transaction_groups(
        ttgir, "partition1(", "ttg.warp_return", [512 if iteration % 2 == 0 else 768
                                                 for iteration in range(iterations)], 2, 192
    )
    _assert_writer_transaction_groups(ttgir, "partition2(", "ttg.warp_return", [256] * iterations, 1, 224)
    _assert_private_full_and_leader_empty(ttgir, capacity, iterations, ["partition1(", "partition2("], "default {", 0)

    assert "ttg.warp_specialize" not in llir, llir
    assert '!"maxntidx", i32 256' in llir, llir
    # The reader rendezvous is allocated during late LLVM lowering, in
    # addition to two private-full rings and one shared-empty ring above.
    assert f"llvm.musa.async.bar.record(i32 {3 * capacity + 1})" in llir, llir
    assert f"llvm.musa.async.init.arrival(i32 {3 * capacity + 1}, i32 4, i32 0)" in llir, llir
    # Both reader.wait and reader.release rendezvous all four reader warps.
    reader_arrivals = re.findall(
        rf"call i32 @llvm\.musa\.async\.arrive\(i32 {3 * capacity + 1}\)", llir
    )
    assert len(reader_arrivals) == 2 * iterations, llir
    _assert_exact_tme_physical_role_ranges(llir, iterations)


def test_mthreads_multi_writer_pipe_rejects_single_stage(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_megamoe_exact_pipeline(1)
    assert "multi-writer TME pipes require capacity >= 2" in capfd.readouterr().err


def test_mthreads_pipe_tracks_transaction_bytes_per_copy():
    ttgir = _compile_variable_bytes_pipeline()
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 4, ttgir
    assert "musa.max_bar_id = 4" in ttgir, ttgir
    add_trans = [line for line in ttgir.splitlines() if "ttmg.barrier_add_trans" in line]
    assert len(add_trans) == 2, ttgir
    assert any("%c256" in line for line in add_trans), ttgir
    assert any("%c512" in line for line in add_trans), ttgir


def test_mthreads_pipe_supports_one_named_reader():
    ttir, ttgir, _ = _compile_pipeline(_named_reader_kernel, 2)
    assert 'readers = ["consumer"]' in ttir, ttir
    assert 'reader_name = "consumer"' in ttir, ttir
    assert "tle.pipe." not in ttgir, ttgir


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (_multiple_named_readers_kernel, "default reader or exactly one named reader"),
        (_one_shot_kernel, "does not support one_shot=True"),
    ],
)
def test_mthreads_pipe_rejects_unsupported_frontend_options(kernel, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        _compile_invalid_pipeline(kernel)


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "must produce at least one pipe field with a TME copy"),
        (1, "has multiple TME copies for pipe field 'data'"),
        (2, "does not support writer.close"),
    ],
)
def test_mthreads_pipe_rejects_unsupported_producer_protocol(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_pipe_kernel, kind)
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "writer.acquire must be paired with exactly one writer.commit"),
        (1, "writer.acquire has no matching writer.commit"),
        (2, "reader.wait must be paired with exactly one reader.release"),
        (3, "reader.wait has no matching reader.release"),
    ],
)
def test_mthreads_pipe_rejects_reused_or_unpaired_endpoints(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_pairing_kernel, kind)
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (_incomplete_ownership_kernel, "writer ownership does not cover pipe field 'second'"),
        (_overlapping_ownership_kernel, "overlaps ownership of pipe field 'first'"),
        (_unstable_ownership_kernel, "changes field ownership"),
    ],
)
def test_mthreads_pipe_rejects_invalid_multi_writer_ownership(capfd, kernel, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(kernel)
    assert diagnostic in capfd.readouterr().err


def test_mthreads_pipe_allows_writer_in_default_partition():
    ttir, ttgir, _ = _compile_pipeline(_misplaced_pipe_kernel, 2)
    assert "tle.pipe.writer_commit" in ttir, ttir
    assert "tle.pipe." not in ttgir, ttgir
    _assert_private_full_and_leader_empty(ttgir, 2, 4, ["default {"], "partition0(", 512)


@pytest.mark.parametrize(
    "block_m,block_n,k_tiles,stages",
    [
        pytest.param(128, 128, 4, 1, id="m128-n128-k256-stage1"),
        pytest.param(128, 128, 4, 2, id="m128-n128-k256-stage2"),
        pytest.param(128, 128, 7, 3, id="m128-n128-k448-stage3"),
        pytest.param(256, 256, 1, 1, id="m256-n256-k64-stage1"),
    ],
)
def test_mthreads_non_ws_pipe_mm_runtime(block_m, block_n, k_tiles, stages):
    torch.manual_seed(42)
    block_k = 64
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    for _ in range(2):
        out.fill_(float("nan"))
        _non_ws_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_mthreads_ws_pipe_mm_runtime(stages):
    torch.manual_seed(42)
    block_m = block_n = 256
    block_k = 64
    # Seven tiles exercise a full two-phase cycle and then reuse slot zero for
    # the third generation when capacity is three.
    k_tiles = 7
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    compiled = _ws_pipe_mm_kernel.warmup(
        a_desc,
        b_desc,
        out,
        K_TILES=k_tiles,
        STAGES=stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert compiled.metadata.shared == stages * 65536
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("ttg.warp_specialize") == 1
    assert "musa_tle.static_ws." not in compiled.asm["ttgir"]
    assert "ttg.warp_specialize" not in compiled.asm["llir"]
    assert "ttg.convert_layout" not in compiled.asm["ttgir"]
    assert "swizzleGranularity = 1 : i32" in compiled.asm["ttgir"]
    assert "swizzleGranularity = 2 : i32" in compiled.asm["ttgir"]
    assert "builtin.unrealized_conversion_cast" not in compiled.asm["llir"]
    # Common entry and exit only; role-internal syncs are role-local barriers.
    assert compiled.asm["llir"].count("call void @llvm.musa.syncthreads.lm()") == 2
    # Two pipe rings, then reader (16-warps) and producer (4-warps) rendezvous.
    assert f"llvm.musa.async.bar.record(i32 {2 * stages + 2})" in compiled.asm["llir"]
    assert f"llvm.musa.async.init.arrival(i32 {2 * stages + 1}, i32 16, i32 0)" in compiled.asm["llir"]
    assert f"llvm.musa.async.init.arrival(i32 {2 * stages + 2}, i32 4, i32 0)" in compiled.asm["llir"]

    for _ in range(2):
        out.fill_(float("nan"))
        _ws_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


@pytest.mark.parametrize(
    "capacity,ctas",
    [(2, 1), (3, 1), (8, 1), (3, 60), (8, 60)],
    ids=["stage2-cta1", "stage3-cta1", "stage8-cta1", "stage3-cta60", "stage8-cta60"],
)
def test_mthreads_megamoe_exact_pipe_phase_wrap_runtime(capacity, ctas):
    # Odd capacities make the same physical slot alternate between narrow
    # (256-byte) and wide (512-byte) SFA transactions across generations.
    iterations = 2 * capacity + 1
    block = 128
    # Distinct values detect cross-field aliasing as well as stale ring slots.
    a = (torch.arange(iterations * block, device="musa") % 257).to(torch.float16)
    b = (torch.arange(iterations * block, device="musa") % 251 + 512).to(torch.float16)
    sfa = (torch.arange(2 * iterations * block, device="musa") % 241 + 1024).to(torch.float16)
    sfa = sfa.reshape(2 * iterations, block)
    out = torch.empty((ctas, iterations), dtype=torch.int32, device="musa")
    payload = torch.empty((ctas, iterations, 4, block), dtype=torch.float16, device="musa")
    dispatch_marker = torch.empty((ctas, ), dtype=torch.int32, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block])
    b_desc = TensorDescriptor.from_tensor(b, [block])
    narrow_sfa_desc = TensorDescriptor.from_tensor(sfa, [1, block])
    wide_sfa_desc = TensorDescriptor.from_tensor(sfa, [2, block])

    compiled = _megamoe_exact_pipe_kernel.warmup(
        a_desc,
        b_desc,
        narrow_sfa_desc,
        wide_sfa_desc,
        out,
        payload,
        dispatch_marker,
        STAGES=capacity,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(ctas, ),
        num_warps=4,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 8
    assert '"ttg.total-num-warps" = 8 : i32' in compiled.asm["ttgir"]
    assert re.search(
        r"warpGroupStartIds\s*=\s*array<i32:\s*4,\s*6,\s*7>",
        compiled.asm["ttgir"],
    )
    assert f"llvm.musa.async.bar.record(i32 {3 * capacity + 1})" in compiled.asm["llir"]
    assert f"llvm.musa.async.init.arrival(i32 {3 * capacity + 1}, i32 4, i32 0)" in compiled.asm["llir"]

    expected = torch.arange(iterations, dtype=torch.int32).unsqueeze(0).expand(ctas, -1)
    expected_payload = torch.stack(
        (a.reshape(iterations, block).cpu(), b.reshape(iterations, block).cpu(),
         sfa[0::2].cpu(), sfa[1::2].cpu()),
        dim=1,
    )
    expected_payload[0::2, 3].fill_(-777)
    expected_payload = expected_payload.unsqueeze(0).expand(ctas, -1, -1, -1)
    for _ in range(100):
        out.fill_(-1)
        payload.fill_(float("nan"))
        dispatch_marker.fill_(-1)
        _megamoe_exact_pipe_kernel[(ctas, )](
            a_desc,
            b_desc,
            narrow_sfa_desc,
            wide_sfa_desc,
            out,
            payload,
            dispatch_marker,
            STAGES=capacity,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected)
        torch.testing.assert_close(payload.cpu(), expected_payload, rtol=0, atol=0)
        torch.testing.assert_close(
            dispatch_marker.cpu(), torch.full((ctas, ), 1701, dtype=torch.int32)
        )


def test_mthreads_pipe_bindings_are_optional_and_backend_local():
    assert hasattr(libtriton.ir.builder, "create_pipe_create")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_acquire")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_commit")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_close")
    assert hasattr(libtriton.ir.builder, "create_pipe_reader_wait")
    assert hasattr(libtriton.ir.builder, "create_pipe_reader_release")
    assert hasattr(libtriton.mthreads.passes.ttgpuir, "add_tle_lower_pipe")
