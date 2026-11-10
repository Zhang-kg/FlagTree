"""Compile coverage for mixed-width SQMMA views of one TME landing tile."""

import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_musa, require_mthreads_libtriton


require_mthreads_libtriton()


@triton.jit
def _mixed_subslice_producer(writer, a_desc, b_desc):
    slot = writer.acquire(0)
    tle.gpu.copy(a_desc, slot.a, (256, 64), (0, 0))
    # Exercise TME lowering with a narrowed destination view whose backing
    # allocation is still the full 64x256 matrix.
    b_subslice = slot.b.slice(0, 128, dim=1)
    tle.gpu.copy(b_desc, b_subslice, (64, 128), (0, 0))
    writer.commit(0)


@triton.jit
def _mixed_subslice_consumer(
    reader,
    sequence,
    out,
    group_id: tl.constexpr,
    col_offset: tl.constexpr,
    block_cols: tl.constexpr,
):
    accumulator = tl.zeros((256, block_cols), dtype=tl.float32)
    ready = reader.wait(0)
    wait_phase: tl.constexpr = 1 if group_id == 0 else 0
    tle.gpu.barrier_wait(sequence[group_id], phaseIdx=wait_phase)
    b_tile = ready.slot.b.slice(col_offset, block_cols, dim=1)
    accumulator = tle.gpu.wgmma(ready.slot.a, b_tile, accumulator)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    reader.release(0)

    offsets_m = tl.arange(0, 256)
    offsets_n = col_offset + tl.arange(0, block_cols)
    offsets = offsets_m[:, None] * 256 + offsets_n[None, :]
    tl.store(out + offsets, accumulator.to(tl.bfloat16))


@triton.jit
def _mixed_subslice_kernel(a_desc, b_desc, out):
    a_smem = tle.gpu.alloc(
        (2, 256, 64),
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (2, 64, 256),
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    sequence = tle.gpu.alloc_barriers(
        4, arrive_count=4, init=tle.gpu.PENDING
    )
    pipe = tle.pipe(
        capacity=2,
        scope="cta",
        name="mixed_subslice_ab",
        a=a_smem,
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _mixed_subslice_consumer,
                (pipe.reader(), sequence, out, 0, 0, 128),
            ),
            (
                _mixed_subslice_consumer,
                (pipe.reader(), sequence, out, 1, 128, 64),
            ),
            (
                _mixed_subslice_consumer,
                (pipe.reader(), sequence, out, 2, 192, 64),
            ),
            (
                _mixed_subslice_producer,
                (pipe.writer(), a_desc, b_desc),
            ),
        ],
        worker_num_warps=[4, 4, 4],
        worker_num_regs=[168, 168, 24],
    )


def test_mthreads_tle_mixed_width_sqmma_subslices_compile():
    compiled = compile_musa(
        _mixed_subslice_kernel,
        signature={
            "a_desc": "tensordesc<bf16[256, 64]>",
            "b_desc": "tensordesc<bf16[64, 128]>",
            "out": "*bf16",
        },
    )
    assert "ttg.tma_copy" not in compiled.asm["ttgir"]
    assert "ttmg.async_tme_copy_global_to_local" in compiled.asm["ttgir"]
