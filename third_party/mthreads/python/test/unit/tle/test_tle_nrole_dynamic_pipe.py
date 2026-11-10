"""Runtime coverage for dynamic-phase, multi-writer MThreads TLE pipes."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _dynamic_math_reader(reader, out, payload, iterations, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.range(0, iterations, num_stages=1):
        wait = reader.wait(iteration)
        a = tl.load(tle.gpu.local_ptr(wait.slot.a, (offsets, )))
        b = tl.load(tle.gpu.local_ptr(wait.slot.b, (offsets, )))
        sfa = tl.load(tle.gpu.local_ptr(wait.slot.sfa, (offsets, )))
        base = payload + iteration * 3 * BLOCK + offsets
        tl.store(base, a)
        tl.store(base + BLOCK, b)
        tl.store(base + 2 * BLOCK, sfa)
        reader.release(iteration)
        tl.store(out + iteration, iteration + tl.where(wait.is_closed, 1000, 0))


@triton.jit
def _dynamic_dispatch_role(dispatch_marker):
    tl.store(dispatch_marker, 1701)


@triton.jit
def _dynamic_a_sfa_writer(writer, a_desc, sfa_desc, iterations, BLOCK: tl.constexpr):
    for iteration in tl.range(0, iterations, num_stages=1):
        slot = writer.acquire(iteration)
        offset = iteration * BLOCK
        tle.gpu.copy(a_desc, slot.a, (BLOCK, ), (offset, ))
        tle.gpu.copy(sfa_desc, slot.sfa, (BLOCK, ), (offset, ))
        writer.commit(iteration)


@triton.jit
def _dynamic_b_writer(writer, b_desc, iterations, BLOCK: tl.constexpr):
    for iteration in tl.range(0, iterations, num_stages=1):
        slot = writer.acquire(iteration)
        tle.gpu.copy(b_desc, slot.b, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _nrole_dynamic_pipe_kernel(
    a_desc,
    b_desc,
    sfa_desc,
    out,
    payload,
    dispatch_marker,
    iterations,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Only capacity and tile shape are constexpr. The loop limit remains an
    # i32 kernel argument and is captured by both worker writer partitions.
    a = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    sfa = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="dynamic_nrole", readers=("math", ), a=a, b=b, sfa=sfa)
    tle.gpu.warp_specialize(
        [
            (_dynamic_math_reader, (pipe.reader(name="math"), out, payload, iterations, BLOCK)),
            (_dynamic_dispatch_role, (dispatch_marker, )),
            (_dynamic_a_sfa_writer, (pipe.writer(), a_desc, sfa_desc, iterations, BLOCK)),
            (_dynamic_b_writer, (pipe.writer(), b_desc, iterations, BLOCK)),
        ],
        worker_num_warps=[2, 1, 1],
        worker_num_regs=[24, 24, 24],
    )


def _assert_dynamic_pipe_contract(compiled, capacity):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    # One runtime loop in math and one in each writer; these are not expanded
    # into 37 static acquire/commit/wait pairs.
    assert ttir.count("scf.for") == 3, ttir
    assert ttir.count("tle.pipe.writer_acquire") == 2, ttir
    assert ttir.count("tle.pipe.writer_commit") == 2, ttir
    assert ttir.count("tle.pipe.reader_wait") == 1, ttir
    assert ttir.count("tle.pipe.reader_release") == 1, ttir
    assert compiled.metadata.num_warps == 8
    assert '"ttg.total-num-warps" = 8 : i32' in ttgir, ttgir
    assert re.search(r"warpGroupStartIds\s*=\s*array<i32:\s*4,\s*6,\s*7>", ttgir), ttgir
    assert "tle.pipe." not in ttgir, ttgir
    assert "ttg.warp_specialize" not in llir, llir
    constants = {name: int(value) for name, value in re.findall(
        r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ttgir
    )}
    arrivals = re.findall(r"ttmg\.init_arrival\s+(%[-\w.]+),\s*(%[-\w.]+),\s*(%[-\w.]+)", ttgir)
    assert len(arrivals) == 3 * capacity, ttgir
    assert [constants[count] for _, count, _ in arrivals] == [1] * (3 * capacity), ttgir
    assert f"musa.max_bar_id = {3 * capacity}" in ttgir, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 4, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 2, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 2, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 3, ttgir
    reader = ttgir.split("default {", 1)[1].split("ttg.warp_yield", 1)[0]
    indices = {result: (base, slot) for result, base, slot in re.findall(
        r"(%[-\w.]+)\s*=\s*arith\.addi\s+(%[-\w.]+),\s*(%[-\w.]+)\s*:\s*i32", reader
    )}
    waits = list(re.finditer(r"ttmg\.wait_barrier\s+(%[-\w.]+),\s*(%[-\w.]+)", reader))
    assert len(waits) == 2, reader
    full_indices = [indices[wait.group(1)] for wait in waits]
    assert {constants[base] for base, _ in full_indices} == {1, capacity + 1}, reader
    assert full_indices[0][1] == full_indices[1][1], reader
    assert waits[0].group(2) == waits[1].group(2), reader
    assert waits[-1].end() < reader.index("ttg.local_load"), reader
    release = re.search(r"ttmg\.warp_arrive_barrier\s+(%[-\w.]+),[^\n]+", reader)
    assert release and "musa_tle.pipe_empty_arrival" in release.group(0), reader
    assert "musa.tme.issue_thread = 0 : i32" in release.group(0), reader
    empty_base, empty_slot = indices[release.group(1)]
    assert constants[empty_base] == 2 * capacity + 1 and empty_slot == full_indices[0][1], reader
    rendezvous = list(re.finditer(r"ttg\.barrier local", reader))
    assert len(rendezvous) == 2, reader
    assert rendezvous[0].start() < waits[0].start() < waits[-1].end() < rendezvous[1].start() < release.start(), reader
    # The elected first reader warp publishes empty, but all four reader
    # warps still rendezvous using a separate count-four hardware barrier.
    assert f"llvm.musa.async.bar.record(i32 {3 * capacity + 1})" in llir, llir
    assert f"llvm.musa.async.init.arrival(i32 {3 * capacity + 1}, i32 4, i32 0)" in llir, llir


@pytest.mark.parametrize("capacity", [2, 3, 8], ids=["stage2", "stage3", "stage8"])
def test_mthreads_four_role_multi_writer_pipe_dynamic_phase_runtime(capacity):
    iterations = 37
    block = 128
    source_indices = torch.arange(iterations * block, dtype=torch.int32, device="musa")
    a = (source_indices % 257).to(torch.float16)
    b = (source_indices % 251 + 512).to(torch.float16)
    sfa = (source_indices % 241 + 1024).to(torch.float16)
    out = torch.empty((iterations, ), dtype=torch.int32, device="musa")
    payload = torch.empty((iterations, 3, block), dtype=torch.float16, device="musa")
    dispatch_marker = torch.empty((1, ), dtype=torch.int32, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block])
    b_desc = TensorDescriptor.from_tensor(b, [block])
    sfa_desc = TensorDescriptor.from_tensor(sfa, [block])

    compiled = _nrole_dynamic_pipe_kernel.warmup(
        a_desc,
        b_desc,
        sfa_desc,
        out,
        payload,
        dispatch_marker,
        iterations,
        STAGES=capacity,
        BLOCK=block,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_dynamic_pipe_contract(compiled, capacity)

    expected_out = torch.arange(iterations, dtype=torch.int32)
    expected_payload = torch.stack(
        (a.reshape(iterations, block).cpu(), b.reshape(iterations, block).cpu(),
         sfa.reshape(iterations, block).cpu()),
        dim=1,
    )
    for _ in range(20):
        out.fill_(-1)
        payload.fill_(float("nan"))
        dispatch_marker.fill_(-1)
        _nrole_dynamic_pipe_kernel[(1, )](
            a_desc,
            b_desc,
            sfa_desc,
            out,
            payload,
            dispatch_marker,
            iterations,
            STAGES=capacity,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected_out)
        torch.testing.assert_close(payload.cpu(), expected_payload, rtol=0, atol=0)
        torch.testing.assert_close(dispatch_marker.cpu(), torch.tensor([1701], dtype=torch.int32))
