"""A specialized CTA must initialize each barrier from physical thread zero."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _initialization_default(ready, out):
    tle.gpu.barrier_wait(ready[0], phaseIdx=0)
    tle.gpu.barrier_wait(ready[1], phaseIdx=0)
    tl.store(out, 101)


@triton.jit
def _initialization_worker(out, OFFSET: tl.constexpr):
    tl.store(out + OFFSET, OFFSET)


@triton.jit
def _initialization_kernel(out, SPECIALIZE: tl.constexpr):
    ready = tle.gpu.alloc_barriers(2, arrive_count=4, init=tle.gpu.READY)
    if SPECIALIZE:
        tle.gpu.warp_specialize(
            [
                (_initialization_default, (ready, out)),
                (_initialization_worker, (out, 1)),
                (_initialization_worker, (out, 2)),
                (_initialization_worker, (out, 3)),
            ],
            worker_num_warps=[2, 1, 1],
            worker_num_regs=[24, 24, 24],
        )
    else:
        tle.gpu.barrier_wait(ready[0], phaseIdx=0)
        tle.gpu.barrier_wait(ready[1], phaseIdx=0)
        tl.store(out, 101)


@pytest.mark.parametrize("specialize", [False, True])
def test_barrier_initialization_uses_unique_physical_cta_leader(specialize):
    source = ASTSource(
        fn=_initialization_kernel,
        signature={"out": "*i32"},
        constexprs={"SPECIALIZE": specialize},
    )
    compiled = triton.compile(source, target=musa_target(), options={"num_warps": 4, "num_stages": 1})
    llir = compiled.asm["llir"]
    assert compiled.metadata.num_warps == (8 if specialize else 4)
    assert llir.count("call void @llvm.musa.async.init.arrival") == 2, llir

    # Both initialization calls are common setup, before the sole initial
    # CTA rendezvous. Their guarding compare must consume the raw tid,
    # not tid & 127 (which also elects worker thread 128 in an 8-warp CTA).
    setup = llir.split("call void @llvm.musa.syncthreads.lm", 1)[0]
    tids = re.findall(
        r"(%[-\w.]+)\s*=\s*(?:tail\s+)?call\s+i32\s+@llvm\.musa\.read\.ptx\.sreg\.tid\.x\(\)",
        setup,
    )
    assert tids, llir
    definitions = {
        value: rhs for value, rhs in re.findall(r"^\s*(%[-\w.]+)\s*=\s*(.+)$", setup, re.MULTILINE)
    }
    leaders = []
    for result, operand in re.findall(r"(%[-\w.]+)\s*=\s*icmp\s+eq\s+i32\s+(%[-\w.]+),\s*0", setup):
        if operand in tids:
            leaders.append(result)
        elif not specialize:
            # Preserve the non-specialized backend's existing local-ID mask.
            rhs = definitions.get(operand, "")
            if any(re.match(rf"and i32 {re.escape(tid)}, 127(?:,|$)", rhs) for tid in tids):
                leaders.append(result)
    assert leaders, setup
    assert any(re.search(rf"br i1 {re.escape(leader)},", setup) for leader in leaders), setup
    if specialize:
        assert not any(re.search(rf"and i32 {re.escape(tid)}, 127", setup) for tid in tids), setup
