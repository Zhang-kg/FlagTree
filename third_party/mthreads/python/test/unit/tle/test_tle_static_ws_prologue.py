"""Common-prologue side effects must not repeat on specialized worker warps."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _prologue_role_marker(markers, ROLE: tl.constexpr, VALUE: tl.constexpr):
    tl.store(markers + tl.program_id(0) * 4 + ROLE, VALUE)


@triton.jit
def _prologue_side_effects_kernel(counts, stores, cas_values, markers, SPECIALIZE: tl.constexpr):
    base = tl.program_id(0) * 129
    offsets = tl.arange(0, 128)

    # Both a scalar effect and a full-default-role tensor effect. Without
    # the physical-thread bound, worker tids 128..255 repeat this prologue
    # with local tids 0..127 and increment every counter twice.
    tl.atomic_add(counts + base, 1)
    tl.atomic_add(counts + base + 1 + offsets, 1)
    tl.store(stores + base, 7)
    tl.store(stores + base + 1 + offsets, offsets)
    tl.atomic_cas(cas_values + base, 0, 7)
    tl.atomic_cas(
        cas_values + base + 1 + offsets,
        tl.full((128, ), 0, tl.int32),
        tl.full((128, ), 9, tl.int32),
    )

    if SPECIALIZE:
        tle.gpu.warp_specialize(
            [
                (_prologue_role_marker, (markers, 0, 11)),
                (_prologue_role_marker, (markers, 1, 22)),
                (_prologue_role_marker, (markers, 2, 33)),
                (_prologue_role_marker, (markers, 3, 44)),
            ],
            worker_num_warps=[2, 1, 1],
            worker_num_regs=[24, 24, 24],
        )
    else:
        _prologue_role_marker(markers, 0, 11)


@pytest.mark.parametrize("specialize", [False, True])
def test_static_ws_prologue_side_effects_compile(specialize):
    source = ASTSource(
        fn=_prologue_side_effects_kernel,
        signature={"counts": "*i32", "stores": "*i32", "cas_values": "*i32", "markers": "*i32"},
        constexprs={"SPECIALIZE": specialize},
    )
    compiled = triton.compile(source, target=musa_target(), options={"num_warps": 4, "num_stages": 1})
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert compiled.metadata.num_warps == (8 if specialize else 4)
    assert ttgir.count("tt.atomic_rmw") == 2, ttgir
    assert ttgir.count("tt.atomic_cas") == 2, ttgir
    # LLVM may tail-duplicate vector operations across the scalar leader
    # branch, so require exact source operation counts before CFG folding.
    assert llir.count("atomicrmw add") >= 2, llir
    assert llir.count("cmpxchg") >= 2, llir
    assert compiled.asm["mubin"]
    if specialize:
        # Scalar prologue effects must be elected from physical tid zero,
        # not from a masked local ID. LLVM folds the default-range bound
        # together with the scalar layout predicate to this direct compare.
        tids = re.findall(
            r"(%[-\w.]+)\s*=\s*(?:tail\s+)?call\s+i32\s+@llvm\.musa\.read\.ptx\.sreg\.tid\.x\(\)",
            llir,
        )
        first_atomic = llir.index("atomicrmw add")
        before_atomic = llir[:first_atomic]
        assert any(re.search(rf"icmp eq i32 {re.escape(tid)}, 0", before_atomic) for tid in tids), llir
        assert any(re.search(rf"icmp ult i32 {re.escape(tid)}, 128", llir) for tid in tids), llir


@pytest.mark.parametrize("specialize", [False, True])
def test_static_ws_prologue_scalar_and_vector_effects_runtime(specialize):
    ctas = 7
    for _ in range(2):
        counts = torch.zeros((ctas, 129), dtype=torch.int32, device="musa")
        stores = torch.full_like(counts, -1)
        cas_values = torch.zeros_like(counts)
        markers = torch.zeros((ctas, 4), dtype=torch.int32, device="musa")
        _prologue_side_effects_kernel[(ctas, )](
            counts, stores, cas_values, markers, SPECIALIZE=specialize, num_warps=4, num_stages=1
        )
        torch.musa.synchronize()

        # This is the decisive duplicate-side-effect check: every logical
        # scalar/tensor element must be incremented exactly once per CTA.
        torch.testing.assert_close(counts.cpu(), torch.ones((ctas, 129), dtype=torch.int32))
        expected_stores = torch.cat((torch.tensor([7], dtype=torch.int32), torch.arange(128, dtype=torch.int32)))
        torch.testing.assert_close(stores.cpu(), expected_stores.expand(ctas, -1))
        expected_cas = torch.full((ctas, 129), 9, dtype=torch.int32)
        expected_cas[:, 0] = 7
        torch.testing.assert_close(cas_values.cpu(), expected_cas)
        expected_markers = torch.tensor([11, 22, 33, 44] if specialize else [11, 0, 0, 0], dtype=torch.int32)
        torch.testing.assert_close(markers.cpu(), expected_markers.expand(ctas, -1))
