"""CPU-only coverage for end-to-end ``loop_unroll_factor=1`` semantics.

The tests build TTIR through optimized LLVM IR but never create a MUSA tensor,
load the runtime driver, or launch a kernel.
"""

from __future__ import annotations

import re

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource

from test_tle_utils import mthreads_backend, require_mthreads_libtriton


require_mthreads_libtriton()


@triton.jit
def _default_loop(out):
    for i in tl.range(0, 3):
        tl.atomic_add(out, i + 1)


@triton.jit
def _no_unroll_loop(out):
    for i in tl.range(0, 3, loop_unroll_factor=1):
        tl.atomic_add(out, i + 1)


@triton.jit
def _factor_two_loop(out):
    for i in tl.range(0, 3, loop_unroll_factor=2):
        tl.atomic_add(out, i + 1)


@triton.jit
def _no_unroll_and_no_licm_loop(out):
    for i in tl.range(
        0,
        3,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        tl.atomic_add(out, i + 1)


def _compile_through_llir(fn):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=fn, signature={"out": "*i32"})
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    metadata = {}
    module = stages["ttir"](module, metadata)
    ttir = str(module)
    module = stages["ttgir"](module, metadata)
    llir = stages["llir"](module, metadata)
    return ttir, llir


def _loop_annotation_uses(ttir):
    return [
        line
        for line in ttir.splitlines()
        if "llvm.loop_annotation = #loop_annotation" in line
    ]


def test_explicit_factor_one_emits_llvm_unroll_disable():
    ttir, llir = _compile_through_llir(_no_unroll_loop)
    assert len(_loop_annotation_uses(ttir)) == 1
    assert "#llvm.loop_unroll<disable = true>" in ttir
    assert "#llvm.loop_annotation<unroll = #loop_unroll>" in ttir
    assert "tt.loop_unroll_factor" not in ttir
    assert "llvm.loop.unroll.disable" in llir


def test_default_loop_keeps_backend_unroll_policy():
    ttir, llir = _compile_through_llir(_default_loop)
    assert not _loop_annotation_uses(ttir)
    assert "tt.loop_unroll_factor" not in ttir
    assert "llvm.loop.unroll.disable" not in llir


def test_factor_greater_than_one_keeps_ir_unroll_semantics():
    ttir, llir = _compile_through_llir(_factor_two_loop)
    assert "tt.loop_unroll_factor" not in ttir
    assert "llvm.loop.unroll.disable" not in ttir
    assert "llvm.loop.unroll.disable" not in llir
    # Factor two must still duplicate the side-effecting body before LLVM.
    assert len(re.findall(r"tt\.atomic_rmw", ttir)) >= 2


def test_factor_one_merges_existing_licm_annotation():
    ttir, llir = _compile_through_llir(_no_unroll_and_no_licm_loop)
    assert len(_loop_annotation_uses(ttir)) == 1
    assert "#llvm.loop_unroll<disable = true>" in ttir
    assert "#llvm.loop_licm<disable = true" in ttir
    assert "#llvm.loop_annotation<unroll = #loop_unroll, licm = #loop_licm>" in ttir
    assert "llvm.loop.unroll.disable" in llir
    assert "llvm.licm.disable" in llir
