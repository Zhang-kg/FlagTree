"""Role-context layouts for real vector computation in 4+2+1+1 workers."""

import math
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target
from test_tle_warp_specialize import _assert_four_role_ttgir, _compile_ws_ir


@triton.jit
def _vector_role(out, counters, rounds, ROLE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.full((BLOCK,), ROLE + 1, tl.int32)
    counter = counters + pid * 4 + ROLE
    observed = tl.atomic_add(counter, 0, sem="acquire")
    while observed < 0:
        observed = tl.atomic_add(counter, 0, sem="acquire")
    iteration = 0
    while iteration < rounds:
        if (iteration & 1) == 0:
            values += offsets + iteration
        else:
            values -= offsets - iteration
        tl.debug_barrier()
        tl.atomic_add(counter, 1, sem="release")
        iteration += 1
    tl.store(out + (pid * 4 + ROLE) * BLOCK + offsets, values)


@triton.jit
def _vector_four_role_kernel(out, counters, rounds, BLOCK: tl.constexpr):
    tle.gpu.warp_specialize(
        [
            (_vector_role, (out, counters, rounds, 0, BLOCK)),
            (_vector_role, (out, counters, rounds, 1, BLOCK)),
            (_vector_role, (out, counters, rounds, 2, BLOCK)),
            (_vector_role, (out, counters, rounds, 3, BLOCK)),
        ],
        worker_num_warps=[2, 1, 1],
        worker_num_regs=[24, 24, 24],
    )


@triton.jit
def _capture_default(out):
    tl.store(out, 1)


@triton.jit
def _capture_worker(out, values):
    offsets = tl.arange(0, 128)
    tl.store(out + 128 + offsets, values)


@triton.jit
def _distributed_capture_kernel(out, source):
    values = tl.load(source + tl.arange(0, 128))
    tle.gpu.warp_specialize(
        [(_capture_default, (out,)), (_capture_worker, (out, values))],
        worker_num_warps=[2],
        worker_num_regs=[24],
    )


def _matching_delimiter(text, start, opening, closing):
    """Return the end of a nested IR group, ignoring quoted location names."""
    assert text[start] == opening
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise AssertionError(f"Unclosed {opening} group in IR: {text[start:]}")


def _assert_role_range_layouts(ttgir):
    blocked_warps = {
        alias: math.prod(int(value.strip()) for value in warps.split(","))
        for alias, warps in re.findall(
            r"^(#\w+) = #ttg\.blocked<\{[^\n]*warpsPerCTA = \[([^]]+)\]",
            ttgir,
            re.MULTILINE,
        )
    }
    default = re.search(r"\bdefault\s*\{", ttgir)
    assert default, ttgir
    default_start = default.end() - 1
    default_end = _matching_delimiter(ttgir, default_start, "{", "}")
    bodies = [(4, ttgir[default_start + 1:default_end - 1])]
    for index, warps in enumerate((2, 1, 1)):
        partition = re.search(rf"\bpartition{index}\(", ttgir)
        assert partition, ttgir
        # Full compiled IR includes loc("argument"(#loc)) in the argument
        # list. Match its balanced closing parenthesis, not the first ')'.
        args_end = _matching_delimiter(ttgir, partition.end() - 1, "(", ")")
        header = re.match(r"\s*num_warps\((\d+)\)\s*\{", ttgir[args_end:])
        assert header and int(header.group(1)) == warps, ttgir
        body_start = args_end + header.end() - 1
        body_end = _matching_delimiter(ttgir, body_start, "{", "}")
        body = ttgir[body_start + 1:body_end - 1]
        assert "ttg.warp_return" in body, body
        bodies.append((warps, body))
    for expected_warps, body in bodies:
        aliases = re.findall(r"tt\.make_range[^\n]*,\s*(#\w+)>", body)
        assert aliases, body
        for alias in aliases:
            assert blocked_warps.get(alias) == expected_warps, (alias, body, ttgir)


@pytest.mark.parametrize("block", [128, 256])
def test_mthreads_tle_worker_vector_contextual_layout_compile(block):
    compiled = triton.compile(
        ASTSource(
            fn=_vector_four_role_kernel,
            signature={"out": "*i32", "counters": "*i32", "rounds": "i32"},
            constexprs={"BLOCK": block},
        ),
        target=musa_target(),
        options={"num_warps": 4, "num_stages": 1},
    )
    ttir, ttgir = compiled.asm["ttir"], compiled.asm["ttgir"]
    assert "scf.while" in ttir, ttir
    assert "tt.atomic_rmw" in ttir, ttir
    _assert_four_role_ttgir(ttgir)
    _assert_role_range_layouts(ttgir)
    # The helper's partial LLVM lowering precedes reconciliation and can
    # legitimately retain struct <-> tensor bridge casts. Check the actual
    # compiler's final LLVM IR, after the complete production pipeline.
    llir = compiled.asm["llir"]
    assert compiled.metadata.num_warps == 8
    assert "ttg.warp_specialize" not in llir, llir
    assert "builtin.unrealized_conversion_cast" not in llir, llir


@pytest.mark.parametrize("block", [128, 256])
def test_mthreads_tle_worker_vector_contextual_layout_runtime(block):
    grid = 3
    rounds = 7
    out = torch.empty((grid, 4, block), dtype=torch.int32, device="musa")
    counters = torch.empty((grid, 4), dtype=torch.int32, device="musa")
    offsets = torch.arange(block, dtype=torch.int32)
    expected = torch.empty((grid, 4, block), dtype=torch.int32)
    for role in range(4):
        values = torch.full((block,), role + 1, dtype=torch.int32)
        for iteration in range(rounds):
            values += offsets + iteration if iteration % 2 == 0 else iteration - offsets
        expected[:, role, :] = values
    for _ in range(4):
        out.fill_(-99)
        counters.zero_()
        _vector_four_role_kernel[(grid,)](
            out, counters, rounds, BLOCK=block, num_warps=4, num_stages=1)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            counters.cpu(), torch.full((grid, 4), rounds, dtype=torch.int32),
            rtol=0, atol=0)


def test_mthreads_tle_worker_rejects_distributed_tensor_capture(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ws_ir(
            _distributed_capture_kernel,
            signature={"out": "*i32", "source": "*i32"},
            constexprs={},
            consumer_warps=4,
        )
    assert "does not support distributed tensor captures" in capfd.readouterr().err
