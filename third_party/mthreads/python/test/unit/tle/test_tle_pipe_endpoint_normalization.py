"""Compile-only regression coverage for writer-endpoint arrival normalization.

These tests never allocate or launch a MUSA tensor.  They stop after TTGIR and
inspect the full/empty barrier protocol emitted by LowerPipe.
"""

from __future__ import annotations

from collections import defaultdict
import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource

from test_tle_utils import mthreads_backend, require_mthreads_libtriton, tme_descriptor_attrs


require_mthreads_libtriton()

_ONE_WARP_LOCAL = tl.constexpr(tle.gpu.BlockEncoding([1], [32], [1], [0]))


@triton.jit
def _endpoint_reader(reader, out, ITERATIONS: tl.constexpr):
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        wait = reader.wait(iteration)
        tl.store(out, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _heterogeneous_writer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    """One endpoint with TME-only, local-only, and mixed commit paths."""
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _ONE_WARP_LOCAL)
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        slot = writer.acquire(iteration)
        kind = iteration % 3
        if kind == 0:
            tle.gpu.copy(desc, slot.a, (BLOCK, ), (iteration * BLOCK, ))
            writer.commit(iteration)
        elif kind == 1:
            tl.store(tle.gpu.local_ptr(slot.a, (offsets, )), offsets)
            writer.commit(iteration)
        else:
            tle.gpu.copy(desc, slot.a, (BLOCK, ), (iteration * BLOCK, ))
            tl.store(tle.gpu.local_ptr(slot.a, (offsets, )), offsets)
            writer.commit(iteration)


@triton.jit
def _tme_only_writer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.b, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _singlewriter_endpoint_signature_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    a = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="singlewriter_endpoint_signature", a=a)
    tle.gpu.warp_specialize(
        [
            (_endpoint_reader, (pipe.reader(), out, ITERATIONS)),
            (_heterogeneous_writer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        [1],
        [48],
    )


@triton.jit
def _multiwriter_endpoint_signature_kernel(
    desc_a,
    desc_b,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    a = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float32, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="multiwriter_endpoint_signature", a=a, b=b)
    tle.gpu.warp_specialize(
        [
            (_endpoint_reader, (pipe.reader(), out, ITERATIONS)),
            (_heterogeneous_writer, (pipe.writer(), desc_a, BLOCK, ITERATIONS)),
            (_tme_only_writer, (pipe.writer(), desc_b, BLOCK, ITERATIONS)),
        ],
        [1, 1],
        [48, 48],
    )


def _compile_ttgir(fn, signature):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=fn,
        signature=signature,
        constexprs={"STAGES": 2, "BLOCK": 128, "ITERATIONS": 6},
        attrs=tme_descriptor_attrs(signature),
    )
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
    return str(stages["ttgir"](module, metadata))


_SSA = r"%[-A-Za-z0-9_.$]+"
_CONST = re.compile(rf"^\s*({_SSA}) = arith\.constant (-?\d+) :")
_ADDI = re.compile(rf"^\s*({_SSA}) = arith\.addi ({_SSA}), ({_SSA}) :")
_INIT = re.compile(rf"ttmg\.init_arrival ({_SSA}), ({_SSA}), ({_SSA})")


def _definitions(lines):
    result = defaultdict(list)
    for line_number, text in enumerate(lines, 1):
        if match := _CONST.match(text):
            result[match.group(1)].append((line_number, "const", (int(match.group(2)), )))
        elif match := _ADDI.match(text):
            result[match.group(1)].append((line_number, "addi", (match.group(2), match.group(3))))
    return result


def _constant_candidates(definitions, value, line_number, seen=None):
    seen = set() if seen is None else set(seen)
    key = (value, line_number)
    if key in seen:
        return set()
    seen.add(key)
    prior = [item for item in definitions.get(value, ()) if item[0] < line_number]
    if not prior:
        return set()
    definition_line, kind, operands = prior[-1]
    if kind == "const":
        return {operands[0]}
    result = set()
    for operand in operands:
        result.update(_constant_candidates(definitions, operand, definition_line, seen))
    return result


def _sole_constant(definitions, value, line_number):
    candidates = _constant_candidates(definitions, value, line_number)
    assert len(candidates) == 1, (line_number, value, candidates)
    return next(iter(candidates))


def _find_ring_topology(ttgir, expected_arrivals):
    lines = ttgir.splitlines()
    definitions = _definitions(lines)
    records = []
    for line_number, text in enumerate(lines, 1):
        if match := _INIT.search(text):
            records.append(
                (
                    _sole_constant(definitions, match.group(1), line_number),
                    _sole_constant(definitions, match.group(2), line_number),
                    _sole_constant(definitions, match.group(3), line_number),
                )
            )
    width = len(expected_arrivals)
    for start in range(len(records) - width + 1):
        window = records[start:start + width]
        ids = [record[0] for record in window]
        arrivals = [record[1] for record in window]
        phases = [record[2] for record in window]
        if ids == list(range(ids[0], ids[0] + width)) and arrivals == expected_arrivals and phases == [0] * width:
            return lines, definitions, ids[0]
    pytest.fail(f"expected barrier topology {expected_arrivals}, got {records}")


def _count_ring_ops(lines, definitions, pattern, base):
    expression = re.compile(rf"{pattern} ({_SSA})[, ]")
    count = 0
    for line_number, text in enumerate(lines, 1):
        match = expression.search(text)
        if match and base in _constant_candidates(definitions, match.group(1), line_number):
            count += 1
    return count


def _assert_heterogeneous_endpoint_is_normalized(ttgir, *, multiwriter):
    # capacity=2.  The heterogeneous endpoint requires one TME control plus
    # one one-warp local arrival in every generation.  The reader contributes
    # four empty arrivals.  A second TME-only writer, when present, owns a
    # separate ring with arrival count one.
    arrivals = [2, 2, 4, 4]
    if multiwriter:
        arrivals = [2, 2, 1, 1, 4, 4]
    lines, definitions, first_base = _find_ring_topology(ttgir, arrivals)
    heterogeneous_base = first_base

    # Three syntactic commit paths exist: TME-only, local-only, and mixed.
    # Endpoint normalization makes all three publish both endpoint-owned
    # control components.  Only the two paths with a real TME copy add bytes.
    assert _count_ring_ops(lines, definitions, r"ttmg\.arrive_barrier_noret", heterogeneous_base) == 3
    assert _count_ring_ops(lines, definitions, r"ttmg\.warp_arrive_barrier", heterogeneous_base) == 3
    assert _count_ring_ops(lines, definitions, r"ttmg\.barrier_add_trans", heterogeneous_base) == 2

    if multiwriter:
        tme_only_base = first_base + 2
        assert _count_ring_ops(lines, definitions, r"ttmg\.arrive_barrier_noret", tme_only_base) == 1
        assert _count_ring_ops(lines, definitions, r"ttmg\.warp_arrive_barrier", tme_only_base) == 0
        assert _count_ring_ops(lines, definitions, r"ttmg\.barrier_add_trans", tme_only_base) == 1


def test_singlewriter_heterogeneous_commit_paths_keep_legacy_normalization():
    ttgir = _compile_ttgir(
        _singlewriter_endpoint_signature_kernel,
        {
            "desc": "tensordesc<fp32[128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
    )
    _assert_heterogeneous_endpoint_is_normalized(ttgir, multiwriter=False)


def test_multiwriter_heterogeneous_commit_paths_are_endpoint_normalized():
    """Regression: v11 emitted only two arrivals of each kind and could stall."""
    ttgir = _compile_ttgir(
        _multiwriter_endpoint_signature_kernel,
        {
            "desc_a": "tensordesc<fp32[128]>",
            "desc_b": "tensordesc<fp32[128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
    )
    _assert_heterogeneous_endpoint_is_normalized(ttgir, multiwriter=True)
