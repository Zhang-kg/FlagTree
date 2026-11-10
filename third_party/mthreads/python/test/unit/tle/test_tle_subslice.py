"""Frontend and IR coverage for shared-memory buffered_tensor subslices."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_to_ttir, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _subslice_kernel(out, START: tl.constexpr, LENGTH: tl.constexpr, DIM: tl.constexpr):
    buffer = tle.gpu.alloc((4, 2, 128), dtype=tl.float16, nv_mma_shared_layout=False)
    view = buffer.slot(0).subslice(START, LENGTH, DIM)
    tl.static_assert(view.type.alloc_shape[0] == 2)
    tl.static_assert(view.type.alloc_shape[1] == 128)
    tl.store(out, view.shape[0] * view.shape[1])


@triton.jit
def _nested_subslice_kernel(out):
    buffer = tle.gpu.alloc((4, 2, 128), dtype=tl.float16, nv_mma_shared_layout=False)
    slot = buffer.slot(0)
    first = slot.subslice(64, 64)
    second = first.subslice(1, 1, 0)
    tl.static_assert(second.shape[0] == 1)
    tl.static_assert(second.shape[1] == 64)
    tl.static_assert(second.type.alloc_shape[0] == 2)
    tl.static_assert(second.type.alloc_shape[1] == 128)
    tl.static_assert(second.type.layout == slot.type.layout)
    tl.store(out, second.shape[0] * second.shape[1])


@triton.jit
def _dynamic_subslice_kernel(out, start):
    buffer = tle.gpu.alloc((2, 128), dtype=tl.float16, nv_mma_shared_layout=False)
    view = buffer.subslice(start, 64)
    tl.store(out, view.shape[1])


@pytest.mark.parametrize("start,length,dim", [(0, 1, 0), (1, 1, -2), (64, 64, 1), (64, 64, -1)])
def test_tle_buffered_tensor_subslice_compile(start, length, dim):
    ttir = compile_to_ttir(
        _subslice_kernel,
        {"out": "*i32"},
        {"START": start, "LENGTH": length, "DIM": dim},
    )
    assert "ttg.memdesc_index" in ttir
    assert "ttg.memdesc_subslice" in ttir


def test_tle_buffered_tensor_subslice_keeps_parent_allocation_and_layout():
    ttir = compile_to_ttir(_nested_subslice_kernel, {"out": "*i32"})
    assert ttir.count("ttg.memdesc_subslice") == 2


@pytest.mark.parametrize(
    "start,length,dim,diagnostic",
    [
        (-1, 1, 0, "start must be nonnegative"),
        (0, 0, 0, "length must be positive"),
        (0, -1, 0, "length must be positive"),
        (1, 2, 0, "out of bounds for dimension 0 of size 2"),
        (0, 1, 2, "dim 2 out of bounds for rank 2"),
        (0, 1, -3, "dim -3 out of bounds for rank 2"),
        (0.0, 1, 0, "start must be a compile-time integer"),
        (True, 1, 0, "start must be a compile-time integer"),
        (0, 1.0, 0, "length must be a compile-time integer"),
        (0, 1, False, "dim must be a compile-time integer"),
    ],
)
def test_tle_buffered_tensor_subslice_rejects_invalid_static_arguments(start, length, dim, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        compile_to_ttir(
            _subslice_kernel,
            {"out": "*i32"},
            {"START": start, "LENGTH": length, "DIM": dim},
        )


def test_tle_buffered_tensor_subslice_rejects_dynamic_start():
    with pytest.raises(CompilationError, match="start must be a compile-time integer"):
        compile_to_ttir(_dynamic_subslice_kernel, {"out": "*i32", "start": "i32"})
