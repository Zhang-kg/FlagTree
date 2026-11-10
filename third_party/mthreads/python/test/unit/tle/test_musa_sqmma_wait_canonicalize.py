"""Regression coverage for MUSA SQMMA wait canonicalization."""

from triton._C import libtriton
from triton._C.libtriton import ir, passes

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


def _parse_module(tmp_path, body, name="sqmma_waits"):
    fixture = f"""module attributes {{"ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1",
    "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @{name}(%arg0: tensor<16x16xf32>,
                         %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {{
{body}    tt.return %second : tensor<16x16xf32>
  }}
}}
"""
    fixture_path = tmp_path / f"{name}.ttgir"
    fixture_path.write_text(fixture)
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    return context, ir.parse_mlir_module(str(fixture_path), context)


def _canonicalize(module, context):
    pm = ir.pass_manager(context)
    passes.common.add_canonicalizer(pm)
    pm.run(module, "canonicalize_musa_sqmma_waits")
    return module.str_nodebug()


def test_musa_sqmma_wait_canonicalizer_merges_adjacent_waits(tmp_path):
    body = """    %first = ttmg.squad_dot_wait %arg0 : tensor<16x16xf32>
    %second = ttmg.squad_dot_wait %first : tensor<16x16xf32>
"""
    context, module = _parse_module(tmp_path, body)
    canonical = _canonicalize(module, context)

    assert canonical.count("ttmg.squad_dot_wait") == 1, canonical
    assert canonical.count("%arg0") >= 2, canonical


def test_musa_sqmma_wait_canonicalizer_does_not_cross_operations(tmp_path):
    body = """    %first = ttmg.squad_dot_wait %arg0 : tensor<16x16xf32>
    %middle = arith.addf %arg0, %arg1 : tensor<16x16xf32>
    %second = ttmg.squad_dot_wait %first : tensor<16x16xf32>
"""
    context, module = _parse_module(tmp_path, body, "sqmma_waits_separated")
    canonical = _canonicalize(module, context)

    assert canonical.count("ttmg.squad_dot_wait") == 2, canonical
