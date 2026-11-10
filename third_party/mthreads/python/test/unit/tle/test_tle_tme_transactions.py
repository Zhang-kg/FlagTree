"""Compile-only coverage for mthreads TLE TME completion transactions."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir, passes
from triton.backends.compiler import Language
from triton.compiler import ASTSource

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _completion_transaction_kernel(
    desc,
    dynamic_k,
    STAGES: tl.constexpr,
    SLOT: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(STAGES, expect_bytes=32768)
    tle.gpu.copy(
        desc,
        smem.slot(SLOT),
        (256, 64),
        (0, dynamic_k),
        barrier=full[SLOT],
    )


def _compile_transaction_module(stages=1, slot=0):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_completion_transaction_kernel,
        signature={
            "desc": "tensordesc<fp16[256, 64]>",
            "dynamic_k": "i32",
            "STAGES": "constexpr",
            "SLOT": "constexpr",
        },
        constexprs={"STAGES": stages, "SLOT": slot},
    )
    module = src.make_ir(
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
    return backend, context, module


def _lower_to_llvm_dialect(module, context):
    pm = ir.pass_manager(context)
    passes.convert.add_scf_to_cf(pm)
    passes.convert.add_index_to_llvmir(pm)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(module, "lower_tle_tme_transaction_to_llvm")
    return module.str_nodebug()


@pytest.mark.parametrize("stages,slot", [(1, 0), (2, 0), (2, 1)])
def test_mthreads_tle_completion_transaction_is_single_and_idempotent(stages, slot):
    _, context, module = _compile_transaction_module(stages, slot)
    ttgir = module.str_nodebug()

    assert ttgir.count("ttmg.barrier_add_trans") == 1, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 1, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 1, ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 3, ttgir
    assert ttgir.count("musa.tme.issue_thread = 0 : i32") == 3, ttgir
    assert "musa_tle.expect_bytes" not in ttgir, ttgir
    assert "ttmg.wait_barrier" not in ttgir, ttgir

    add_pos = ttgir.index("ttmg.barrier_add_trans")
    copy_pos = ttgir.index("ttmg.async_tme_copy_global_to_local")
    arrive_pos = ttgir.index("ttmg.arrive_barrier_noret")
    # Match MUTLASS MP31's arrive_and_expect_tx producer protocol: register
    # expected bytes and arrival before issuing the asynchronous TME load.
    assert add_pos < arrive_pos < copy_pos, ttgir
    add_barrier, byte_value = re.search(r"ttmg\.barrier_add_trans\s+(%[-\w.]+),\s*(%[-\w.]+)", ttgir).groups()
    copy_barrier = re.search(r"ttmg\.async_tme_copy_global_to_local\s+[^\n]*\],\s*(%[-\w.]+)", ttgir).group(1)
    arrive_barrier = re.search(r"ttmg\.arrive_barrier_noret\s+(%[-\w.]+)", ttgir).group(1)
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ttgir)
    }
    assert add_barrier == copy_barrier == arrive_barrier, ttgir
    assert constants[byte_value] == 32768, ttgir

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "rerun_tle_tme_transaction_lowering")
    rerun = module.str_nodebug()
    assert rerun.count("ttmg.barrier_add_trans") == 1, rerun
    assert rerun.count("ttmg.arrive_barrier_noret") == 1, rerun


def test_mthreads_tle_explicit_completion_uses_raw_issue_thread_without_barrier0(tmp_path):
    backend, _, module = _compile_transaction_module()
    fixture = module.str_nodebug().replace(
        "musa.tme.issue_thread = 0 : i32",
        "musa.tme.issue_thread = 512 : i32",
    )
    fixture_path = tmp_path / "raw_issue_thread_512.ttgir"
    fixture_path.write_text(fixture)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    reparsed = ir.parse_mlir_module(str(fixture_path), context)
    llir = _lower_to_llvm_dialect(reparsed, context)

    assert "llvm.musa.barrier0" not in llir, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.async.add.trans"') == 1, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.tme.ld.tile.2d"') == 1, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.async.arrive"') == 1, llir
    assert "llvm.musa.async.arrive.none.phaseid" not in llir, llir
    issue_constant = re.search(r"(%\d+) = llvm\.mlir\.constant\(512 : i32\) : i32", llir).group(1)
    assert len(re.findall(rf'llvm\.icmp "eq" %\d+, {re.escape(issue_constant)} : i32', llir)) == 3, llir
    assert (llir.index("llvm.musa.async.add.trans") < llir.index("llvm.musa.async.arrive") <
            llir.index("llvm.musa.tme.ld.tile.2d")), llir


def test_mthreads_tle_completion_copy_uses_default_issue_thread_zero(tmp_path):
    fixture = """#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32,
    ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @copy_in_consumer(%desc: !tt.tensordesc<tensor<256x64xf16, #shared>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24>}
    default {
      ttmg.async_tme_copy_global_to_local %desc[%c0, %c0], %c1, %smem, %true {
        blockShape = array<i32: 256, 64>, cachePolicy = 0 : i32,
        innerPersistence = 2 : i32, musa_tle.expect_bytes = 32768 : i32,
        outerPersistence = 2 : i32, prefetchSize = 0 : i32,
        swizzleGranularity = 0 : i32, swizzleLine = 1 : i32,
        swizzleStride = 3 : i32
      } : !tt.tensordesc<tensor<256x64xf16, #shared>>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
"""
    fixture_path = tmp_path / "completion_copy_in_default.ttgir"
    fixture_path.write_text(fixture)

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "lower_completion_copy_in_default")
    ttgir = module.str_nodebug()
    assert ttgir.count("ttmg.barrier_add_trans") == 1, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 1, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 1, ttgir
    assert ttgir.count("musa.tme.issue_thread = 0 : i32") == 3, ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 3, ttgir
    assert "musa_tle.expect_bytes" not in ttgir, ttgir


def _four_role_completion_fixture(with_start_ids, group_writer=False,
                                  secondary_accounting="musa_tle.skip_transaction_accounting"):
    desc_type = "!tt.tensordesc<tensor<256x64xf16, #shared>>"
    smem_type = "!ttg.memdesc<256x64xf16, #shared, #smem, mutable>"

    def copy_body(desc, smem, prefix, accounting="musa_tle.expect_bytes = 32768 : i32", emit_constants=True):
        constants = f"""
      %{prefix}_zero = arith.constant 0 : i32
      %{prefix}_bar = arith.constant 1 : i32
      %{prefix}_pred = arith.constant true
""" if emit_constants else ""
        return constants + f"""
      ttmg.async_tme_copy_global_to_local {desc}[%{prefix}_zero, %{prefix}_zero], %{prefix}_bar, {smem}, %{prefix}_pred {{
        blockShape = array<i32: 256, 64>, cachePolicy = 0 : i32,
        innerPersistence = 2 : i32, {accounting},
        outerPersistence = 2 : i32, prefetchSize = 0 : i32,
        swizzleGranularity = 0 : i32, swizzleLine = 1 : i32,
        swizzleStride = 3 : i32
      }} : {desc_type}, {smem_type}
"""

    default_body = "" if group_writer else copy_body("%desc", "%smem", "default")
    if group_writer:
        asfa_body = copy_body("%asfa_desc", "%asfa_smem", "asfa", "musa_tle.expect_bytes = 65536 : i32")
        asfa_body += copy_body("%asfa_desc", "%asfa_smem", "asfa", secondary_accounting, emit_constants=False)
    else:
        asfa_body = copy_body("%asfa_desc", "%asfa_smem", "asfa")
    start_ids = ", warpGroupStartIds = array<i32: 4, 6, 7>" if with_start_ids else ""
    return f"""#shared = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @four_role_completion(%desc: {desc_type}) {{
    %smem = ttg.local_alloc : () -> {smem_type}
    ttg.warp_specialize(%desc, %smem) attributes {{requestedRegisters = array<i32: 24, 24, 24>{start_ids}}}
    default {{
      {default_body}
      ttg.warp_yield
    }}
    partition0(%dispatch_desc: {desc_type}, %dispatch_smem: {smem_type}) num_warps(2) {{
      ttg.warp_return
    }}
    partition1(%asfa_desc: {desc_type}, %asfa_smem: {smem_type}) num_warps(1) {{
      {asfa_body}
      ttg.warp_return
    }}
    partition2(%b_desc: {desc_type}, %b_smem: {smem_type}) num_warps(1) {{
      {copy_body("%b_desc", "%b_smem", "b")}
      ttg.warp_return
    }} : ({desc_type}, {smem_type}) -> ()
    tt.return
  }}
}}
"""


@pytest.mark.parametrize("with_start_ids", [False, True])
def test_mthreads_tle_completion_copy_uses_exact_four_role_issue_threads(tmp_path, with_start_ids):
    # Exercise both pre-allocation prefix sums and explicit warp-group starts.
    # The dispatch role does not issue TME; A+SFA and B are led by tids 192/224.
    fixture_path = tmp_path / "four_role_completion.ttgir"
    fixture_path.write_text(_four_role_completion_fixture(with_start_ids))
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "lower_exact_four_role_completion")
    ttgir = module.str_nodebug()

    for issue_thread in (0, 192, 224):
        assert ttgir.count(f"musa.tme.issue_thread = {issue_thread} : i32") == 3, ttgir
    for role, terminator, issue_thread in (
        ("default {", "ttg.warp_yield", 0),
        ("partition1(", "ttg.warp_return", 192),
        ("partition2(", "ttg.warp_return", 224),
    ):
        body = ttgir.split(role, maxsplit=1)[1].split(terminator, maxsplit=1)[0]
        assert body.count(f"musa.tme.issue_thread = {issue_thread} : i32") == 3, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 3, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 3, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 3, ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 9, ttgir
    assert "musa_tle.expect_bytes" not in ttgir, ttgir

    # Rerunning the pass must not duplicate any of the three transactions.
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "rerun_exact_four_role_completion")
    rerun = module.str_nodebug()
    assert rerun.count("ttmg.barrier_add_trans") == 3, rerun
    assert rerun.count("ttmg.arrive_barrier_noret") == 3, rerun


@pytest.mark.parametrize("with_start_ids", [False, True])
def test_mthreads_tle_grouped_writer_accounts_once_and_marks_every_copy(tmp_path, with_start_ids):
    fixture_path = tmp_path / "grouped_writer_completion.ttgir"
    fixture_path.write_text(_four_role_completion_fixture(with_start_ids, group_writer=True))
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "lower_grouped_writer_completion")
    ttgir = module.str_nodebug()

    # The A+SFA writer contributes one summed transaction and one arrival,
    # while both copies retain its physical issue thread and completion ID.
    assert ttgir.count("ttmg.barrier_add_trans") == 2, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 2, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 3, ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 7, ttgir
    assert "musa_tle.expect_bytes" not in ttgir, ttgir
    assert "musa_tle.skip_transaction_accounting" not in ttgir, ttgir
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ttgir)
    }
    for role, issue_thread, num_copies, byte_count in (
        ("partition1(", 192, 2, 65536),
        ("partition2(", 224, 1, 32768),
    ):
        body = ttgir.split(role, maxsplit=1)[1].split("ttg.warp_return", maxsplit=1)[0]
        assert body.count(f"musa.tme.issue_thread = {issue_thread} : i32") == num_copies + 2, body
        op_names = re.findall(
            r"ttmg\.(barrier_add_trans|arrive_barrier_noret|async_tme_copy_global_to_local)\b", body
        )
        assert op_names == ["barrier_add_trans", "arrive_barrier_noret"] + [
            "async_tme_copy_global_to_local"
        ] * num_copies, body
        add_barrier, byte_value = re.search(
            r"ttmg\.barrier_add_trans\s+(%[-\w.]+),\s*(%[-\w.]+)", body
        ).groups()
        assert constants[byte_value] == byte_count, body
        arrive_barrier = re.search(r"ttmg\.arrive_barrier_noret\s+(%[-\w.]+)", body).group(1)
        copy_barriers = re.findall(
            r"ttmg\.async_tme_copy_global_to_local\s+[^\n]*\],\s*(%[-\w.]+)", body
        )
        assert copy_barriers == [add_barrier] * num_copies, body
        assert arrive_barrier == add_barrier, body

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "rerun_grouped_writer_completion")
    assert module.str_nodebug() == ttgir


@pytest.mark.parametrize(
    "secondary_accounting,diagnostic",
    [
        ("musa_tle.skip_transaction_accounting = 1 : i32", "skip_transaction_accounting must be a unit attribute"),
        ("musa_tle.skip_transaction_accounting, musa_tle.expect_bytes = 32768 : i32",
         "expect_bytes and skip_transaction_accounting are mutually exclusive"),
    ],
)
def test_mthreads_tle_grouped_writer_rejects_invalid_accounting_attributes(tmp_path, capfd,
                                                                         secondary_accounting, diagnostic):
    fixture_path = tmp_path / "invalid_grouped_writer_completion.ttgir"
    fixture_path.write_text(_four_role_completion_fixture(True, group_writer=True,
                                                         secondary_accounting=secondary_accounting))
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "reject_invalid_grouped_writer_accounting")
    assert diagnostic in capfd.readouterr().err


def _ttg_grouped_writer_fixture(secondary_accounting="musa_tle.skip_transaction_accounting",
                                target="musa:ph1", secondary_barrier=True, secondary_store=False):
    desc_type = "!tt.tensordesc<tensor<256x64xf16, #shared>>"
    smem_type = "!ttg.memdesc<256x64xf16, #shared, #smem, mutable>"
    target_attr = f'ttg.target = "{target}", ' if target is not None else ""
    barrier = " barrier %c1" if secondary_barrier else ""
    attrs = f" {{{secondary_accounting}}}" if secondary_accounting else ""
    operands = "%second, %desc" if secondary_store else "%desc, %second"
    types = f"{smem_type}, {desc_type}" if secondary_store else f"{desc_type}, {smem_type}"
    return f"""#shared = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    {target_attr}"ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @ttg_grouped_writer(%desc: {desc_type}) {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %first = ttg.local_alloc : () -> {smem_type}
    %second = ttg.local_alloc : () -> {smem_type}
    ttg.tma_copy %desc, %first, [%c0, %c0] barrier %c1 {{expect_bytes = 65536 : i32}}
        : {desc_type}, {smem_type}
    ttg.tma_copy {operands}, [%c0, %c0]{barrier}{attrs} : {types}
    tt.return
  }}
}}
"""


def _parse_ttg_completion_fixture(tmp_path, fixture):
    fixture_path = tmp_path / "ttg_grouped_writer.ttgir"
    fixture_path.write_text(fixture)
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    return context, ir.parse_mlir_module(str(fixture_path), context)


def test_mthreads_tle_ttg_grouped_copy_verifies_and_preserves_accounting_protocol(tmp_path):
    # Cover the real ttg.tma_copy verifier, not only the later ttmg operation.
    context, module = _parse_ttg_completion_fixture(tmp_path, _ttg_grouped_writer_fixture())
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tme_lowering(pm)
    pm.run(module, "lower_verified_ttg_grouped_writer")
    ttgir = module.str_nodebug()
    assert ttgir.count("musa_tle.expect_bytes = 65536 : i32") == 1, ttgir
    assert ttgir.count("musa_tle.skip_transaction_accounting") == 1, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 2, ttgir

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "account_verified_ttg_grouped_writer")
    ttgir = module.str_nodebug()
    assert ttgir.count("ttmg.barrier_add_trans") == 1, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 1, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 2, ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 4, ttgir
    assert ttgir.count("musa.tme.issue_thread = 0 : i32") == 4, ttgir
    assert "musa_tle.skip_transaction_accounting" not in ttgir, ttgir
    assert "musa_tle.expect_bytes" not in ttgir, ttgir


@pytest.mark.parametrize("target", ["musa:ph1", "cuda:90"])
def test_mthreads_tle_ttg_verifier_preserves_explicit_positive_expect_bytes(tmp_path, target):
    # No skip marker: the existing explicit-copy contract remains accepted,
    # including when NVIDIA-target IR is inspected using this backend fork.
    _, module = _parse_ttg_completion_fixture(
        tmp_path, _ttg_grouped_writer_fixture("expect_bytes = 32768 : i32", target=target)
    )
    assert module.str_nodebug().count("ttg.tma_copy") == 2


@pytest.mark.parametrize(
    "overrides,diagnostic",
    [
        ({"secondary_accounting": "musa_tle.skip_transaction_accounting, expect_bytes = 32768 : i32"},
         "expect_bytes and skip_transaction_accounting are mutually exclusive"),
        ({"secondary_accounting": "musa_tle.skip_transaction_accounting = 1 : i32"},
         "skip_transaction_accounting must be a unit attribute"),
        ({"secondary_barrier": False},
         "skip_transaction_accounting requires a global-to-local copy with an explicit completion barrier"),
        ({"secondary_store": True},
         "skip_transaction_accounting requires a global-to-local copy with an explicit completion barrier"),
        ({"target": "cuda:90"}, "skip_transaction_accounting is only supported for a MUSA target"),
        ({"target": None}, "skip_transaction_accounting is only supported for a MUSA target"),
        ({"secondary_accounting": ""}, "completion barrier requires a positive expect_bytes attribute"),
        ({"secondary_accounting": "", "target": "cuda:90"},
         "completion barrier requires a positive expect_bytes attribute"),
        ({"secondary_accounting": "expect_bytes = 0 : i32"},
         "completion barrier requires a positive expect_bytes attribute"),
        ({"secondary_accounting": "expect_bytes = 32768 : i64"},
         "completion barrier requires a positive expect_bytes attribute"),
        ({"secondary_accounting": "expect_bytes = 32768 : i32", "secondary_barrier": False},
         "expect_bytes requires a completion barrier operand"),
    ],
)
def test_mthreads_tle_ttg_verifier_rejects_invalid_grouped_completion(tmp_path, capfd, overrides, diagnostic):
    with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
        _parse_ttg_completion_fixture(tmp_path, _ttg_grouped_writer_fixture(**overrides))
    assert diagnostic in capfd.readouterr().err
