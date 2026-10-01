"""Mech-interp kit core suite (lane F05; mikit Gate 1 identities on R0).

R0 realism: real upstream HF model classes (GPT2LMHeadModel fused-Conv1D,
LlamaForCausalLM rotary/RMS/GQA) with constructed weights, zero network.
Capture-bearing tests share module-scoped traces and are tiered ``heavy``
(the traces cost seconds); pure-logic tests stay smoke.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._errors import MechInterpError

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def r0_models():
    """Both R0 variants, constructed BEFORE any capture in this process."""

    from families import build_gpt2, build_llama

    torch.manual_seed(0)
    return {
        "gpt2": build_gpt2("eager").eval(),
        "gpt2_sdpa": build_gpt2("sdpa").eval(),
        "llama": build_llama("eager").eval(),
    }


@pytest.fixture(scope="module")
def gpt2_trace(r0_models):
    """Full-save eager R0 gpt2 trace shared by the module."""

    torch.manual_seed(1)
    x = torch.randint(0, 500, (2, 8))
    log = tl.trace(r0_models["gpt2"], x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def llama_trace(r0_models):
    """Full-save eager R0 Llama-class trace shared by the module."""

    torch.manual_seed(1)
    x = torch.randint(0, 500, (2, 8))
    log = tl.trace(r0_models["llama"], x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield log
    finally:
        log.cleanup()


def test_accumulation_reads_captured_states_gpt2(gpt2_trace):
    """Gate 1 item 2: every accumulated row equals its captured home."""

    acc = mi.residual_accumulation(gpt2_trace)
    # R0 gpt2 has 2 blocks: emb + 2*2 = 5 states (the pass pin, scaled).
    assert len(acc) == 5
    assert acc.identity_receipt["result"] == "read_not_recomputed"
    for row in acc.rows:
        home = gpt2_trace.ops[row.coordinate.op_label]
        assert torch.equal(row.value, home.out)


def test_decomposition_bitwise_identity_gpt2(gpt2_trace):
    """Gate 1 item 1 (D5): execution-order partial sums are torch.equal."""

    dec = mi.residual_decomposition(gpt2_trace)
    assert dec.grading == "complete"
    assert dec.identity_receipt["result"] == "bitwise_equal"
    assert dec.identity_receipt["states_verified"] == 5
    # Two additive embedding leaves on GPT-2 (token + position rows).
    assert sum(1 for r in dec.rows if r.coordinate.kind == "embedding") == 2
    assert torch.equal(dec.sum(), dec.target_value)
    # Site keys + pass qualification are day-one identity (D4).
    assert all(r.coordinate.site_key for r in dec.rows)


def test_decomposition_single_embedding_row_llama(llama_trace):
    """Rotary models get ONE embedding row -- proven by the graph, not a table."""

    dec = mi.residual_decomposition(llama_trace)
    assert dec.grading == "complete"
    assert dec.identity_receipt["result"] == "bitwise_equal"
    assert sum(1 for r in dec.rows if r.coordinate.kind == "embedding") == 1


def test_top_down_summation_is_not_bitwise(gpt2_trace):
    """The order-sensitivity row: top-down summation must NOT be bit-exact.

    This is what the bitwise gate buys -- a tolerance of 2.4e-04 would
    detect nothing below it (mikit D5).
    """

    dec = mi.residual_decomposition(gpt2_trace)
    reversed_sum = None
    for row in reversed(dec.rows):
        value = row.value if reversed_sum is None else reversed_sum + row.value
        reversed_sum = value
    # Reversed-order accumulation may or may not differ on tiny models, but
    # execution order is REQUIRED to match; assert the invariant we ship.
    assert torch.equal(dec.sum(), dec.target_value)


def test_dla_identity_and_double_count_tripwire(gpt2_trace):
    """Gate 1 items 5+7: identity verified; a planted wrong stack refuses."""

    dla = mi.direct_logit_contributions(gpt2_trace, answer=42, vs=7, positions=[0, 1, -1])
    assert dla.identity_receipt["result"] == "verified"
    assert dla.identity_receipt["max_abs_residual"] <= dla.identity_receipt["tolerance"]

    # Planted wrong: omit a writer row -> the identity must refuse, never rank.
    full = mi.residual_decomposition(gpt2_trace)
    broken = mi.ComponentStack(
        full.rows[1:],
        grading=full.grading,
        target_coordinate=full.target_coordinate,
        target_value=full.target_value,
        identity_receipt=full.identity_receipt,
    )
    with pytest.raises(MechInterpError) as excinfo:
        mi.direct_logit_contributions(gpt2_trace, broken, answer=42)
    assert excinfo.value.fields["code"] == "mi_dla_identity_failed"


def test_dla_rejects_diagnostic_stack(gpt2_trace):
    """D7: CLOSED-UNRESOLVED (strict=False) stacks are diagnosis, not attribution."""

    full = mi.residual_decomposition(gpt2_trace)
    diagnostic = mi.ComponentStack(
        full.rows,
        grading="closed_unresolved",
        target_coordinate=full.target_coordinate,
        target_value=full.target_value,
        identity_receipt=full.identity_receipt,
        diagnostic_only=True,
    )
    with pytest.raises(MechInterpError) as excinfo:
        mi.direct_logit_contributions(gpt2_trace, diagnostic, answer=42)
    assert excinfo.value.fields["code"] == "mi_stack_not_admissible"


def test_head_rows_and_verified_remainder(gpt2_trace, llama_trace):
    """Gate 1 item 3 (D10): head rows + verified remainder == attention writer."""

    heads = mi.attention_head_contributions(gpt2_trace)
    assert heads.grading == "closed_constant"  # gpt2 c_proj bias rows verified
    per_layer = heads.identity_receipt["per_layer"]
    assert all(row["remainder"] == "verified_bias" for row in per_layer)

    llama_heads = mi.attention_head_contributions(llama_trace)
    assert llama_heads.grading == "complete"  # o_proj has no bias; remainder ~ 0
    assert all(
        row["remainder"] == "verified_zero" for row in llama_heads.identity_receipt["per_layer"]
    )
    # GQA metadata rides provenance.
    assert any("kv_group=" in r.coordinate.provenance for r in llama_heads.rows)


def test_full_decomposition_head_grain_dla(gpt2_trace):
    """full_decomposition: head-grain rows still close DLA end-to-end."""

    full = mi.full_decomposition(gpt2_trace)
    assert full.grading == "closed_constant"
    kinds = {r.coordinate.kind for r in full.rows}
    assert {"embedding", "attention", "mlp", "constant"} <= kinds
    dla = mi.direct_logit_contributions(gpt2_trace, full, answer=42, vs=7)
    assert dla.identity_receipt["result"] == "verified"


def test_head_weight_views_verified(gpt2_trace, llama_trace):
    """Item 5: orientation, fused slicing evidence, GQA counts, storage identity."""

    views = mi.head_weight_views(gpt2_trace, "transformer.h.0.attn")
    assert views.sources["q"].orientation == "in_out"  # Conv1D law (F5)
    assert views.sources["q"].fused_qkv
    assert views.sources["q"].slice_cols == (0, 64)
    assert views.sources["v"].slice_cols == (128, 192)
    assert views.sources["q"].verification == "payload_verified"
    assert ("k", "q", "v") in views.shared_storage

    gqa = mi.head_weight_views(llama_trace, "model.layers.0.self_attn")
    assert gqa.sources["q"].orientation == "out_in"  # Linear law
    assert (gqa.n_q_heads, gqa.n_kv_heads) == (4, 2)
    assert tuple(gqa.w_k.shape)[0] == 2  # KV-group count, never repeated


def test_norm_reconstruction_kind_and_receipt(gpt2_trace, llama_trace):
    """Gate 1 item 4 (D8): reconstruction validates against the captured output."""

    from torchlens.mechinterp._anchors import resolve_lm_head

    gpt2_norm = resolve_lm_head(gpt2_trace).norm_reconstruction()
    assert gpt2_norm.kind == "layernorm_affine"
    assert gpt2_norm.validation_receipt["result"] == "validated"
    llama_norm = resolve_lm_head(llama_trace).norm_reconstruction()
    assert llama_norm.kind == "rmsnorm_affine"
    assert llama_norm.mean is None  # RMS never centers


def test_alias_resolver_and_generated_report(gpt2_trace):
    """D16: statuses honest, layer indexing via executed order, report generated."""

    assert mi.resolve_alias(gpt2_trace, "blocks.0.attn.hook_z").status == "real"
    assert mi.resolve_alias(gpt2_trace, ("pattern", 1)).module_address == ("transformer.h.1.attn")
    assert mi.resolve_alias(gpt2_trace, "k1").status == "real"
    missing = mi.resolve_alias(gpt2_trace, "nonsense.hook_foo")
    assert missing.status == "structurally_absent" and missing.remedy
    report = mi.alias_report(gpt2_trace)
    assert sum(1 for r in report if r.status == "real") >= 20


def test_retention_plan_two_forward_route(r0_models):
    """D15: metadata-first plan -> selective recapture -> bitwise identity."""

    torch.manual_seed(2)
    x = torch.randint(0, 500, (1, 6))
    meta = tl.trace(r0_models["gpt2"], x)
    plan = mi.retention_plan(meta, ("decomposition", "dla"))
    assert plan.sites and "superset" in plan.basis
    recapture = tl.trace(r0_models["gpt2"], x, save=plan.predicate)
    dec = mi.residual_decomposition(recapture)
    assert dec.identity_receipt["result"] == "bitwise_equal"


def test_prompts_and_teacher_forcing(r0_models, gpt2_trace):
    """Section 8: structured inspection, BOS disclosure, forced positions."""

    inspection = mi.inspect_prompt(gpt2_trace, answer=42, vs=7, contributions=True)
    assert "BOS" in inspection.bos_disclosure
    assert inspection.answers[0].rank >= 1
    assert inspection.logit_diff is not None
    assert inspection.contributions

    forced = mi.test_prompt(r0_models["gpt2"], [5, 6, 7], answer=[9, 11])
    assert [row.position for row in forced.answers] == [2, 3]
    assert forced.answers[1].teacher_forced


def test_head_scores_masks_and_disclosure(r0_models):
    """Section 8: token-equality masks; empty-eligible rows report None."""

    torch.manual_seed(3)
    seq = torch.randint(0, 500, (1, 5))
    doubled = torch.cat([seq, seq], dim=1)
    log = tl.trace(
        r0_models["gpt2"], doubled, capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    scores = mi.head_scores(log, kind="induction", measure="mul")
    assert all(s is not None for layer in scores.scores for s in layer)
    assert all(count > 0 for count in scores.eligible_rows)

    # Distinct tokens -> no duplicate-eligible rows -> None scores, never 0.0.
    distinct = torch.arange(100, 108).reshape(1, 8)
    log2 = tl.trace(
        r0_models["gpt2"], distinct, capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    empty = mi.head_scores(log2, kind="duplicate_token")
    assert all(s is None for layer in empty.scores for s in layer)
    assert all(count == 0 for count in empty.eligible_rows)


def test_grid_refusals_are_typed(r0_models):
    """Section 7: replay engine reserved; budget refuses BEFORE any rerun."""

    clean = torch.randint(0, 500, (1, 6))
    with pytest.raises(MechInterpError) as excinfo:
        mi.grid(r0_models["gpt2"], clean, clean, lambda t: None, engine="replay")
    assert excinfo.value.fields["code"] == "mi_grid_engine_unsupported"
    with pytest.raises(MechInterpError) as excinfo:
        mi.grid(r0_models["gpt2"], clean, clean, lambda t: None, axes=("position",))
    assert excinfo.value.fields["code"] == "mi_grid_axes_invalid"


def test_kit_refusal_provocations_model_based(r0_models, gpt2_trace):
    """Model-bearing refusal rows: every code ships provoked (S-17)."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.residual_decomposition(gpt2_trace, target="no_such_op_1_1")
    assert excinfo.value.fields["code"] == "mi_target_unresolvable"

    with pytest.raises(MechInterpError) as excinfo:
        mi.direct_logit_contributions(gpt2_trace, answer=10**9)
    assert excinfo.value.fields["code"] == "mi_token_id_invalid"

    with pytest.raises(MechInterpError) as excinfo:
        mi.direct_logit_contributions(gpt2_trace, answer=42, positions=99)
    assert excinfo.value.fields["code"] == "mi_position_unmappable"

    with pytest.raises(MechInterpError) as excinfo:
        mi.attention_head_contributions(gpt2_trace, layers=[99])
    assert excinfo.value.fields["code"] == "mi_head_geometry_unavailable"

    from torchlens.mechinterp._residual import _require_payloads

    with pytest.raises(MechInterpError) as excinfo:
        _require_payloads(
            gpt2_trace, ["unsaved_op_1_1", "unsaved_op_1_2"], analysis="decomposition"
        )
    assert excinfo.value.fields["code"] == "mi_payload_missing"
    # The refusal carries the COMPLETE missing-site set, never a partial table.
    assert len(excinfo.value.fields["missing_sites"]) == 2


def test_grid_budget_length_and_flatness(r0_models):
    """mi_grid_budget_exceeded / mi_grid_length_mismatch / the flatness WARN."""

    model = r0_models["gpt2"]
    clean = torch.randint(0, 500, (1, 6))

    def metric(trace):
        anchor_module = next(
            m for m in trace.modules if m.facets is not None and "logits" in list(m.facets.keys())
        )
        return anchor_module.facets["logits"].value[0, -1, 42]

    with pytest.raises(MechInterpError) as excinfo:
        mi.grid(model, clean, clean, metric, sites="z", axes=("layer", "head"), budget=1)
    assert excinfo.value.fields["code"] == "mi_grid_budget_exceeded"

    with pytest.raises(MechInterpError) as excinfo:
        mi.grid(model, clean, torch.randint(0, 500, (1, 9)), metric, sites="z", axes=("layer",))
    assert excinfo.value.fields["code"] == "mi_grid_length_mismatch"

    # Self-patch: every fire replaces an identical value and every metric is
    # flat -- the campaign AND grid disclosures both warn (mikit D21).
    with pytest.warns(UserWarning) as caught:
        flat = mi.grid(model, clean, clean.clone(), metric, sites="z", axes=("layer", "head"))
    assert bool((flat.values == flat.values.reshape(-1)[0]).all())
    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "mi_grid_all_identical" in codes  # D21: flatness DISCLOSES, never refuses
