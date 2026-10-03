"""The R0 DEEP SWEEP (testing MEMO build row A4; PR-smoke blocking).

Fourteen real upstream architecture families (plus the structural fixtures)
built from vendored real configs at fixed seeds, ZERO network, swept DEEP:
exact recipe classification, per-family facet floors with shape checks
against the model's own dims (never mask-shaped), payload identity against
independent same-process oracles, the absence-claim adjudicator against the
enumerated known-false manifest, and the flagship logit_lens failure pinned
enumerated-red. The single highest value-to-cost item in the memo (A4).

One explicit test function per model family: the smoke duration tripwire
budgets aggregate cost per test-function family, and each family's
build+trace cost must land in its own budget, not the first test's.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tests.real_model.r0.adjudication import adjudicate
from tests.real_model.r0.expectations import (
    FACET_SHAPE_SPECS,
    FALSE_CLAIM_OWNERS,
    GPT2_LN_INPUT_AVAILABLE_ADDRESSES,
    SHAPE_DIM_OVERRIDES,
    SHAPE_SPEC_OVERRIDES,
    WRONG_PAYLOAD_MANIFEST,
    WRONG_PAYLOAD_OWNER,
    load_expectations,
)
from tests.real_model.r0.families import STRUCTURAL_FIXTURES, build_structural

pytestmark = [pytest.mark.real_model, pytest.mark.real_class]

EXPECTATIONS = load_expectations()

MASK_SHAPE_HINT = (
    "a facet payload shaped like a mask or the token ids instead of an"
    " activation is the silent-wrong class the sweep exists to kill"
    " (memo D7 class 5)"
)

BOTH_IMPLS = pytest.mark.parametrize("impl", ["eager", "sdpa"])
EAGER_ONLY = pytest.mark.parametrize("impl", ["eager"])


def _module_by_address(trace, address):
    for module in trace.modules:
        if str(module.address) == address:
            return module
    raise AssertionError(f"module {address!r} not found in trace")


def _facet_value(view, facet):
    payload = view[facet]
    return getattr(payload, "value", payload)


def _expected_shape(family, address, facet, dims):
    letters = SHAPE_SPEC_OVERRIDES.get((family, address, facet), FACET_SHAPE_SPECS[facet])
    mapping = {
        "b": 1,
        "s": dims["seq"],
        "h": dims["hidden"],
        "n": dims["heads"],
        "g": dims.get("kv_heads", dims["heads"]),  # GQA KV heads; = heads for MHA
        "d": dims["d_head"],
        "v": dims["vocab"],
    }
    best = ""
    for prefix, overrides in SHAPE_DIM_OVERRIDES.get(family, ()):
        if address.startswith(prefix) and len(prefix) > len(best):
            best = prefix
            mapping = {**mapping, **overrides}
    return tuple(mapping[letter] for letter in letters)


def _deep_sweep(family, impl, r0_capture):
    """Every per-(family, impl) sweep assertion, in dependency order."""

    cap = r0_capture(family, impl)
    expected = EXPECTATIONS[family][impl]

    # 1. Capture honesty: the outcome settled COMPLETE.
    assert cap.trace.outcome.status.name == "COMPLETE"

    # 2. Exact recipe classification.
    measured: dict[str, int] = {}
    for row in cap.coverage.classified:
        for recipe in row.recipes:
            measured[recipe] = measured.get(recipe, 0) + 1
    assert measured == expected["recipes"], (
        f"{family}/{impl}: recipe classification drifted from the committed"
        f" expectation. measured={measured} expected={expected['recipes']}."
        " Fewer rows is a regression; more is a conscious golden update in the"
        " owning fix lane."
    )

    # 3. Facet floors with honest shapes (positive expectation ledger).
    floors = expected["floors"]
    dims = cap.spec.dims
    seq_mask_shapes = {
        (1, 1, dims["seq"], dims["seq"]),
        (1, dims["seq"], dims["seq"]),
    }
    for row in cap.coverage.classified:
        for recipe in row.recipes:
            floor = set(floors.get(recipe, ()))
            missing = floor - set(row.available)
            assert not missing, (
                f"{family}/{impl} {row.address} ({recipe}): floor facets"
                f" {sorted(missing)} are no longer available_now -- the"
                " per-family facet floor only grows (memo D2 positive"
                " expectation ledger)."
            )
            module = _module_by_address(cap.trace, row.address)
            view = module.facets
            for facet in sorted(floor):
                if facet not in FACET_SHAPE_SPECS:
                    continue
                value = _facet_value(view, facet)
                assert isinstance(value, torch.Tensor), (
                    f"{family}/{impl} {row.address}.{facet}: floor facet with a"
                    f" shape contract served a {type(value).__name__}, not a tensor"
                )
                manifest_key = (family, impl, row.address, facet)
                if manifest_key in WRONG_PAYLOAD_MANIFEST:
                    pinned_wrong = WRONG_PAYLOAD_MANIFEST[manifest_key]
                    assert tuple(value.shape) == pinned_wrong, (
                        f"STALE wrong-payload manifest row {manifest_key}: the"
                        f" payload is now {tuple(value.shape)}, no longer the"
                        f" recorded wrong shape {pinned_wrong}."
                        f" {WRONG_PAYLOAD_OWNER} fixed the anchor -- delete the"
                        " manifest row in that lane so the honest-shape floor"
                        " takes over."
                    )
                    continue
                expected_shape = _expected_shape(family, row.address, facet, dims)
                assert tuple(value.shape) == expected_shape, (
                    f"{family}/{impl} {row.address}.{facet}: shape"
                    f" {tuple(value.shape)} != expected {expected_shape} from"
                    f" the model's own config dims. {MASK_SHAPE_HINT}"
                )
                assert tuple(value.shape) not in seq_mask_shapes, (
                    f"{family}/{impl} {row.address}.{facet} is mask-shaped. {MASK_SHAPE_HINT}"
                )

    # 4. Absence-claim adjudication against the enumerated manifest.
    adjudicated = [[f.address, f.facet, f.rule] for f in adjudicate(cap.coverage, cap.model)]
    manifest = expected["false_claims"]
    new_claims = [row for row in adjudicated if row not in manifest]
    stale_rows = [row for row in manifest if row not in adjudicated]
    owners = sorted({FALSE_CLAIM_OWNERS.get(rule, rule) for _, _, rule in manifest})
    assert not new_claims, (
        f"{family}/{impl}: NEW false structurally_absent claim(s) {new_claims} --"
        " the module tree falsifies these claims and they are not in the"
        " enumerated known-false manifest (expectations_r0.json). A coverage-"
        "or-absence report whose silence is indistinguishable from coverage is"
        " the memo's #2 measured class; fix the recipe, never the adjudicator."
    )
    assert not stale_rows, (
        f"{family}/{impl}: STALE known-false manifest row(s) {stale_rows} -- the"
        f" claim no longer adjudicates false (owners: {owners}). Delete the row"
        " from expectations_r0.json in the fixing lane's own change."
    )

    # 5. Op-count golden, keyed on the resolved-config fingerprint.
    if hasattr(cap.model, "config"):
        from tests.real_model.registry import resolved_config_fingerprint

        assert expected.get("fingerprint"), (
            f"{family}/{impl}: the committed expectation carries no fingerprint;"
            " regenerate expectations_r0.json deliberately."
        )
        fingerprint = resolved_config_fingerprint(model=cap.model)
        if fingerprint != expected["fingerprint"]:
            pytest.fail(
                f"{family}/{impl}: resolved-config fingerprint drifted"
                f" ({fingerprint[:12]}... != recorded"
                f" {expected['fingerprint'][:12]}...). A standard flag changed"
                " the resolved config (the silent use_cache class, memo D7 #4);"
                " re-derive the op-count golden CONSCIOUSLY, never let it ride"
                " a stale key."
            )
        assert len(cap.trace.ops) == expected["n_ops"], (
            f"{family}/{impl}: op count {len(cap.trace.ops)} != golden"
            f" {expected['n_ops']} under an UNCHANGED resolved-config"
            " fingerprint -- the captured graph moved with no config cause."
        )


@pytest.mark.smoke_cells("test_deep_sweep_gpt2[sdpa]")
@BOTH_IMPLS
def test_deep_sweep_gpt2(impl, r0_capture):
    _deep_sweep("gpt2", impl, r0_capture)


@BOTH_IMPLS
def test_deep_sweep_distilgpt2(impl, r0_capture):
    _deep_sweep("distilgpt2", impl, r0_capture)


@pytest.mark.smoke_cells("test_deep_sweep_llama[sdpa]")
@BOTH_IMPLS
def test_deep_sweep_llama(impl, r0_capture):
    _deep_sweep("llama", impl, r0_capture)


@BOTH_IMPLS
def test_deep_sweep_qwen2(impl, r0_capture):
    _deep_sweep("qwen2", impl, r0_capture)


@pytest.mark.smoke_cells("test_deep_sweep_albert[sdpa]")
@BOTH_IMPLS
def test_deep_sweep_albert(impl, r0_capture):
    _deep_sweep("albert", impl, r0_capture)


@pytest.mark.smoke
@BOTH_IMPLS
def test_deep_sweep_distilbert(impl, r0_capture):
    _deep_sweep("distilbert", impl, r0_capture)


@pytest.mark.smoke_cells("test_deep_sweep_bert[sdpa]")
@BOTH_IMPLS
def test_deep_sweep_bert(impl, r0_capture):
    _deep_sweep("bert", impl, r0_capture)


@EAGER_ONLY
def test_deep_sweep_t5(impl, r0_capture):
    _deep_sweep("t5", impl, r0_capture)


@BOTH_IMPLS
def test_deep_sweep_vit(impl, r0_capture):
    _deep_sweep("vit", impl, r0_capture)


@pytest.mark.smoke_cells("test_deep_sweep_clip[sdpa]")
@BOTH_IMPLS
def test_deep_sweep_clip(impl, r0_capture):
    _deep_sweep("clip", impl, r0_capture)


@pytest.mark.smoke_cells("test_deep_sweep_whisper[sdpa]")
@BOTH_IMPLS
def test_deep_sweep_whisper(impl, r0_capture):
    _deep_sweep("whisper", impl, r0_capture)


@pytest.mark.smoke
@EAGER_ONLY
def test_deep_sweep_mamba(impl, r0_capture):
    _deep_sweep("mamba", impl, r0_capture)


@pytest.mark.smoke
@EAGER_ONLY
def test_deep_sweep_rwkv(impl, r0_capture):
    _deep_sweep("rwkv", impl, r0_capture)


@EAGER_ONLY
def test_deep_sweep_detection(impl, r0_capture):
    _deep_sweep("detection", impl, r0_capture)


@EAGER_ONLY
def test_deep_sweep_lstm_gru(impl, r0_capture):
    _deep_sweep("lstm_gru", impl, r0_capture)


def test_gpt2_payload_identity_independent_oracles(r0_capture):
    cap = r0_capture("gpt2", "sdpa")
    model, kwargs = cap.model, cap.input_kwargs
    # Independent oracle 1: the logits facet equals a direct same-process rerun.
    with torch.no_grad():
        direct = model(**kwargs).logits
    head = _module_by_address(cap.trace, "self")
    assert torch.equal(_facet_value(head.facets, "logits"), direct), (
        "lm_head logits facet != direct forward logits: the facet is anchored to"
        " the wrong op or the capture perturbed the computation"
    )
    # True embedding oracle (flipped by A02 when the resid_pre anchor was
    # fixed): block-0 resid_pre equals wte+wpe computed by hand -- bitwise,
    # since eval-mode embedding dropout is inert.
    ids = kwargs["input_ids"]
    with torch.no_grad():
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        embedded = model.transformer.wte(ids) + model.transformer.wpe(positions)
    block0 = _module_by_address(cap.trace, "transformer.h.0")
    resid_pre = _facet_value(block0.facets, "resid_pre")
    assert torch.equal(resid_pre, embedded), (
        "gpt2 resid_pre(block 0) != wte+wpe computed by hand: the residual"
        " anchor drifted off the embedding path"
    )
    # Aliasing identity: unembed_weight is the LIVE tied parameter, not a copy.
    unembed = _facet_value(head.facets, "unembed_weight")
    assert unembed is model.lm_head.weight, (
        "unembed_weight facet must alias the live parameter object"
    )
    assert model.lm_head.weight is model.transformer.wte.weight, (
        "GPT-2 ties lm_head to wte; the config-built fixture lost the tie"
    )


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_modern_lm_layer0_resid_pre_is_the_embedding(family, r0_capture):
    # Layer 0 serves the TRUE residual today (layer 1 is in the wrong-payload
    # manifest) -- this is the independent oracle that stays green and catches
    # any regression of the correct half.
    cap = r0_capture(family, "sdpa")
    ids = cap.input_kwargs["input_ids"]
    with torch.no_grad():
        embedded = cap.model.get_input_embeddings()(ids)
    block0 = _module_by_address(cap.trace, "model.layers.0")
    assert torch.equal(_facet_value(block0.facets, "resid_pre"), embedded), (
        f"{family}: resid_pre(layer 0) != the model's own input embeddings"
    )


def test_mamba_recipes_refuse_not_guess(r0_capture):
    cap = r0_capture("mamba", "eager")
    for row in cap.coverage.classified:
        assert not any("attention" in recipe for recipe in row.recipes), (
            f"attention recipe claimed on attention-free Mamba at {row.address}:"
            " recipes must refuse, not guess (memo 4.1)"
        )
        fabricated = {"q", "k", "v", "attn_out"} & set(row.available)
        assert not fabricated, (
            f"fabricated attention facets {sorted(fabricated)} on Mamba at {row.address}"
        )


def test_gpt2_ln_input_capture_dependency_pinned(r0_capture):
    # Memo D7 rank 7: needs_capture on ln_2.input while ln_1.input is available
    # on the SAME run. Pinned exactly so a fix (or a regression to
    # all-unavailable) surfaces here instead of shifting silently.
    cap = r0_capture("gpt2", "sdpa")
    serving = []
    for row in cap.coverage.classified:
        if "layer_norm" in row.recipes and "input" in row.available:
            serving.append(row.address)
    assert tuple(serving) == GPT2_LN_INPUT_AVAILABLE_ADDRESSES, (
        f"LayerNorm `input` availability moved: now served at {serving},"
        f" pinned {GPT2_LN_INPUT_AVAILABLE_ADDRESSES}. If a fix made"
        " availability consistent, update the pin consciously in the fixing"
        " lane (memo D7 class 7)."
    )


def test_logit_lens_true_oracle_final_layer_equals_model_logits(r0_capture):
    """logit_lens succeeds on the config-built GPT-2 (enumerated-red flipped).

    The flagship logit-lens failure's root cause was the false final_norm_*
    absence claims; A02's final-norm dataflow anchor (mikit F6) fixed it, so
    this test now asserts the TRUE oracle: the final layer's lens projection
    reproduces the model's own logits bitwise (same-process fp32 CPU).
    """

    import torchlens as tl

    cap = r0_capture("gpt2", "sdpa")
    result = tl.semantic.logit_lens(cap.trace)
    assert result.entries, "logit_lens returned no per-layer projections"
    with torch.no_grad():
        direct = cap.model(**cap.input_kwargs).logits
    assert torch.equal(result.entries[-1].logits, direct), (
        "final-layer logit-lens projection != the model's own logits: the"
        " reconstructed head (final norm + unembedding) drifted off the model"
    )


@pytest.mark.parametrize("fixture_name", STRUCTURAL_FIXTURES)
def test_structural_fixture_traces_faithfully(fixture_name):
    import torchlens as tl

    model, args, kwargs = build_structural(fixture_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            direct = model(*args, **kwargs)
        if fixture_name == "train-bn":
            # Reset stats so capture-time mutation is measured from a known
            # state, and prove the buffers MOVE during capture.
            model.bn.reset_running_stats()
            baseline = model.bn.running_mean.clone()
        trace = tl.trace(model, args, kwargs)
    assert trace.outcome.status.name == "COMPLETE"
    if fixture_name == "container-output":
        leaves = [op.out for op in trace.output_ops]
        for name, expected in [
            ("main", direct["main"]),
            ("parts[0]", direct["parts"][0]),
            ("parts[1]", direct["parts"][1]),
            ("pair[0]", direct["pair"][0]),
            ("pair[1]", direct["pair"][1]),
        ]:
            assert any(
                leaf.shape == expected.shape and torch.equal(leaf, expected) for leaf in leaves
            ), f"container-output leaf {name} not transported exactly"
    elif fixture_name == "train-bn":
        assert not torch.equal(model.bn.running_mean, baseline), (
            "train-mode BatchNorm running stats did not move during capture --"
            " the capture did not run the real train-mode path"
        )
    elif fixture_name == "tied-weight":
        assert model.unembed.weight is model.emb.weight
        assert torch.equal(trace.output_ops[0].out, direct)
    else:
        direct_leaf = direct if isinstance(direct, torch.Tensor) else direct[0]
        assert torch.equal(trace.output_ops[0].out, direct_leaf), (
            f"{fixture_name}: traced output != direct output"
        )
