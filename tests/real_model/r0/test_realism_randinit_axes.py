"""R0 axes rows (testing MEMO build row A5).

The four measured cross-cutting axes at zero network: the eager/sdpa FACET
DIFF asserted explicitly (D7 class 1, the top measured class), kwargs
identity + the static shadowing intersection check (D7 class 3), the
pass-qualification pin on distilgpt2's 2-pass residual adds (D7 class 8,
OPUS Q5), and resolved-config-fingerprint rows for the silent config-flag
class (D7 class 4). Plus the enumerated-red pins for the container-output
replay crash (FIX-A) and the kwargs-only plain-module transport defect.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from tests.real_model.r0.expectations import (
    EAGER_SDPA_FACET_DIFF,
    EXPECTED_TRACE_PARAM_COLLISIONS,
    FORWARD_KWARG_CORPUS,
    KNOWN_RED_BY_ID,
    SDPA_UNSUPPORTED_FAMILIES,
    load_expectations,
)
from tests.real_model.r0.families import FAMILY_BY_NAME, SEED, VOCAB

pytestmark = [pytest.mark.smoke, pytest.mark.real_model, pytest.mark.real_class]

EXPECTATIONS = load_expectations()

# Families with both impls in the roster; the diff table covers the ones
# with a recipe on their attention modules, the rest must diff EMPTY.
DUAL_IMPL_FAMILIES = [
    name
    for name, impls in ((spec.name, spec.impls) for spec in FAMILY_BY_NAME.values())
    if impls == ("eager", "sdpa")
]

# The primary output leaf each family's kwargs-identity row compares, named
# after the direct-model attribute.
PRIMARY_LEAF = {
    "gpt2": "logits",
    "distilgpt2": "logits",
    "llama": "logits",
    "qwen2": "logits",
    "albert": "last_hidden_state",
    "distilbert": "last_hidden_state",
    "bert": "last_hidden_state",
    "vit": "last_hidden_state",
    "whisper": "logits",
    "mamba": "logits",
    "rwkv": "logits",
    "clip": "logits_per_image",
}


def test_eager_sdpa_facet_diff_is_exactly_the_pinned_table():
    """The FACET DIFF between attention implementations, never mere both-run.

    Measured: attn_out is structurally absent under EAGER -- the branch
    mechinterp users deliberately select -- and present under SDPA. A silent
    change in EITHER direction fails here; A01 updates the table when the
    fused-class gate dies.
    """

    for family in DUAL_IMPL_FAMILIES:
        eager_floors = EXPECTATIONS[family]["eager"]["floors"]
        sdpa_floors = EXPECTATIONS[family]["sdpa"]["floors"]
        measured_diff: dict[str, dict[str, tuple[str, ...]]] = {}
        for recipe in sorted(set(eager_floors) | set(sdpa_floors)):
            eager_set = set(eager_floors.get(recipe, ()))
            sdpa_set = set(sdpa_floors.get(recipe, ()))
            if eager_set != sdpa_set:
                measured_diff[recipe] = {
                    "sdpa_only": tuple(sorted(sdpa_set - eager_set)),
                    "eager_only": tuple(sorted(eager_set - sdpa_set)),
                }
        expected_diff = EAGER_SDPA_FACET_DIFF.get(family, {})
        assert measured_diff == expected_diff, (
            f"{family}: the eager/sdpa facet diff moved."
            f" measured={measured_diff} pinned={expected_diff}. An"
            " attention-implementation-dependent facet surface must be pinned"
            " explicitly (memo D7 class 1); update the table consciously in"
            " the fixing lane (A01)."
        )


def test_t5_refuses_sdpa_upstream():
    for family in SDPA_UNSUPPORTED_FAMILIES:
        spec = FAMILY_BY_NAME[family]
        assert spec.impls == ("eager",)
        with pytest.raises(ValueError, match="scaled_dot_product_attention"):
            spec.build("sdpa")


@pytest.mark.parametrize("family", sorted(PRIMARY_LEAF))
def test_kwargs_identity_traced_equals_direct(family, r0_capture):
    """Traced output == model(**kwargs) EXACTLY (memo D7 class 3).

    The capture must not shadow, drop, or perturb any forward kwarg: some
    traced output leaf is byte-identical to the direct model's primary leaf
    from an independent same-process rerun.
    """

    impl = "sdpa" if "sdpa" in FAMILY_BY_NAME[family].impls else "eager"
    cap = r0_capture(family, impl)
    with torch.no_grad():
        direct = getattr(cap.model(*cap.input_args, **cap.input_kwargs), PRIMARY_LEAF[family])
    matches = [
        op
        for op in cap.trace.output_ops
        if op.out.shape == direct.shape and torch.equal(op.out, direct)
    ]
    assert matches, (
        f"{family}: no traced output leaf equals the direct {PRIMARY_LEAF[family]}"
        " exactly -- a kwarg was shadowed/dropped or the capture perturbed the"
        " computation"
    )


def test_kwargs_identity_discriminates_a_dropped_mask(r0_capture):
    """The discriminating row: a real attention_mask CHANGES the values.

    If the capture transport ever dropped the mask kwarg, traced output
    would match the unmasked run instead -- both equalities are asserted so
    the row cannot pass vacuously.
    """

    import torchlens as tl

    spec = FAMILY_BY_NAME["gpt2"]
    model = spec.build("sdpa")
    kwargs = spec.input_kwargs()
    mask = torch.ones_like(kwargs["input_ids"])
    mask[0, 0] = 0  # a genuinely masked position
    with torch.no_grad():
        masked_direct = model(input_ids=kwargs["input_ids"], attention_mask=mask).logits
        unmasked_direct = model(input_ids=kwargs["input_ids"]).logits
    assert not torch.equal(masked_direct, unmasked_direct), (
        "fixture defect: the mask must change the computation for this row to discriminate"
    )
    trace = tl.trace(model, (), {"input_ids": kwargs["input_ids"], "attention_mask": mask})
    matches = [
        op
        for op in trace.output_ops
        if op.out.shape == masked_direct.shape and torch.equal(op.out, masked_direct)
    ]
    assert matches, "traced masked logits != direct masked logits: the mask kwarg was mangled"


def test_trace_signature_never_absorbs_forward_kwarg_names():
    """Static intersection check (memo D7 class 3).

    Model kwargs travel inside the input_kwargs MAPPING today, so nothing
    can shadow -- this pin exists so a flat-kwargs regression or a new
    tl.trace parameter colliding with a plausible forward-kwarg name fails
    loudly at PR time, with the collision list maintained consciously.
    """

    import torchlens as tl

    params = set(inspect.signature(tl.trace).parameters) - {"model", "input_args", "input_kwargs"}
    collisions = tuple(sorted(params & set(FORWARD_KWARG_CORPUS)))
    assert collisions == EXPECTED_TRACE_PARAM_COLLISIONS, (
        f"tl.trace parameter names colliding with the committed forward-kwarg"
        f" corpus moved: {collisions} != pinned {EXPECTED_TRACE_PARAM_COLLISIONS}."
        " A NEW collision invites silent shadowing if kwargs ever flatten;"
        " update the pin only with the collision consciously reviewed."
    )


class _LogitsOnly(nn.Module):
    """distilgpt2 wrapped to a bare-tensor output.

    The ModelOutput-container spelling crashes fork.do() today (the FIX-A
    enumerated-red row below); the bare-tensor spelling exercises the
    pass-qualified engine truth NOW so A05's fix cannot regress it.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids=input_ids).logits


def _multipass_add_sites(trace) -> list[str]:
    sites = []
    for layer in trace.layers:
        ops = getattr(layer, "ops", None)
        if ops is not None and len(ops) > 1 and "add" in ops[0].label:
            sites.append(ops[0].label.split(":")[0])
    return sites


def test_pass_qualification_pin_on_distilgpt2_two_pass_adds():
    """The OPUS-Q5 pin: pass-qualified edits land on exactly the addressed pass.

    A plain shrunk distilgpt2 forward (use_cache=False) carries 2-pass
    residual-add sites -- no ALBERT, no generation loop needed. Bare labels
    refuse; the pass-1 edit changes pass 1, recomputes downstream, and an
    identity donor patch is a no-op.
    """

    import torchlens as tl

    spec = FAMILY_BY_NAME["distilgpt2"]
    torch.manual_seed(SEED)
    model = _LogitsOnly(spec.build("sdpa")).eval()
    ids = spec.input_kwargs()["input_ids"]
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
    sites = _multipass_add_sites(trace)
    assert len(sites) >= 2, (
        f"expected 2-pass residual-add sites on the 2-layer distilgpt2, found {sites}"
    )
    site = sites[0]
    # Bare label refuses typed on a multi-pass site.
    fork_bare = trace.fork()
    with pytest.raises(Exception, match="ambiguous|multipass_bare_label"):
        fork_bare.do(
            tl.units(site, torch.ones_like(trace[f"{site}:1"].out, dtype=torch.bool)).resolve(
                fork_bare
            ),
            tl.zero_ablate(),
        )
    # Pass-qualified zero-ablation of one slice of pass 1.
    fork = trace.fork()
    mask = torch.zeros_like(trace[f"{site}:1"].out, dtype=torch.bool)
    mask[0, 0, :] = True
    fork.do(tl.units(f"{site}:1", mask).resolve(fork), tl.zero_ablate())
    assert torch.all(fork[f"{site}:1"].out[0, 0, :] == 0), "edited slice not ablated"
    assert torch.equal(fork[f"{site}:1"].out[0, 1:, :], trace[f"{site}:1"].out[0, 1:, :]), (
        "untouched elements of the addressed pass moved"
    )
    assert not torch.equal(fork[f"{site}:2"].out, trace[f"{site}:2"].out), (
        "pass 2 was not recomputed downstream of the pass-1 edit"
    )
    assert not torch.equal(fork.output_ops[0].out, trace.output_ops[0].out), (
        "the edit never reached the output"
    )
    # Identity donor: patch pass 2 from the unedited source trace -> no-op.
    fork_donor = trace.fork()
    fork_donor.do(tl.units(f"{site}:2", mask).resolve(fork_donor), tl.patch_from(trace))
    assert torch.equal(fork_donor.output_ops[0].out, trace.output_ops[0].out), (
        "identity-donor patch changed the output"
    )
    assert torch.equal(fork_donor[f"{site}:1"].out, trace[f"{site}:1"].out), (
        "patching pass 2 touched pass 1 (pass-blind donor regression)"
    )


def test_container_output_do_crash_enumerated_red(r0_capture):
    import torchlens as tl

    red = KNOWN_RED_BY_ID["container-output-do-replay"]
    spec = FAMILY_BY_NAME["distilgpt2"]
    model = spec.build("sdpa")
    ids = spec.input_kwargs()["input_ids"]
    trace = tl.trace(
        model, (), {"input_ids": ids}, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    sites = _multipass_add_sites(trace)
    fork = trace.fork()
    mask = torch.zeros_like(trace[f"{sites[0]}:1"].out, dtype=torch.bool)
    mask[0, 0, :] = True
    try:
        fork.do(tl.units(f"{sites[0]}:1", mask).resolve(fork), tl.zero_ablate())
    except red.exception_class() as exc:
        # Enumerated-red: a failure whose TYPE moves escapes this narrow
        # catch and errors raw -- re-pin or fix per the row's owner.
        assert red.message_substring in str(exc)
        return
    pytest.fail(
        f"STALE enumerated-red row {red.red_id!r}: fork.do() through a"
        " ModelOutput-container trace now works. {red.owner} landed FIX-A --"
        " delete the KNOWN_RED row and flip this test to assert the edit"
        " lands correctly through the container."
    )


def test_kwargs_only_plain_module_enumerated_red():
    import torchlens as tl

    red = KNOWN_RED_BY_ID["kwargs-only-plain-module"]

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    torch.manual_seed(SEED)
    model = Plain().eval()
    value = torch.randn(2, 4)
    try:
        tl.trace(model, (), {"x": value})
    except TypeError as exc:
        assert red.message_substring in str(exc), (
            f"kwargs-only transport failure signature drifted: {exc}"
        )
        return
    pytest.fail(
        f"STALE enumerated-red row {red.red_id!r}: kwargs-only tl.trace on a"
        f" plain module now works. {red.owner} fixed the transport -- delete"
        " the KNOWN_RED row and flip this to assert traced == direct output."
    )


def test_config_fingerprint_use_cache_row():
    """The silent-config-flag class (memo D7 #4): use_cache moves the graph.

    The resolved-config fingerprint MUST move with it, so op-count goldens
    keyed on the fingerprint can never silently ride a changed graph.
    """

    import torchlens as tl
    from tests.real_model.r0.families import build_gpt2
    from tests.real_model.registry import resolved_config_fingerprint

    model_no_cache = build_gpt2("sdpa")
    assert model_no_cache.config.use_cache is False
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(SEED)
    model_cache = GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=VOCAB,
            n_positions=64,
            bos_token_id=0,
            eos_token_id=0,
            use_cache=True,
        )
    ).eval()
    model_cache.config._attn_implementation = "sdpa"
    fp_no_cache = resolved_config_fingerprint(model=model_no_cache)
    fp_cache = resolved_config_fingerprint(model=model_cache)
    assert fp_no_cache != fp_cache
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, VOCAB, (1, 8), generator=generator)
    ops_no_cache = len(tl.trace(model_no_cache, (), {"input_ids": ids}).ops)
    ops_cache = len(tl.trace(model_cache, (), {"input_ids": ids}).ops)
    assert ops_no_cache != ops_cache, (
        "use_cache flipped without changing the captured graph? the"
        " config-fingerprint class lost its measured basis; re-measure"
    )


def test_config_fingerprint_gradient_checkpointing_row():
    """gradient_checkpointing_enable changes execution WITHOUT touching
    config.use_cache serialization -- exactly the laundering the fingerprint
    exists to catch: it folds in gradient-checkpointing and train state.
    """

    from tests.real_model.r0.families import build_gpt2
    from tests.real_model.registry import resolved_config_fingerprint

    model = build_gpt2("sdpa")
    fp_before = resolved_config_fingerprint(model=model)
    use_cache_before = model.config.use_cache
    model.gradient_checkpointing_enable()
    assert model.config.use_cache == use_cache_before, (
        "5.x now mutates config.use_cache at enable time; the fingerprint rows"
        " should be re-derived against the new behavior"
    )
    fp_gc = resolved_config_fingerprint(model=model)
    assert fp_gc != fp_before, "fingerprint blind to gradient checkpointing"
    model.train()
    fp_train = resolved_config_fingerprint(model=model)
    assert fp_train != fp_gc, "fingerprint blind to train mode"
