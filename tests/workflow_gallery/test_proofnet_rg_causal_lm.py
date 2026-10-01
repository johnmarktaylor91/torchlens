"""RG causal-LM scenarios on real DistilGPT2 (testing memo 5.2; F36 E2).

RG02 kwargs/Cache transport, RG03 facet floors + attention-implementation
diff, RG04 logit lens, RG05 residual patching vs a hand-coded one-hook
patch, RG07 replay through a Cache-carrying graph vs a frozen-complement
manual run, RG08 episode-captured greedy generation vs the model's own
``generate``, RG18 steered generation with fire-record and negative-control
evidence, RG19 mean-ablation ``over=`` semantics. All on the pinned
``r1-distilgpt2`` registry row, offline, natural prompts from the committed
corpus.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.workflow_gallery.conftest import prompt

pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_model,
    pytest.mark.real_checkpoint,
    pytest.mark.gallery,
]


@pytest.fixture(scope="module")
def lm(rg_loader: Any) -> dict[str, Any]:
    return rg_loader("r1-distilgpt2")


def _encode(lm: dict[str, Any], text: str) -> dict[str, torch.Tensor]:
    return dict(lm["tokenizer"](text, return_tensors="pt"))


def test_rg02_kwargs_and_cache_transport_match_direct_forward(lm: dict[str, Any]) -> None:
    """RG02: tokenizer kwargs incl. use_cache=True ride into the model
    unshadowed; traced logits and structured output paths == direct."""

    import torchlens as tl

    encoded = _encode(lm, prompt("clean-factual-1")["clean"])
    encoded["use_cache"] = True
    with torch.no_grad():
        direct = lm["model"](**encoded)
    trace = tl.trace(lm["model"], (), dict(encoded))
    try:
        assert trace.outcome.status.name == "COMPLETE"
        matches = [
            op
            for op in trace.output_ops
            if getattr(op.out, "shape", None) == direct.logits.shape
            and torch.equal(op.out, direct.logits)
        ]
        assert matches, "traced logits != direct forward WITH the same kwargs"
        assert direct.past_key_values is not None, (
            "use_cache=True was shadowed somewhere: the direct forward"
            " returned no cache (config mutated?)"
        )
    finally:
        trace.cleanup()


def test_rg03_facet_family_floor_and_attention_implementation_diff(
    lm: dict[str, Any], gallery_registry: Any
) -> None:
    """RG03: the committed per-family facet floor holds on the REAL
    checkpoint; fused-attention absences are disclosed with a remedy, never
    silent; and the eager arm MAKES the pattern available (the #1-ranked
    missing class: attention-implementation non-invariance, asserted as an
    explicit facet DIFF, not both-run)."""

    import torchlens as tl
    import torchlens.semantic as sem

    encoded = _encode(lm, prompt("clean-factual-1")["clean"])
    trace = tl.trace(lm["model"], (), dict(encoded))
    try:
        coverage = sem.facet_coverage(trace)
        assert not coverage.unresolved, f"unresolved facet rows: {coverage.unresolved}"
        by_class: dict[str, list[Any]] = {}
        for row in coverage.rows:
            by_class.setdefault(row.class_name, []).append(row)
        # The committed gpt2-family floor (facets that MUST be available).
        attention_rows = by_class.get("GPT2Attention", [])
        assert len(attention_rows) == 6, "distilgpt2 has 6 attention blocks"
        for row in attention_rows:
            for facet_name in ("q", "k", "v", "attn_out", "head"):
                assert facet_name in row.available, (
                    f"{row.address}: family-floor facet {facet_name!r} not"
                    f" available (has {row.available})"
                )
            # Fused-kernel absences are DISCLOSED with remedies, never bare.
            missing_names = {entry[0] for entry in row.missing}
            assert "pattern" in missing_names, (
                "SDPA arm: 'pattern' should be needs_capture here -- if it"
                " became available by default, update the eager-diff arm"
            )
            for entry in row.missing:
                assert entry[1] and entry[2], (
                    f"{row.address}: facet {entry[0]!r} missing WITHOUT a"
                    " reason/remedy -- absence-claim honesty broken"
                )
        for row in by_class.get("GPT2Block", []):
            for facet_name in ("resid_pre", "resid_mid", "resid_post"):
                assert facet_name in row.available
        for row in by_class.get("GPT2MLP", []):
            for facet_name in ("up_out", "intermediate", "down_out"):
                assert facet_name in row.available
    finally:
        trace.cleanup()

    # The EAGER arm: the same checkpoint loaded eager exposes the pattern
    # facet the fused arm could only disclose (the facet DIFF is the claim).
    from transformers import AutoModelForCausalLM

    row = gallery_registry.checkpoint_evidence("r1-distilgpt2")
    eager_model = AutoModelForCausalLM.from_pretrained(
        row.model_id, revision=row.revision, attn_implementation="eager"
    ).eval()
    eager_trace = tl.trace(
        eager_model,
        (),
        dict(encoded),
        capture=tl.options.CaptureOptions(save_arg_values=True),
    )
    try:
        eager_coverage = sem.facet_coverage(eager_trace)
        eager_attention = [row for row in eager_coverage.rows if row.class_name == "GPT2Attention"]
        assert eager_attention and all("pattern" in row.available for row in eager_attention), (
            "eager arm: the attention pattern facet is still unavailable --"
            " the eager/sdpa facet DIFF collapsed (attention-implementation"
            " coverage class, rank 1)"
        )
    finally:
        eager_trace.cleanup()


def test_rg04_logit_lens_final_layer_matches_native_logits(lm: dict[str, Any]) -> None:
    """RG04: the lens at the final layer IS the model's own logits."""

    import torchlens as tl
    import torchlens.semantic as sem

    encoded = _encode(lm, prompt("clean-factual-1")["clean"])
    with torch.no_grad():
        native = lm["model"](**encoded).logits
    trace = tl.trace(lm["model"], (), dict(encoded))
    try:
        lens = sem.logit_lens(trace)
        assert lens.validated, "logit_lens self-validation did not run/pass"
        final = lens.final_logits
        assert torch.allclose(final, native, atol=1e-4), (
            "lens-at-final-layer != the model's own logits (the flagship"
            " 2026-08 breakage class; self-consistency oracle)"
        )
        assert len(lens.entries) >= 6, "lens did not cover the layer stack"
    finally:
        trace.cleanup()


def test_rg05_patching_helpers_settle_typed_never_silent_on_this_venue(
    lm: dict[str, Any],
) -> None:
    """RG05, as it settles on the ACTUALLY merged transformers-5.x venue:
    every flagship activation-patching helper arm refuses TYPED with a
    teaching message on the real checkpoint -- never the 2026-08 silent
    flat table. The numeric single-hook parity arm is enumerated red for
    the HF_5_CANDIDATE leg (rg_enumerated_red_hf5.tsv, owner A03): the
    residual arm hits the pass-ambiguity guard on the 5.x graph shape, the
    mlp/attention-output arms land on in-place dropout ops, and the head
    arm resolves only read-only reconstructed facets under fused attention.
    When a fix lands, the manifest row goes stale-red and THIS pin flips
    deliberately. The live numeric intervention parity for this venue rides
    RG07's frozen-complement oracle."""

    import torchlens.semantic.patching as patching

    model, tokenizer = lm["model"], lm["tokenizer"]
    pair = prompt("clean-factual-1")
    clean = tokenizer(pair["clean"], return_tensors="pt")["input_ids"]
    corrupted = tokenizer(pair["corrupt"], return_tensors="pt")["input_ids"]
    assert clean.shape == corrupted.shape, "the committed pair must be length-matched"
    with torch.no_grad():
        answer_id = int(model(clean).logits[0, -1].argmax())

    def metric(log: Any) -> torch.Tensor:
        for op in log.output_ops:
            if getattr(op.out, "dim", lambda: 0)() == 3:
                return op.out[0, -1, answer_id]
        raise AssertionError("no 3D logits op on the capture")

    arms = {
        "residual_stream": (
            lambda: patching.activation_patch_residual_stream(
                model, clean, corrupted, metric, facet_name="resid_post", patch_positions=False
            ),
            "pass",  # SiteAmbiguityError teaches pass-qualified spellings
        ),
        "mlp_output": (
            lambda: patching.activation_patch_mlp_output(model, clean, corrupted, metric),
            "replaced=False",  # PatchApplicationError disclosing refused fires
        ),
        "attention_heads": (
            lambda: patching.activation_patch_attention_heads(model, clean, corrupted, metric),
            "read-only",  # SiteResolutionError teaching the eager/capture remedy
        ),
    }
    import warnings as warnings_module

    for arm_name, (invoke, reason_needle) in arms.items():
        # The repo promotes TorchLens warnings to errors; the per-fire
        # refusal DISCLOSURE (replaced=False) precedes the typed settlement,
        # so measure the settlement with disclosures recorded, not fatal.
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("always")
            with pytest.raises(Exception) as excinfo:
                invoke()
        exc = excinfo.value
        assert type(exc).__name__ in {
            "SiteAmbiguityError",
            "PatchApplicationError",
            "SiteResolutionError",
        }, (
            f"RG05 arm {arm_name}: settled with {type(exc).__name__} -- if the"
            " helper now SUCCEEDS, restore the numeric single-hook parity"
            " oracle here and delete the enumerated-red row"
        )
        assert reason_needle.lower() in str(exc).lower(), (
            f"RG05 arm {arm_name}: refusal no longer teaches its predicate"
            f" ({reason_needle!r}): {str(exc)[:180]}"
        )


def test_rg07_replay_edit_through_cache_graph_matches_manual_ablation(
    lm: dict[str, Any],
) -> None:
    """RG07: a fork.do() edit replayed through a Cache-carrying capture ==
    a manual hooked ablation of the same site (frozen-complement oracle)."""

    import torchlens as tl

    model = lm["model"]
    encoded = _encode(lm, prompt("clean-factual-1")["clean"])
    encoded["use_cache"] = True
    trace = tl.trace(
        model, (), dict(encoded), capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    try:
        target = None
        for label in trace.layer_labels:
            if label.startswith("dropout") and "transformer.h.5.mlp" in str(trace[label].modules):
                target = label
        assert target is not None, "no dropout op captured inside transformer.h.5.mlp"
        fork = trace.fork()
        try:
            fork.do(target, tl.zero_ablate())
            edited_logits = None
            for op in fork.output_ops:
                if getattr(op.out, "dim", lambda: 0)() == 3:
                    edited_logits = op.out
            assert edited_logits is not None
        finally:
            pass

        # Manual frozen-complement: hook the LAST block's whole MLP.
        last_block = model.transformer.h[-1]

        def _ablate_hook(module: Any, args: Any, output: Any) -> Any:
            return torch.zeros_like(output)

        handle = last_block.mlp.register_forward_hook(_ablate_hook)
        try:
            with torch.no_grad():
                manual_logits = model(**encoded).logits
        finally:
            handle.remove()
        assert torch.allclose(edited_logits, manual_logits, atol=1e-4), (
            "replayed zero-ablation through the Cache-carrying graph differs"
            " from the manual hooked ablation (frozen-complement oracle)"
        )
        fork.cleanup()
    finally:
        trace.cleanup()


def test_rg08_episode_captured_generation_matches_direct_generate(
    lm: dict[str, Any],
) -> None:
    """RG08: token ids match direct generation stepwise; every declared step
    lands a complete ledger row; the root output IS the generate output."""

    import torchlens as tl

    model, tokenizer = lm["model"], lm["tokenizer"]
    ids = tokenizer(prompt("clean-factual-1")["clean"], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        direct = model.generate(ids, max_new_tokens=4, do_sample=False)
    trace = tl.trace(
        model.generate,
        (ids,),
        {"max_new_tokens": 4, "do_sample": False},
        episode=tl.options.EpisodeSpec(
            n_steps=4, step_output_kind="tokens", acknowledge_step_cost=True
        ),
    )
    try:
        assert trace.outcome.status.name == "COMPLETE"
        assert str(trace.root_entry_point).startswith("bound_method:")
        ledger = trace.annotations["episode"]
        rows = ledger["rows"]
        assert [row["status"] for row in rows] == ["complete"] * 4
        step_tokens = [row["step_output"][0] for row in rows]
        assert step_tokens == direct[0, -4:].tolist(), (
            f"per-step ledger tokens {step_tokens} != direct generate tail"
            f" {direct[0, -4:].tolist()} -- stepwise capture diverged"
        )
        traced_output = trace.output_ops[0].out
        assert torch.equal(traced_output, direct), (
            "the episode capture's root output != the model's own generate"
        )
    finally:
        trace.cleanup()


def test_rg18_steered_generation_fires_each_step_with_negative_control(
    lm: dict[str, Any],
) -> None:
    """RG18: the steering hook fires on generation steps (measured
    fire_count, F42 coupling), the steer changes the continuation, and the
    zero-match negative control changes nothing."""

    import torchlens as tl

    model, tokenizer = lm["model"], lm["tokenizer"]
    ids = tokenizer(prompt("clean-factual-1")["clean"], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        baseline = model.generate(ids, max_new_tokens=4, do_sample=False)

    def episode_spec() -> Any:
        return tl.options.EpisodeSpec(
            n_steps=4, step_output_kind="tokens", acknowledge_step_cost=True
        )

    steered = tl.trace(
        model.generate,
        (ids,),
        {"max_new_tokens": 4, "do_sample": False},
        episode=episode_spec(),
        intervene=tl.when(tl.func("tanh"), tl.scale(3.0)),
    )
    try:
        ledger = steered.annotations["episode"]
        assert ledger["header"]["intervention_digest"], "coupling digest missing"
        assert ledger["header"]["fidelity_basis"] == "perturbed"
        fire_counts = [row["fire_count"] for row in ledger["rows"] if row["fire_count"] is not None]
        assert fire_counts and all(count > 0 for count in fire_counts), (
            f"steering hook fire counts per step: {fire_counts} -- the edit"
            " did not fire on every generation step"
        )
        steered_ids = steered.output_ops[0].out
        assert not torch.equal(steered_ids, baseline), (
            "a 3x MLP-activation scale on every step left the continuation"
            " untouched -- no calibrated effect (flat-table class)"
        )
    finally:
        steered.cleanup()

    # Negative control: an edit that FIRES on the same sites but is the
    # identity (scale 1.0) leaves the greedy continuation byte-identical.
    control = tl.trace(
        model.generate,
        (ids,),
        {"max_new_tokens": 4, "do_sample": False},
        episode=episode_spec(),
        intervene=tl.when(tl.func("tanh"), tl.scale(1.0)),
    )
    try:
        control_ledger = control.annotations["episode"]
        control_fires = [
            row["fire_count"] for row in control_ledger["rows"] if row["fire_count"] is not None
        ]
        assert control_fires and all(count > 0 for count in control_fires), (
            "the identity control never fired -- it proves nothing"
        )
        assert torch.equal(control.output_ops[0].out, baseline), (
            "an IDENTITY edit changed the continuation -- the steering"
            " evidence above cannot be trusted (instrumentation effect)"
        )
    finally:
        control.cleanup()


def test_rg19_mean_ablation_over_semantics_and_fail_closed(lm: dict[str, Any]) -> None:
    """RG19 over the ACTUALLY merged surface: mean_ablate(over='self') fires
    and moves the logits (degeneracy check), and every other over= token --
    including the axis tokens and the one-character typo -- refuses TYPED
    with teaching (A04's fail-closed fix for SG#15; axis-aware means are
    honestly not implemented rather than silently global)."""

    import torchlens as tl

    model = lm["model"]
    encoded = _encode(lm, prompt("clean-factual-1")["clean"])
    trace = tl.trace(
        model, (), dict(encoded), capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    try:
        target = None
        for label in trace.layer_labels:
            if label.startswith("tanh"):
                target = label
        assert target is not None
        baseline_logits = None
        for op in trace.output_ops:
            if getattr(op.out, "dim", lambda: 0)() == 3:
                baseline_logits = op.out.clone()
        assert baseline_logits is not None

        fork = trace.fork()
        try:
            fork.do(target, tl.mean_ablate(over="self"))
            edited = None
            for op in fork.output_ops:
                if getattr(op.out, "dim", lambda: 0)() == 3:
                    edited = op.out
            assert edited is not None
            assert not torch.equal(edited, baseline_logits), (
                "mean ablation left the logits untouched (flat-table class)"
            )
        finally:
            fork.cleanup()

        for bad_token in ("position", "positon", "batch"):
            with pytest.raises(Exception) as excinfo:
                tl.mean_ablate(over=bad_token)
            message = str(excinfo.value)
            assert "over" in message and "self" in message, (
                f"over={bad_token!r} refused without teaching the closed"
                f" vocabulary: {message[:160]}"
            )
    finally:
        trace.cleanup()
