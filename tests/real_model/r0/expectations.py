"""Committed R0 expectations: floors, diffs, manifests (build rows A4/A5).

``expectations_r0.json`` is the REVIEWED measured golden (generated against
transformers 5.18.0 / torch 2.13.0 on 2026-10-04 by tracing the committed
``families.py`` builders; regenerate deliberately, never with ``--fix``).
The fingerprint and n_ops fields were re-recorded from a prior
transformers 5.14.1 golden (FK-transformers re-record, 2026-10-04): every
family's fingerprint moved because ``config.to_dict()`` embeds
``transformers_version``, which changes on every release regardless of
behavior. Four rows also carry a genuine op-count change confirmed against
the installed 5.14.1 vs 5.18.0 package sources, not just the trace: llama
(both impls, 186->183 / 150->147) lost 3 redundant ``.float()`` casts in
``LlamaRotaryEmbedding.forward`` (upstream now folds the dtype cast into a
single ``.to(dtype=..., device=...)`` and stopped re-casting
already-float32 operands); t5 (eager, 326->324) and mamba (eager, 255->279)
moved under the same upstream attention-interface / kernel-dispatch
refactor that added T5's SDPA support (see ``SDPA_UNSUPPORTED_FAMILIES``
below). No row's recipe classification, facet floors, or false-claims
manifest changed.
rwkv (eager, 585->589, 2026-10-05) is the intended effect of capturing in-place
ops on a prepared ``nn.Parameter``: ``RwkvModel._rescale_layers`` runs, on the
first eval forward and under ``torch.no_grad()``,
``block.attention.output.weight.div_(...)`` and
``block.feed_forward.value.weight.div_(...)`` for each of the 2 blocks. Those
4 ``div_`` ops are now logged with the weight as their parameter input, and the
block's output ``linear`` reads the mutated weight through them.

- ``recipes``: the EXACT per-family recipe classification map (a recipe
  appearing on fewer modules is a regression; on more, a conscious update).
- ``floors``: per recipe, the facet set that MUST be ``available_now`` on
  EVERY row of that recipe (memo D2 ladder #2: the positive expectation
  ledger). Fix lanes may only grow these.
- ``false_claims``: the enumerated KNOWN-FALSE ``structurally_absent``
  manifest, adjudicated mechanically (``adjudication.py``). The sweep
  asserts adjudication output == this list EXACTLY per (family, impl): a
  NEW false claim fails immediately; a FIXED one goes stale loudly and the
  fixing lane deletes its row (the memo's enumerated-red idiom, test-side).
- ``n_ops``: op-count golden, asserted only under a matching
  resolved-config fingerprint (memo D7 class 4).
- ``n_ops_by_transformers``: optional per-impl list of ``{"min_version",
  "n_ops"}`` entries that replace ``n_ops`` when the installed transformers is
  at least ``min_version`` (:func:`expected_n_ops`). Only llama and qwen2
  carry one: transformers 5.19 rewrote their rotary embedding ``forward``
  (``inv_freq`` ``__getitem__``/``expand``/``__matmul__``/``transpose``, plus
  llama's ``to`` and qwen2's three ``float`` casts, became one ``__mul__``), so
  llama is 180 (eager) / 144 (sdpa) and qwen2 180 / 144 there, against 183 / 147
  and 186 / 150 on the pinned 5.18.0. The op diff was checked to be exactly that
  rotary block on torch 2.7.1 and 2.14.1 (2026-10-07, next-release CI round).

Ownership of the known-false rows (for the teaching message):
``final_norm_present`` -> lane A02 (semantic RESIDUAL slice, final-norm
anchor); ``attention_projections_present`` -> lane A01 (semantic ATTENTION
slice, config-based fused detection, eager+sdpa).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EXPECTATIONS_PATH = Path(__file__).resolve().parent / "expectations_r0.json"

FALSE_CLAIM_OWNERS = {
    "final_norm_present": "A02 (semantic residual slice: final-norm anchor)",
    "attention_projections_present": "A01 (semantic attention slice: fused-class gate)",
    "same_recipe_serves_it_elsewhere": "A01/A02 (recipe coherence)",
}

# The FACET DIFF between attention implementations, asserted explicitly
# (memo 4.1: "with the FACET DIFF asserted explicitly (not mere both-run)").
# Post-A01 (fused-class gate dead, detection graph+config-keyed): attn_out is
# available on BOTH implementations, and the remaining diff is the honest
# read-vs-reconstruct boundary -- eager serves scores/pattern/z as REAL
# captured ops (plus the computed per-head result where the output projection
# lives inside the module), while a plain SDPA capture reports them
# needs_capture until reconstruction_ready=True saves the SDPA arguments.
# A silent change in either direction still fails here.
_EAGER_REAL_OPS = {"sdpa_only": (), "eager_only": ("pattern", "result", "scores", "z")}
EAGER_SDPA_FACET_DIFF: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "gpt2": {"gpt2_attention": dict(_EAGER_REAL_OPS)},
    "distilgpt2": {"gpt2_attention": dict(_EAGER_REAL_OPS)},
    "llama": {"gqa_attention": dict(_EAGER_REAL_OPS)},
    "vit": {"gqa_attention": dict(_EAGER_REAL_OPS)},
    "distilbert": {"distilbert_attention": dict(_EAGER_REAL_OPS)},
    # BERT's output projection lives OUTSIDE BertSelfAttention (in
    # BertSelfOutput), so per-head result is structurally absent on both
    # implementations and only the real-op trio differs.
    "bert": {"bert_self_attention": {"sdpa_only": (), "eager_only": ("pattern", "scores", "z")}},
    # qwen2 (forks.10) rides the Llama gqa recipe; clip (U55.1) has its own
    # clip_attention recipe on both towers. Both show the same honest
    # read-vs-reconstruct boundary as llama and vit.
    "qwen2": {"gqa_attention": dict(_EAGER_REAL_OPS)},
    "clip": {"clip_attention": dict(_EAGER_REAL_OPS)},
    # albert/whisper: no facet-availability diff between implementations
    # today (their attention carries no recipe, so there is nothing to differ).
}

# Families whose upstream class REFUSES sdpa outright (pinned upstream
# behavior, not a TorchLens gap). T5 raised ValueError at construction
# through transformers 5.14.1; 5.18.0 added a unified attention-interface
# (ALL_ATTENTION_FUNCTIONS) to T5Attention and set `_supports_sdpa = True`,
# so T5 now builds under sdpa instead of refusing (FK-transformers
# re-record, 2026-10-04; see test_t5_sdpa_now_builds_upstream). T5 stays
# eager-only in FamilySpec.impls -- adding its sdpa facet floors to the deep
# sweep is a conscious coverage expansion for a future lane, not a
# re-record.
SDPA_UNSUPPORTED_FAMILIES = ()

# Facet -> expected shape, resolved against FamilySpec.dims at assert time.
# ("b" = batch, "s" = sequence, "h" = hidden, "n" = QUERY heads, "g" = KV
# heads (= n for MHA), "d" = d_head.)
# The point is the mask-shape kill: a facet that should be (b, s, h) failing
# because it arrived ids- or mask-shaped is the silent-wrong class the memo
# names. q keeps the served layout (b, s, n, d); k/v carry KV heads (GQA
# serves them un-expanded); A01's per-head facets pin the canonical layouts
# the head view slices: scores/pattern are head-major (b, n, dst, src), z is
# head-major (b, n, s, d), and the per-head result is position-major
# (b, s, n, h) -- conscious A01 table update, never silent.
FACET_SHAPE_SPECS: dict[str, tuple[str, ...]] = {
    "resid_pre": ("b", "s", "h"),
    "resid_mid": ("b", "s", "h"),
    "resid_post": ("b", "s", "h"),
    "q": ("b", "s", "n", "d"),
    "k": ("b", "s", "g", "d"),
    "v": ("b", "s", "g", "d"),
    "attn_out": ("b", "s", "h"),
    "scores": ("b", "n", "s", "s"),
    "pattern": ("b", "n", "s", "s"),
    "z": ("b", "n", "s", "d"),
    "result": ("b", "s", "n", "h"),
    "normalized": ("b", "s", "h"),
    "logits": ("b", "s", "v"),
}

# Per-site dimension overrides: sites whose CORRECT dims differ from the
# family-level dims (dual towers, downsampled encoders, factorized
# embeddings). Matched by address prefix, longest match wins.
SHAPE_DIM_OVERRIDES: dict[str, tuple[tuple[str, dict[str, int]], ...]] = {
    "clip": (("vision_model", {"s": 5}),),  # 4 patches + CLS on the 32px/16px fixture
    "whisper": (("model.encoder", {"s": 32}),),  # mel frames downsample to 32 positions
    "albert": (("embeddings", {"h": 32}),),  # factorized embedding_size=32 before projection
}

# Per-(address, facet) full shape-spec overrides for sites whose facet rank
# legitimately differs (pooled outputs).
SHAPE_SPEC_OVERRIDES: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("clip", "vision_model.post_layernorm", "normalized"): ("b", "h"),  # pooled CLS
}

# ENUMERATED WRONG-PAYLOAD MANIFEST (the A02 flagship class): `resid_pre` was
# served IDS-SHAPED -- anchored to the token ids, not the residual stream --
# on 12 exact rows measured 2026-08-26 (gpt2/distilgpt2: both blocks, both
# impls; llama/qwen2: layer 1). A02 (semantic residual slice: "resid_pre by
# dataflow+shape") fixed the anchor and DELETED every row in its own change,
# so the honest-shape floor now enforces the stream shape on all of them.
# The manifest stays as the mechanism: a NEW wrong-payload site gets a row
# here (with its owner) and the sweep pins its exact wrong shape until fixed.
WRONG_PAYLOAD_MANIFEST: dict[tuple[str, str, str, str], tuple[int, ...]] = {}
WRONG_PAYLOAD_OWNER = "A02 (semantic residual slice)"

# The class-7 capture-dependency inconsistency, pinned exactly (memo D7
# rank 7): on the SAME gpt2 capture, block 0's ln_1 serves `input` while
# every later LayerNorm reports needs_capture for it. User-visible
# inconsistency; pinned so a fix (or a regression to all-unavailable)
# surfaces here rather than shifting silently.
#
# grouping rule rows 10 and 12: each residual add in a gpt2 block is its own
# single-pass layer, so every LayerNorm's input op resolves and the
# inconsistency is gone; the pin now holds all five LayerNorms (a regression
# to the partial or all-unavailable state still surfaces here).
GPT2_LN_INPUT_AVAILABLE_ADDRESSES = (
    "transformer.h.0.ln_1",
    "transformer.h.0.ln_2",
    "transformer.h.1.ln_1",
    "transformer.h.1.ln_2",
    "transformer.ln_f",
)

# Common forward-kwarg name corpus (memo D7 class 3). The static
# intersection check pins signature(tl.trace) INTERSECT this corpus to the
# committed set below; a new tl.trace parameter colliding with a plausible
# model-forward kwarg fails loudly at PR time.
FORWARD_KWARG_CORPUS = (
    "attention_mask",
    "backend",
    "cache",
    "cache_position",
    "capture",
    "context",
    "decoder_input_ids",
    "encoder_hidden_states",
    "episode",
    "halt",
    "head_mask",
    "hidden_states",
    "images",
    "input_features",
    "input_ids",
    "inputs_embeds",
    "intervene",
    "labels",
    "lookback",
    "mask",
    "mode",
    "name",
    "past_key_values",
    "pixel_values",
    "position_ids",
    "profile",
    "recipes",
    "return_dict",
    "save",
    "scale",
    "state",
    "storage",
    "streaming",
    "temperature",
    "token_type_ids",
    "top_k",
    "use_cache",
    "output_attentions",
    "output_hidden_states",
)
# tl.trace parameter names that ALSO read as plausible model-forward kwargs
# (measured against the 23-parameter post-P00 signature). These cannot
# shadow anything today -- model kwargs travel inside the input_kwargs
# MAPPING -- but the pin fails the moment a flat-kwargs regression or a new
# colliding parameter lands.
EXPECTED_TRACE_PARAM_COLLISIONS = (
    "backend",
    "capture",
    "episode",
    "halt",
    "intervene",
    "lookback",
    "profile",
    "recipes",
    "save",
    "storage",
    "streaming",
)


@dataclass(frozen=True)
class KnownRed:
    """One enumerated-red row: a defect pinned by its CURRENT signature.

    The pinning test asserts the failure still reproduces with this exact
    signature. When the owning lane fixes the defect the row goes STALE and
    the test fails loudly; the fixing lane deletes the row and flips the
    assertion to the true oracle in the same change.
    """

    red_id: str
    owner: str
    exception_type: str
    message_substring: str
    notes: str

    def exception_class(self) -> type[BaseException]:
        """Resolve the pinned exception-type name to its concrete class.

        Pinning tests catch exactly this class: a defect whose failure TYPE
        moves escapes the narrow catch and surfaces as a raw error, which is
        the re-pin signal.
        """

        import builtins

        resolved = getattr(builtins, self.exception_type, None)
        if resolved is None:
            from torchlens.semantic.logit_lens import LogitLensError

            resolved = {"LogitLensError": LogitLensError}[self.exception_type]
        assert isinstance(resolved, type) and issubclass(resolved, BaseException)
        return resolved


KNOWN_RED: tuple[KnownRed, ...] = (
    # The "logit-lens-final-norm" row (flagship logit_lens failure, owner
    # A03) was deleted by A02: its root cause was the false final_norm_*
    # absence claims -- the final-norm dataflow anchor fix (mikit F6) makes
    # logit_lens succeed on the config-built GPT-2, and the sweep test now
    # asserts the true oracle (final-layer lens == the model's own logits).
    # container-output-do-replay (A05, FIX-A) was fixed 2026-08-26: the replay
    # engine no longer re-applies a boundary output node's recorded MODEL-output
    # container path to the replayed call's already-resolved output. The axes
    # test flipped to assert the edit lands correctly through the container.
    # kwargs-only-plain-module was fixed 2026-10-11 (walk U02.2): an empty
    # positional tuple on a single-parameter forward is no longer wrapped into
    # [[]], so tl.trace(model, (), {name: tensor}) reaches forward(name=...)
    # once. The axes test asserts traced == direct output.
)

KNOWN_RED_BY_ID = {row.red_id: row for row in KNOWN_RED}


@dataclass(frozen=True)
class DatedGap:
    """A dated known coverage gap: a leg no venue runs yet, never a red row.

    ``KNOWN_RED`` pins defects that reproduce; a dated gap names coverage
    that cannot execute on any current venue, with its entry date, the ruling
    that created it and the condition that deletes it.
    """

    gap_id: str
    dated: str
    ruling: str
    owner: str
    closes_when: str
    notes: str


DATED_GAPS: tuple[DatedGap, ...] = (
    DatedGap(
        gap_id="flash-attention-2-impl",
        dated="2026-10-12",
        ruling="U42.1",
        owner="the GPU real-model leg (campaign 2)",
        closes_when=(
            "a GPU leg captures and validates a real-model family under"
            " attn_implementation='flash_attention_2'"
        ),
        notes="FA2 kernels need CUDA and the flash-attn package; every real-model"
        " leg today is CPU, so no family runs the flash_attention_2"
        " implementation (FamilySpec.impls stays eager/sdpa)",
    ),
    DatedGap(
        gap_id="vjepa2-full-clip-forward-validation",
        dated="2026-10-12",
        ruling="U55.4",
        owner="the GPU real-model leg (campaign 2)",
        closes_when=(
            "forward validation of r2-vjepa2-vitl-fpc64-256 at its native 64 frames"
            " passes on a GPU venue (tests/real_model/gpu_legs/vjepa2_full_leg.py)"
        ),
        notes="measured on one 48 GB L40 at fp32: the 64-frame capture passes"
        " (5937 ops, traced output bitwise equal, 55.9 GB retained on host), but"
        " forward validation keeps saved activations on the device and the default"
        " save budget refused it at 64, 32 and 16 frames (projected 9.5, 14.2 and"
        " 16.5 GB lower bounds against half the free device memory); the gather"
        " index error seen at 8 and 16 frames is a separate defect report",
    ),
)

DATED_GAPS_BY_ID = {row.gap_id: row for row in DATED_GAPS}


def expected_n_ops(expected: dict) -> int:
    """Return the op-count golden for the installed transformers version.

    Parameters
    ----------
    expected:
        One family/impl row of ``expectations_r0.json``.

    Returns
    -------
    int
        The ``n_ops`` of the highest ``n_ops_by_transformers`` entry whose
        ``min_version`` the installed transformers meets, else ``n_ops``.
    """

    import transformers
    from packaging.version import Version

    installed = Version(transformers.__version__)
    applicable = [
        entry
        for entry in expected.get("n_ops_by_transformers", ())
        if installed >= Version(entry["min_version"])
    ]
    if not applicable:
        return int(expected["n_ops"])
    return int(max(applicable, key=lambda entry: Version(entry["min_version"]))["n_ops"])


def load_expectations() -> dict:
    """Parse the committed measured-golden JSON."""

    return json.loads(EXPECTATIONS_PATH.read_text())
