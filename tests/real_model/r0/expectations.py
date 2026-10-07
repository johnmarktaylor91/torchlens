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
  at least ``min_version`` (:func:`expected_n_ops`). Only llama carries one:
  transformers 5.19 rewrote ``LlamaRotaryEmbedding.forward`` (``inv_freq``
  ``__getitem__``/``expand``/``to``/``__matmul__``/``transpose`` became one
  ``to`` and ``__mul__``), so llama is 180 (eager) / 144 (sdpa) there, against
  183 / 147 on the pinned 5.18.0. The op diff was checked to be exactly that
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
    # albert/qwen2/clip/whisper: no facet-availability diff between
    # implementations today (their attention carries no recipe, so there is
    # nothing to differ -- the qwen2 emptiness itself is the KNOWN-GAP row).
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
GPT2_LN_INPUT_AVAILABLE_ADDRESSES = ("transformer.h.0.ln_1",)

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
    KnownRed(
        red_id="kwargs-only-plain-module",
        owner="fix bundle (A04/A05 kwarg transport)",
        exception_type="TypeError",
        message_substring="got multiple values for argument",
        notes="tl.trace(model, (), {name: tensor}) on a plain nn.Module passes the"
        " kwarg twice; HF classes escape it. Loud today, but the loud/silent"
        " asymmetry is the memo's kwarg-shadowing defect class",
    ),
)

KNOWN_RED_BY_ID = {row.red_id: row for row in KNOWN_RED}


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
