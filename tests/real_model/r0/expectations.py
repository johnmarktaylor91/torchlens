"""Committed R0 expectations: floors, diffs, manifests (build rows A4/A5).

``expectations_r0.json`` is the REVIEWED measured golden (generated against
transformers 5.14.1 / torch 2.13.0 on 2026-08-26 by tracing the committed
``families.py`` builders; regenerate deliberately, never with ``--fix``):

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
# Measured: attn_out is structurally absent under EAGER -- the branch
# mechinterp users deliberately select -- and present under SDPA (memo D7
# class 1, the top-ranked measured class). Lane A01 flips these rows when
# the fused-class gate dies; until then a silent change in either direction
# fails here.
EAGER_SDPA_FACET_DIFF: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "gpt2": {"gpt2_attention": {"sdpa_only": ("attn_out",), "eager_only": ()}},
    "distilgpt2": {"gpt2_attention": {"sdpa_only": ("attn_out",), "eager_only": ()}},
    "llama": {"gqa_attention": {"sdpa_only": ("attn_out",), "eager_only": ()}},
    "distilbert": {"distilbert_attention": {"sdpa_only": ("attn_out",), "eager_only": ()}},
    "bert": {"bert_self_attention": {"sdpa_only": ("attn_out",), "eager_only": ()}},
    # albert/qwen2/vit/clip/whisper: no facet-availability diff between
    # implementations today (their attention carries no recipe, so there is
    # nothing to differ -- the qwen2 emptiness itself is the KNOWN-GAP row).
}

# Families whose upstream class REFUSES sdpa outright (pinned upstream
# behavior, not a TorchLens gap): T5 raises ValueError at construction.
SDPA_UNSUPPORTED_FAMILIES = ("t5",)

# Facet -> expected shape, resolved against FamilySpec.dims at assert time.
# ("b" = batch, "s" = sequence, "h" = hidden, "n" = heads, "d" = d_head.)
# The point is the mask-shape kill: a facet that should be (b, s, h) failing
# because it arrived ids- or mask-shaped is the silent-wrong class the memo
# names. q/k/v pin the MEASURED served layout (b, s, n, d) -- pre-permute --
# deliberately: if A01's per-head work normalizes the layout, this table is
# updated consciously in that lane, never silently.
FACET_SHAPE_SPECS: dict[str, tuple[str, ...]] = {
    "resid_pre": ("b", "s", "h"),
    "resid_mid": ("b", "s", "h"),
    "resid_post": ("b", "s", "h"),
    "q": ("b", "s", "n", "d"),
    "k": ("b", "s", "n", "d"),
    "v": ("b", "s", "n", "d"),
    "attn_out": ("b", "s", "h"),
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

# ENUMERATED WRONG-PAYLOAD MANIFEST (the A02 flagship class, measured at R0
# on 2026-08-26): `resid_pre` is served IDS-SHAPED -- anchored to the token
# ids, not the residual stream -- on these exact rows. gpt2/distilgpt2: both
# blocks, both impls; llama/qwen2: layer 1 only (layer 0 is correct). The
# sweep asserts each row still reproduces its recorded wrong shape (stale
# rows fail loudly); every non-manifest site enforces the honest shape.
# Owner: A02 (semantic residual slice: "resid_pre by dataflow+shape, kills
# the mask-shaped silent wrong read"). Delete rows there when fixed.
WRONG_PAYLOAD_MANIFEST: dict[tuple[str, str, str, str], tuple[int, ...]] = {
    ("gpt2", "eager", "transformer.h.0", "resid_pre"): (1, 8),
    ("gpt2", "eager", "transformer.h.1", "resid_pre"): (1, 8),
    ("gpt2", "sdpa", "transformer.h.0", "resid_pre"): (1, 8),
    ("gpt2", "sdpa", "transformer.h.1", "resid_pre"): (1, 8),
    ("distilgpt2", "eager", "transformer.h.0", "resid_pre"): (1, 8),
    ("distilgpt2", "eager", "transformer.h.1", "resid_pre"): (1, 8),
    ("distilgpt2", "sdpa", "transformer.h.0", "resid_pre"): (1, 8),
    ("distilgpt2", "sdpa", "transformer.h.1", "resid_pre"): (1, 8),
    ("llama", "eager", "model.layers.1", "resid_pre"): (1, 8),
    ("llama", "sdpa", "model.layers.1", "resid_pre"): (1, 8),
    ("qwen2", "eager", "model.layers.1", "resid_pre"): (1, 8),
    ("qwen2", "sdpa", "model.layers.1", "resid_pre"): (1, 8),
}
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
    KnownRed(
        red_id="logit-lens-final-norm",
        owner="A03 (semantic reconstruction slice)",
        exception_type="LogitLensError",
        message_substring="final_norm_kind",
        notes="the flagship logit_lens failure reproduced byte-identically on a"
        " config-built GPT-2 at zero network (memo ground truth); root cause is"
        " the false final_norm_* absence claims in the known-false manifest",
    ),
    KnownRed(
        red_id="container-output-do-replay",
        owner="A05 (FIX-A: replay output contract)",
        exception_type="IndexError",
        message_substring="too many indices",
        notes="fork.do() through the replay engine on a trace whose model returned"
        " an HF ModelOutput container re-applies a recorded container path to an"
        " already-resolved member and crashes (the HF Cache crash root cause)",
    ),
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


def load_expectations() -> dict:
    """Parse the committed measured-golden JSON."""

    return json.loads(EXPECTATIONS_PATH.read_text())
