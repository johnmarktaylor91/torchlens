"""Typed display records for transformer pictures (tviz memo D2).

A small CLOSED family of immutable records -- attention view, token scores,
prediction trajectory, prediction table, token metrics, score decomposition,
causal receipt, annotation, artifact -- carrying coordinates, provenance, and
honesty metadata. One semantic truth feeds matplotlib, the zero-dependency
HTML emitter, and the CircuitsVis bridge without re-derivation (no general
vector IR; the memo's D2 ruling). Records share the site-key/pass coordinate
protocol with the mech-interp kit (:class:`torchlens.mechinterp.Coordinate`);
attachment is by coordinates, never array position.

Records are SESSION-ONLY display values: they are never persisted into
``.tlspec`` artifacts and declare no schema fields (live-only, field-intent
declaration in the lane report). Episode coordinates land in the records from
day one (tviz memo D22) so the wave-2 generation views recompute nothing.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ._errors import refuse
from ._wording import (
    PROVENANCE_RECONSTRUCTED,
    PROVENANCE_USER_SUPPLIED,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ANNOTATION_KINDS",
    "Annotation",
    "Artifact",
    "AttentionView",
    "CausalReceipt",
    "CropInfo",
    "EpisodeCoordinates",
    "GqaInfo",
    "MaskInfo",
    "PredictionTable",
    "PredictionTrajectory",
    "ScoreDecomposition",
    "TokenAxis",
    "TokenMetrics",
    "TokenScoreRow",
    "TokenScores",
]

#: Closed annotation-kind vocabulary (tviz memo D16). ``intervention_effect``
#: is constructible ONLY from a valid :class:`CausalReceipt`.
ANNOTATION_KINDS = (
    "descriptive_head_score",
    "additive_logit_contribution",
    "screening_estimate",
    "intervention_effect",
)

#: Closed provenance vocabulary for attention views (tviz memo D5/D12:
#: pattern source is disclosed separately from effect source).
_PROVENANCE_VOCAB = ("captured", "reconstructed", "user_supplied")

#: Display wording per provenance token (composition row 12: raw-tensor
#: constructor inputs never earn a reconstruction badge).
PROVENANCE_WORDING = {
    "captured": "captured",
    "reconstructed": PROVENANCE_RECONSTRUCTED,
    "user_supplied": PROVENANCE_USER_SUPPLIED,
}

#: Closed mask-provenance vocabulary (tviz memo D6, the three-source
#: hierarchy; inference from zeros is banned and has no token).
_MASK_SOURCES = ("sdpa_call_args", "eager_additive_mask", "user_metadata")

#: Tolerance for the fixed shared probability domain check.
_PROBABILITY_TOL = 1e-4


def _require(condition: bool, message: str, remedy: str, **payload: Any) -> None:
    """Refuse ``tv_record_invalid`` unless ``condition`` holds."""

    if not condition:
        refuse(code="tv_record_invalid", message=message, remedy=remedy, **payload)


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Return the sha256 content fingerprint of a tensor's float32 bytes.

    Parameters
    ----------
    tensor:
        Any real-valued tensor; hashed in float32 on CPU so the fingerprint
        is device- and storage-independent.

    Returns
    -------
    str
        ``"sha256:<hex>"`` over the contiguous float32 byte image plus the
        shape header (two same-byte tensors of different shape differ).
    """

    work = tensor.detach().to(torch.float32).cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(repr(tuple(work.shape)).encode())
    digest.update(work.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class TokenAxis:
    """One labeled token axis (query and key stay SEPARATE records, D5).

    Parameters
    ----------
    role:
        ``"query"``, ``"key"``, or ``"position"``.
    tokens:
        Display strings, one per axis position, never merged.
    ids:
        Optional tokenizer ids aligned with ``tokens``.
    """

    role: str
    tokens: tuple[str, ...]
    ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate role vocabulary and id alignment."""

        _require(
            self.role in ("query", "key", "position"),
            f"TokenAxis role {self.role!r} is not one of ('query', 'key', 'position').",
            "pass role='query' for rows (attend from) or role='key' for columns (attend to)",
        )
        _require(len(self.tokens) > 0, "TokenAxis needs at least one token.", "pass the tokens")
        if self.ids is not None:
            _require(
                len(self.ids) == len(self.tokens),
                f"TokenAxis ids ({len(self.ids)}) and tokens ({len(self.tokens)}) differ.",
                "pass one id per token or ids=None",
            )

    def __len__(self) -> int:
        """Return the axis length."""

        return len(self.tokens)


@dataclass(frozen=True)
class EpisodeCoordinates:
    """Episode coordinates carried by records from day one (tviz memo D22).

    Parameters
    ----------
    step:
        Zero-based generation step (step zero equals a single forward on the
        same prefix -- the addressability contract).
    role:
        ``"prefill"`` or ``"decode"``.
    completion:
        Step-status token from the episode ledger (``"complete"`` /
        ``"interrupted"`` / ``"absent"``).
    member:
        Optional bundle member label when the record came from a Bundle.
    """

    step: int
    role: str = "decode"
    completion: str = "complete"
    member: str | None = None

    def __post_init__(self) -> None:
        """Validate the step and the closed role vocabulary."""

        _require(self.step >= 0, "EpisodeCoordinates.step must be >= 0.", "pass the 0-based step")
        _require(
            self.role in ("prefill", "decode"),
            f"EpisodeCoordinates role {self.role!r} is not 'prefill' or 'decode'.",
            "pass role='prefill' or role='decode'",
        )


@dataclass(frozen=True)
class MaskInfo:
    """Mask provenance for an attention view (tviz memo D6).

    The three-source hierarchy: recorded SDPA call arguments, the captured
    eager additive-mask operand, or explicit user/model metadata. Inference
    from zero-valued pattern cells is BANNED (it passes the canonical causal
    row perfectly -- the scariest kind of wrong); a view without one of these
    sources renders unmarked with the exact unmarked wording.

    Parameters
    ----------
    mask:
        Boolean ``[n_destination, n_source]`` tensor, ``True`` = masked out.
    source:
        One of ``"sdpa_call_args"``, ``"eager_additive_mask"``,
        ``"user_metadata"``.
    """

    mask: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        """Validate mask dtype/rank and the closed source vocabulary."""

        _require(
            self.source in _MASK_SOURCES,
            f"MaskInfo source {self.source!r} is not one of {_MASK_SOURCES}; deriving a "
            "mask from zero-valued pattern cells is banned (tviz memo D6).",
            "pass a mask read from recorded SDPA args, the captured additive-mask "
            "operand, or explicit user metadata",
        )
        _require(
            self.mask.dtype == torch.bool and self.mask.ndim == 2,
            f"MaskInfo.mask must be a 2D bool tensor, got {self.mask.dtype} rank {self.mask.ndim}.",
            "pass mask.bool() shaped [n_destination, n_source]",
        )


@dataclass(frozen=True)
class CropInfo:
    """Crop disclosure: cropping never renormalizes (tviz memo D5).

    Parameters
    ----------
    query_range:
        Kept destination (row) index range ``(start, stop)``.
    key_range:
        Kept source (column) index range ``(start, stop)``.
    omitted_keys:
        Number of source positions cropped away.
    omitted_mass_max:
        Maximum probability mass omitted from any kept destination row --
        measured on the UNCROPPED pattern, so the disclosure is exact.
    """

    query_range: tuple[int, int]
    key_range: tuple[int, int]
    omitted_keys: int
    omitted_mass_max: float

    def disclosure(self) -> str:
        """Return the rendered crop-disclosure line."""

        return (
            f"cropped: keys {self.key_range[0]}..{self.key_range[1] - 1}, "
            f"queries {self.query_range[0]}..{self.query_range[1] - 1}; "
            f"{self.omitted_keys} keys omitted carrying up to "
            f"{self.omitted_mass_max:.3f} attention mass per row; values NOT renormalized"
        )


@dataclass(frozen=True)
class GqaInfo:
    """Grouped-query-attention disclosure (tviz memo D5/D15).

    Parameters
    ----------
    n_query_heads:
        Number of query heads.
    n_kv_heads:
        Number of key/value heads (groups); shared storage is never
        presented as independent heads.
    """

    n_query_heads: int
    n_kv_heads: int

    def __post_init__(self) -> None:
        """Validate the grouping arithmetic."""

        _require(
            self.n_kv_heads > 0 and self.n_query_heads % self.n_kv_heads == 0,
            f"GQA grouping {self.n_query_heads} query heads over {self.n_kv_heads} kv heads "
            "does not divide evenly.",
            "pass the model config's real num_attention_heads / num_key_value_heads",
        )

    def group_of(self, query_head: int) -> int:
        """Return the KV group index a query head reads from."""

        return query_head // (self.n_query_heads // self.n_kv_heads)

    def header(self, query_head: int) -> str:
        """Return the rendered KV-group header line for one query head."""

        from ._wording import KV_GROUP_WORDING

        per_group = self.n_query_heads // self.n_kv_heads
        group = self.group_of(query_head)
        return KV_GROUP_WORDING.format(
            group=group + 1,
            n_groups=self.n_kv_heads,
            first=group * per_group,
            last=(group + 1) * per_group - 1,
        )


def _validate_attention_geometry(view: AttentionView) -> None:
    """Validate pattern rank and token-axis alignment for a view."""

    _require(
        view.pattern.ndim == 3,
        f"AttentionView.pattern must be [n_heads, n_destination, n_source]; got rank "
        f"{view.pattern.ndim}.",
        "select one batch element and stack heads on dim 0",
    )
    _require(
        len(view.heads) == view.pattern.shape[0],
        f"heads tuple ({len(view.heads)}) does not match pattern head dim "
        f"({view.pattern.shape[0]}).",
        "pass the original head index for every pattern row",
    )
    _require(
        view.query_tokens.role == "query" and view.key_tokens.role == "key",
        "query_tokens must have role='query' and key_tokens role='key' (kept separate "
        "even for self-attention; T5 cross-attention is rectangular).",
        "construct TokenAxis(role='query', ...) and TokenAxis(role='key', ...)",
    )
    _require(
        len(view.query_tokens) == view.pattern.shape[1],
        f"query axis length {len(view.query_tokens)} != pattern destination dim "
        f"{view.pattern.shape[1]}.",
        "pass one query token per destination row",
    )
    _require(
        len(view.key_tokens) == view.pattern.shape[2],
        f"key axis length {len(view.key_tokens)} != pattern source dim {view.pattern.shape[2]}.",
        "pass one key token per source column",
    )


def _validate_attention_values(view: AttentionView) -> None:
    """Validate finiteness, the domain vocabulary, and the [0, 1] claim."""

    _require(
        bool(torch.isfinite(view.pattern).all()),
        "AttentionView.pattern contains non-finite values.",
        "fix the capture (a NaN pattern is a capture defect, not a rendering choice)",
    )
    _require(
        view.domain in ("probability", "per_panel"),
        f"AttentionView domain {view.domain!r} is not 'probability' or 'per_panel'.",
        "use the default fixed [0, 1] probability domain; per_panel is expert-only",
    )
    _require(
        view.provenance in _PROVENANCE_VOCAB,
        f"AttentionView provenance {view.provenance!r} is not one of {_PROVENANCE_VOCAB}.",
        "pass 'captured', 'reconstructed', or 'user_supplied'",
    )
    if view.domain == "probability":
        low = float(view.pattern.min())
        high = float(view.pattern.max())
        _require(
            low >= -_PROBABILITY_TOL and high <= 1.0 + _PROBABILITY_TOL,
            f"pattern values [{low:.4g}, {high:.4g}] violate the fixed [0, 1] probability "
            "domain claim.",
            "pass domain='per_panel' (rendered with the 'panels not comparable' label) "
            "for non-probability matrices such as scores",
        )
    if view.mask is not None:
        _require(
            tuple(view.mask.mask.shape) == tuple(view.pattern.shape[1:]),
            f"mask shape {tuple(view.mask.mask.shape)} != pattern cell shape "
            f"{tuple(view.pattern.shape[1:])}.",
            "pass the [n_destination, n_source] mask for THIS view",
        )


@dataclass(frozen=True)
class AttentionView:
    """One attention pattern in canonical coordinates (tviz memo D5).

    Canonical layout is ``[query_head, destination_query, source_key]``;
    every artifact prints "rows attend from, columns attend to". The fixed
    shared ``[0, 1]`` probability domain is the default; ``per_panel`` is
    the expert-only rescale and renders visibly labeled "panels not
    comparable".

    Parameters
    ----------
    pattern:
        ``[n_heads, n_destination, n_source]`` float tensor.
    query_tokens:
        Destination-axis tokens (``role="query"``).
    key_tokens:
        Source-axis tokens (``role="key"``); a separate record even for
        self-attention (T5 cross-attention is rectangular).
    heads:
        Original head indices for each pattern row.
    layer:
        Human layer label (module address in eval mode).
    coordinate:
        Optional shared-protocol coordinate (site key / pass-qualified op
        label) from :class:`torchlens.mechinterp.Coordinate`.
    domain:
        ``"probability"`` (fixed [0, 1]) or ``"per_panel"``.
    provenance:
        ``"captured"`` (eager observed), ``"reconstructed"`` (fused-path
        read-only reconstruction), or ``"user_supplied"`` (raw-tensor
        constructor; never earns a reconstruction badge).
    mask:
        Optional mask provenance; absent renders unmarked with the exact
        unmarked wording (never inferred from zeros).
    crop:
        Optional crop disclosure (cropping never renormalizes).
    gqa:
        Optional grouped-query-attention disclosure.
    episode:
        Optional episode coordinates (tviz memo D22).
    """

    pattern: torch.Tensor
    query_tokens: TokenAxis
    key_tokens: TokenAxis
    heads: tuple[int, ...]
    layer: str
    coordinate: Any | None = None
    domain: str = "probability"
    provenance: str = "captured"
    mask: MaskInfo | None = None
    crop: CropInfo | None = None
    gqa: GqaInfo | None = None
    episode: EpisodeCoordinates | None = None

    def __post_init__(self) -> None:
        """Validate geometry, values, and vocabularies at construction."""

        _validate_attention_geometry(self)
        _validate_attention_values(self)

    @property
    def fingerprint(self) -> str:
        """Return the sha256 content fingerprint of the pattern."""

        return tensor_fingerprint(self.pattern)

    @property
    def provenance_wording(self) -> str:
        """Return the rendered provenance wording for this view."""

        return PROVENANCE_WORDING[self.provenance]

    def head_view(self, head: int) -> AttentionView:
        """Return a single-head view selected by ORIGINAL head index."""

        _require(
            head in self.heads,
            f"head {head} is not in this view's heads {self.heads}.",
            "pass one of the view's original head indices",
        )
        row = self.heads.index(head)
        return AttentionView(
            pattern=self.pattern[row : row + 1],
            query_tokens=self.query_tokens,
            key_tokens=self.key_tokens,
            heads=(head,),
            layer=self.layer,
            coordinate=self.coordinate,
            domain=self.domain,
            provenance=self.provenance,
            mask=self.mask,
            crop=self.crop,
            gqa=self.gqa,
            episode=self.episode,
        )

    def crop_to(self, queries: slice, keys: slice) -> AttentionView:
        """Return a cropped view with an exact omitted-mass disclosure.

        Cropping NEVER renormalizes (tviz memo D5): the omitted key count
        and the maximum omitted per-row mass are measured on the uncropped
        pattern and travel on the returned view's :class:`CropInfo`.

        Parameters
        ----------
        queries:
            Destination (row) slice to keep.
        keys:
            Source (column) slice to keep.

        Returns
        -------
        AttentionView
            The cropped view carrying the crop disclosure.
        """

        n_dst = self.pattern.shape[1]
        n_src = self.pattern.shape[2]
        q_start, q_stop, q_step = queries.indices(n_dst)
        k_start, k_stop, k_step = keys.indices(n_src)
        _require(
            q_step == 1 and k_step == 1,
            "crop_to only supports contiguous slices (step 1).",
            "pass plain start:stop slices",
        )
        _require(
            q_stop > q_start and k_stop > k_start,
            "crop_to produced an empty view.",
            "pass non-empty query/key ranges",
        )
        kept = self.pattern[:, q_start:q_stop, :]
        omitted_mass = kept.sum(dim=-1) - kept[:, :, k_start:k_stop].sum(dim=-1)
        crop = CropInfo(
            query_range=(q_start, q_stop),
            key_range=(k_start, k_stop),
            omitted_keys=n_src - (k_stop - k_start),
            omitted_mass_max=float(omitted_mass.max()) if omitted_mass.numel() else 0.0,
        )
        mask = self.mask
        if mask is not None:
            mask = MaskInfo(mask=mask.mask[q_start:q_stop, k_start:k_stop], source=mask.source)
        return AttentionView(
            pattern=kept[:, :, k_start:k_stop],
            query_tokens=TokenAxis(
                role="query",
                tokens=self.query_tokens.tokens[q_start:q_stop],
                ids=None
                if self.query_tokens.ids is None
                else self.query_tokens.ids[q_start:q_stop],
            ),
            key_tokens=TokenAxis(
                role="key",
                tokens=self.key_tokens.tokens[k_start:k_stop],
                ids=None if self.key_tokens.ids is None else self.key_tokens.ids[k_start:k_stop],
            ),
            heads=self.heads,
            layer=self.layer,
            coordinate=self.coordinate,
            domain=self.domain,
            provenance=self.provenance,
            mask=mask,
            crop=crop,
            gqa=self.gqa,
            episode=self.episode,
        )


@dataclass(frozen=True)
class TokenScoreRow:
    """One labeled row of per-token scores (strip row).

    Parameters
    ----------
    label:
        Row label (method name, factor index, metric name).
    scores:
        One float per token; ``None`` marks a not-applicable cell (rendered
        N/A, never zero).
    """

    label: str
    scores: tuple[float | None, ...]


@dataclass(frozen=True)
class TokenScores:
    """Colored-token strip record, single- or multi-row (tviz memo roster).

    The multi-row form serves the NMF factor view and any per-method
    comparison strip; all rows share ONE token axis.

    Parameters
    ----------
    tokens:
        Display tokens shared by every row.
    rows:
        Score rows; equal length to ``tokens``.
    domain:
        ``"zero_centered_diverging"`` (signed scores),
        ``"unit_interval"``, or ``"magnitude_sequential"``.
    footer_lines:
        Disclosure lines the renderer MUST show (target, method, baseline,
        completeness -- whatever the producer contracted).
    provenance:
        Producer description (method + source).
    ids:
        Optional tokenizer ids aligned with ``tokens``.
    """

    tokens: tuple[str, ...]
    rows: tuple[TokenScoreRow, ...]
    domain: str = "zero_centered_diverging"
    footer_lines: tuple[str, ...] = ()
    provenance: str = "user_supplied"
    ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate row alignment and the domain vocabulary."""

        _require(len(self.rows) > 0, "TokenScores needs at least one row.", "pass rows")
        for row in self.rows:
            _require(
                len(row.scores) == len(self.tokens),
                f"row {row.label!r} has {len(row.scores)} scores for {len(self.tokens)} tokens.",
                "align every row to the shared token axis",
            )
            _require(
                all(score is None or math.isfinite(score) for score in row.scores),
                f"row {row.label!r} contains non-finite scores.",
                "pass None for not-applicable cells (rendered N/A, never zero)",
            )
        _require(
            self.domain in ("zero_centered_diverging", "unit_interval", "magnitude_sequential"),
            f"TokenScores domain {self.domain!r} unknown.",
            "pass one of the three documented score domains",
        )

    @classmethod
    def from_attribution(cls, payload: Any) -> TokenScores:
        """Build a strip record from an attribution token payload (D28).

        Parameters
        ----------
        payload:
            A :class:`torchlens.attribution.TokenAttributionPayload`-shaped
            object (``display_tokens``, ``scores``, ``footer_lines``,
            ``score_domain`` attributes).

        Returns
        -------
        TokenScores
            Single-row strip carrying the payload's footer disclosures.
        """

        return cls(
            tokens=tuple(payload.display_tokens),
            rows=(
                TokenScoreRow(
                    label="attribution",
                    scores=tuple(float(score) for score in payload.scores),
                ),
            ),
            domain=str(getattr(payload, "score_domain", "zero_centered_diverging")),
            footer_lines=tuple(payload.footer_lines),
            provenance="torchlens.attribution token payload",
        )


@dataclass(frozen=True)
class PredictionTrajectory:
    """Per-layer prediction trajectory record (Ecco layer predictions, matched).

    Built from the streaming lens result; provenance wording travels row by
    row so an intermediate projection can never claim to be the native
    output (tviz memo D9 boundary; UNVALIDATED-LENS-AS-NATIVE sentinel).

    Parameters
    ----------
    layers:
        Layer labels (module addresses) in execution order.
    provenance:
        Per-layer provenance wording (``"projected through final norm/head"``
        or ``"native output"``).
    position:
        The inspected sequence position (absolute).
    top_tokens:
        ``[n_layers][k]`` decoded top tokens per layer.
    top_probs:
        ``[n_layers][k]`` full-vocabulary-denominator probabilities.
    target_token:
        Optional tracked answer token string.
    target_probs / target_ranks:
        Per-layer probability and ONE-BASED rank of the tracked token.
    lens_source:
        ``"model_head"`` or ``"user"``.
    validated:
        Whether the reconstructed lens passed numeric validation; ``False``
        for user lenses, rendered as unvalidated.
    tie_convention:
        Recorded rank tie convention.
    """

    layers: tuple[str, ...]
    provenance: tuple[str, ...]
    position: int
    top_tokens: tuple[tuple[str, ...], ...]
    top_probs: tuple[tuple[float, ...], ...]
    target_token: str | None = None
    target_probs: tuple[float, ...] | None = None
    target_ranks: tuple[int, ...] | None = None
    lens_source: str = "model_head"
    validated: bool = False
    tie_convention: str = ""

    def __post_init__(self) -> None:
        """Validate per-layer alignment."""

        n_layers = len(self.layers)
        for name in ("provenance", "top_tokens", "top_probs"):
            _require(
                len(getattr(self, name)) == n_layers,
                f"PredictionTrajectory.{name} has {len(getattr(self, name))} entries for "
                f"{n_layers} layers.",
                "align every per-layer field to layers",
            )
        for name in ("target_probs", "target_ranks"):
            value = getattr(self, name)
            _require(
                value is None or len(value) == n_layers,
                f"PredictionTrajectory.{name} misaligned with layers.",
                "align every per-layer field to layers",
            )


@dataclass(frozen=True)
class PredictionTable:
    """Top-k tokens per position (inspectus token table, matched).

    Parameters
    ----------
    positions:
        Absolute sequence positions, whole positions only (pagination policy
        lives in the renderer).
    context_tokens:
        The input token at each position.
    top_tokens:
        ``[n_positions][k]`` predicted tokens.
    top_probs:
        ``[n_positions][k]`` full-vocabulary-denominator probabilities.
    provenance:
        Source wording (native output vs projected).
    """

    positions: tuple[int, ...]
    context_tokens: tuple[str, ...]
    top_tokens: tuple[tuple[str, ...], ...]
    top_probs: tuple[tuple[float, ...], ...]
    provenance: str = "native output"

    def __post_init__(self) -> None:
        """Validate positional alignment."""

        n_positions = len(self.positions)
        for name in ("context_tokens", "top_tokens", "top_probs"):
            _require(
                len(getattr(self, name)) == n_positions,
                f"PredictionTable.{name} misaligned with positions.",
                "provide one entry per position",
            )


@dataclass(frozen=True)
class TokenMetrics:
    """Per-token metric strip record (loss / entropy; inspectus matched).

    Parameters
    ----------
    tokens:
        Display tokens.
    values:
        One value per token; ``None`` = not applicable (the first token has
        no loss; padding has neither) -- rendered N/A, never zero.
    metric:
        ``"loss"`` or ``"entropy"``.
    convention:
        The recorded alignment convention (e.g. the loss shift direction).
    source_fingerprint:
        Fingerprint of the ONE logits tensor both metrics were computed
        from -- the position-alignment proof travels with the record.
    """

    tokens: tuple[str, ...]
    values: tuple[float | None, ...]
    metric: str
    convention: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        """Validate metric vocabulary and alignment."""

        _require(
            self.metric in ("loss", "entropy"),
            f"TokenMetrics metric {self.metric!r} is not 'loss' or 'entropy'.",
            "pass metric='loss' or metric='entropy'",
        )
        _require(
            len(self.values) == len(self.tokens),
            f"TokenMetrics has {len(self.values)} values for {len(self.tokens)} tokens.",
            "align values with tokens; use None for N/A cells",
        )
        _require(
            all(value is None or math.isfinite(value) for value in self.values),
            "TokenMetrics contains non-finite values.",
            "pass None for not-applicable cells",
        )


@dataclass(frozen=True)
class ScoreDecomposition:
    """Term-complete score decomposition for one (head, dst, src) pair (D19).

    The BertViz neuron-view DATA record: post-transform q/k vectors, their
    per-dimension products, the scale, and every additive term, with a HARD
    sum-to-score invariant -- construction refuses rather than
    mis-decomposing.

    Parameters
    ----------
    layer:
        Layer label (module address).
    head:
        Query head index.
    destination / source:
        Query (row) and key (column) token positions.
    query_vector / key_vector:
        Post-transform per-head vectors, ``[d_head]``.
    products:
        Elementwise ``q_i * k_i`` products, ``[d_head]``.
    scale:
        The score scale divisor (``sqrt(d_head)`` in the vanilla family).
    additive_terms:
        Named additive contributions applied after the scaled dot product
        (``"mask"``, ``"relative_bias"``, ...). RoPE families whose terms do
        not close refuse per-family upstream.
    reference_score:
        The captured scores-facet value this decomposition claims to equal.
    tolerance:
        Closure tolerance used by the invariant.
    """

    layer: str
    head: int
    destination: int
    source: int
    query_vector: torch.Tensor
    key_vector: torch.Tensor
    products: torch.Tensor
    scale: float
    additive_terms: dict[str, float] = field(default_factory=dict)
    reference_score: float = 0.0
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        """Enforce the hard sum-to-score invariant (refuse, never mislead)."""

        total = self.total
        error = abs(total - self.reference_score)
        scale_ref = max(1.0, abs(self.reference_score))
        if error > self.tolerance * scale_ref:
            refuse(
                code="tv_decomposition_unclosed",
                message=(
                    f"Score decomposition terms sum to {total:.6g} but the captured score "
                    f"is {self.reference_score:.6g} (|err|={error:.3g} > tol) at layer "
                    f"{self.layer!r} head {self.head} ({self.destination}<-{self.source}). "
                    "The terms do not reproduce the score facet, so the record refuses "
                    "rather than mis-decomposing (tviz memo D19)."
                ),
                remedy=(
                    "this family's score path has terms outside the vanilla "
                    "dot-product/scale/additive vocabulary (RoPE and friends refuse "
                    "per-family until term extraction covers them); decompose a "
                    "gpt2/bert-class head, or extend the family terms"
                ),
                total=total,
                reference_score=self.reference_score,
                error=error,
            )

    @property
    def total(self) -> float:
        """Return the reproduced score: ``sum(products)/scale + additives``."""

        return float(self.products.sum()) / self.scale + sum(self.additive_terms.values())


def _validate_receipt_controls(receipt: CausalReceipt) -> None:
    """Refuse ``tv_receipt_invalid`` unless both controls pass (D12)."""

    if receipt.fires <= 0:
        refuse(
            code="tv_receipt_invalid",
            message="Receipt records ZERO intervention fires; zero fires is a typed error, "
            "never a zero effect (tviz memo D12; the silent-no-fire class).",
            remedy="run the cells through a runner with a wrapping fire-counting mutator",
            fires=receipt.fires,
        )
    if abs(receipt.negative_control) > receipt.negative_control_tol:
        refuse(
            code="tv_receipt_invalid",
            message=f"Negative control (self-patch) moved the metric by "
            f"{receipt.negative_control:+.3g} (tol {receipt.negative_control_tol:.3g}); the "
            "measurement machinery is not inert.",
            remedy="fix the runner: a self-patch must not move the metric",
            negative_control=receipt.negative_control,
        )
    if abs(receipt.positive_control) < receipt.positive_control_min:
        refuse(
            code="tv_receipt_invalid",
            message=f"Positive control moved the metric by only "
            f"{receipt.positive_control:+.3g} (< {receipt.positive_control_min:.3g}); a "
            "deliberately large perturbation MUST move the metric -- this is the only "
            "check that catches the silent no-fire class (tviz memo D12).",
            remedy="verify the intervention route actually fires before attaching effects",
            positive_control=receipt.positive_control,
        )


@dataclass(frozen=True)
class CausalReceipt:
    """Validated measured-intervention evidence for annotation (D11/D12).

    tviz owns ATTACHMENT validation: a receipt refuses at construction on
    zero fires, a moved negative control, or an unmoved positive control.
    Effects are per original head index; an unmeasured head is ``None``
    (rendered blank, never zero).

    Parameters
    ----------
    layer:
        Layer label the effects attach to (coordinate join, never array
        position).
    heads:
        Original head indices, aligned with ``effects``.
    effects:
        Measured metric deltas per head; ``None`` = unmeasured.
    metric:
        Metric wording (printed on the figure).
    intervention:
        Intervention wording (e.g. "head contribution zeroed", never
        "attention module removed").
    engine:
        Executing engine (``"rerun"`` capture route, ``"shadow"``,
        ``"lowered"``).
    disclosure:
        ``"direct"`` or the exact output-equivalent wording for lowered
        engines.
    fires:
        Total positive fire count from the runner's wrapping mutator.
    negative_control:
        Measured self-patch metric delta (must be ~0).
    positive_control:
        Measured large-perturbation metric delta (must move).
    joint_effect:
        Measured effect of ablating ALL displayed heads together (one extra
        forward); REQUIRED by the grid figure (tviz memo D13).
    negative_control_tol / positive_control_min:
        Control thresholds, printed with the controls.
    pattern_source / effect_source:
        Disclosed separately (a fused pattern joined to an eager effect is
        a TWO-RUN figure and says so).
    fingerprints:
        Content fingerprints for the joined artifacts.
    """

    layer: str
    heads: tuple[int, ...]
    effects: tuple[float | None, ...]
    metric: str
    intervention: str
    engine: str
    disclosure: str
    fires: int
    negative_control: float
    positive_control: float
    joint_effect: float | None = None
    negative_control_tol: float = 1e-6
    positive_control_min: float = 1e-3
    pattern_source: str = "captured"
    effect_source: str = "capture route"
    fingerprints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment, finiteness, and both mandatory controls."""

        _require(
            len(self.effects) == len(self.heads),
            f"CausalReceipt has {len(self.effects)} effects for {len(self.heads)} heads.",
            "align effects with heads; use None for unmeasured heads",
        )
        for effect in self.effects:
            if effect is not None and not math.isfinite(effect):
                refuse(
                    code="tv_receipt_invalid",
                    message="A receipt effect is non-finite.",
                    remedy="a non-finite metric is a failed measurement; re-run the cell",
                )
        _validate_receipt_controls(self)

    @property
    def measured_n(self) -> int:
        """Return how many displayed heads carry a measured effect."""

        return sum(1 for effect in self.effects if effect is not None)

    @property
    def cell_sum(self) -> float:
        """Return the sum of measured single-head effects."""

        return sum(effect for effect in self.effects if effect is not None)


@dataclass(frozen=True)
class Annotation:
    """One typed annotation attached to an attention picture (D16).

    The closed four-kind vocabulary keeps descriptive scores from quietly
    becoming causal claims: each kind renders with its own glyph and legend
    section, and only a valid :class:`CausalReceipt` can mint
    ``intervention_effect``.

    Parameters
    ----------
    kind:
        One of :data:`ANNOTATION_KINDS`.
    heads:
        Original head indices, aligned with ``values``.
    values:
        Per-head numbers; ``None`` = unmeasured (blank, never zero).
    source:
        Producer wording rendered in the legend.
    legend:
        Kind legend line.
    receipt:
        The validating receipt; REQUIRED for ``intervention_effect``.
    """

    kind: str
    heads: tuple[int, ...]
    values: tuple[float | None, ...]
    source: str
    legend: str = ""
    receipt: CausalReceipt | None = None

    def __post_init__(self) -> None:
        """Validate the closed kind vocabulary and the receipt gate."""

        if self.kind not in ANNOTATION_KINDS:
            refuse(
                code="tv_annotation_invalid",
                message=f"Annotation kind {self.kind!r} is not in the closed set "
                f"{ANNOTATION_KINDS}.",
                remedy="pick the kind that states what the number IS; there is no "
                "'just a number painted on a head'",
                kind=self.kind,
            )
        if self.kind == "intervention_effect" and self.receipt is None:
            refuse(
                code="tv_annotation_invalid",
                message="intervention_effect annotations require a valid CausalReceipt; "
                "only measured interventions may claim causal wording (tviz memo D16).",
                remedy="attach the receipt from the runner, or use kind="
                "'descriptive_head_score' / 'screening_estimate' for non-causal numbers",
            )
        _require(
            len(self.values) == len(self.heads),
            f"Annotation has {len(self.values)} values for {len(self.heads)} heads.",
            "align values with heads; use None for unmeasured heads",
        )

    @classmethod
    def from_receipt(cls, receipt: CausalReceipt) -> Annotation:
        """Mint the ``intervention_effect`` annotation from a valid receipt."""

        return cls(
            kind="intervention_effect",
            heads=receipt.heads,
            values=receipt.effects,
            source=(
                f"{receipt.intervention}; metric: {receipt.metric}; engine: "
                f"{receipt.engine} ({receipt.disclosure})"
            ),
            legend="measured intervention effect",
            receipt=receipt,
        )


@dataclass(frozen=True)
class Artifact:
    """The result of saving a picture: paths + everything the figure said.

    Parameters
    ----------
    paths:
        Written files in page order (pagination never silently drops;
        every page is listed).
    format:
        ``"png"`` / ``"svg"`` / ``"pdf"`` / ``"html"``.
    title:
        Figure title as rendered.
    disclosure_lines:
        Every honesty line the artifact rendered, in order.
    provenance:
        Data provenance wording.
    fingerprint:
        Content fingerprint of the underlying record.
    svg_fonttype:
        The matplotlib ``svg.fonttype`` used (``"path"`` default; ``"none"``
        is the editing opt-in), recorded per the memo's manifest rule.
    pages_total:
        Total page count (equals ``len(paths)`` for file output).
    """

    paths: tuple[Path, ...]
    format: str
    title: str
    disclosure_lines: tuple[str, ...]
    provenance: str
    fingerprint: str
    svg_fonttype: str | None = None
    pages_total: int = 1
