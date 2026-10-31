"""Contrastive directions built natively from paired traces.

:func:`direction_from_pairs` reads one row per prompt at one site from the
positive and the negative traces and fits a direction with one of three
methods. The math lives here only: the steering-vectors, repeng and dialz
bridges convert the returned :class:`Direction` to their library objects
without recomputing anything.

Methods (rows are ``[n_pairs, hidden]``, ``D = positive - negative``):

- ``"mean_diff"``: ``D.mean(0)``, on the activation scale (steering-vectors'
  ``mean_aggregator``).
- ``"pca"``: the first principal axis of the paired differences, uncentered
  (equivalently PCA of ``D`` and ``-D`` stacked), unit norm, signed so the
  mean projection of ``D`` is positive (steering-vectors' ``pca_aggregator``).
- ``"logistic"``: L2-regularised logistic regression (``C = 1``, no
  intercept) separating the pair-centred rows ``D / 2`` from ``-D / 2``,
  solved by Newton's method in float64 with torch only, unit norm
  (steering-vectors' ``logistic_aggregator`` with its scikit-learn defaults).

The provenance record is the subspace producer's
:class:`~torchlens.selection_subspace.BasisProvenance` (origin, method,
geometry and the sha256 content digest of the canonical float64 vector); the
origin also records the site, pair count, read position and a digest of the
rows the direction was fitted on.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._errors import InvalidArgumentError, PayloadUnavailableError
from ..selection_subspace import BasisProvenance, _basis_digest

if TYPE_CHECKING:
    from .types import Edit

METHODS = ("mean_diff", "pca", "logistic")
_LOGISTIC_MAX_ITER = 100
_LOGISTIC_TOL = 1e-10
_TRAILING_INDEX = re.compile(r"\.(\d+)$")

ReadIndex = int | Sequence[int] | None


@dataclass(frozen=True, eq=False)
class Direction:
    """One contrastive direction at one site, with its provenance.

    Parameters
    ----------
    vector:
        The direction, a 1-D tensor over the site's trailing feature axis, in
        the dtype of the rows it was fitted on.
    site:
        Label of the site the rows were read at.
    method:
        ``"mean_diff"``, ``"pca"`` or ``"logistic"``.
    n_pairs:
        Number of positive/negative row pairs.
    read_token_index:
        Token position(s) read per prompt, or ``None`` for unsliced outs.
    model_type:
        ``config.model_type`` of the traced model, or ``None`` when it has none.
    provenance:
        Origin, method, geometry and content digest of the vector.
    """

    vector: torch.Tensor
    site: str
    method: str
    n_pairs: int
    read_token_index: ReadIndex
    model_type: str | None
    provenance: BasisProvenance

    def __repr__(self) -> str:
        """Return a one-line summary without the tensor values."""

        return (
            f"Direction(site={self.site!r}, method={self.method!r}, n_pairs={self.n_pairs}, "
            f"dim={self.vector.shape[0]}, sha256={self.provenance.sha256[:16]})"
        )

    def steer(self, magnitude: float = 1.0) -> Edit:
        """Return the edit that adds ``magnitude * vector`` on the trailing axis.

        Parameters
        ----------
        magnitude:
            Scalar multiplier.

        Returns
        -------
        Edit
            ``torchlens.steer(vector, magnitude, feature_axis=-1)``; place it
            with ``tl.when(tl.module(direction.site), ...)``.
        """

        from .helpers import steer

        return steer(self.vector, magnitude, feature_axis=-1)

    def to_steering_vectors(
        self, *, layer: int | None = None, layer_type: str = "decoder_block"
    ) -> Any:
        """Return a ``steering_vectors.SteeringVector`` holding this vector.

        Parameters
        ----------
        layer:
            Decoder layer the library patches. Defaults to the trailing index
            of the site label (``"model.layers.3"`` is layer 3).
        layer_type:
            steering-vectors layer type (default ``"decoder_block"``).

        Returns
        -------
        Any
            A real ``steering_vectors.SteeringVector``.
        """

        from ..bridge.steering_vectors import from_direction

        return from_direction(self, layer=layer, layer_type=layer_type)

    def to_repeng(self, *, layer: int | None = None, model_type: str | None = None) -> Any:
        """Return a ``repeng.ControlVector`` holding this vector.

        Parameters
        ----------
        layer:
            Decoder layer ``ControlModel`` applies it to. Defaults to the
            trailing index of the site label.
        model_type:
            ``ControlVector.model_type``; defaults to :attr:`model_type`.

        Returns
        -------
        Any
            A real ``repeng.ControlVector``.
        """

        from ..bridge.repeng import from_direction

        return from_direction(self, layer=layer, model_type=model_type)

    def upload_to_neuronpedia(  # noqa: PLR0913 -- the endpoint's fields, keyword-only
        self,
        *,
        api_key: str,
        scan: str,
        vector_label: str,
        layer: int | None = None,
        hook_type: str = "resid_post",
        steer_strength: float = 1.0,
    ) -> str:
        """Upload this vector to Neuronpedia and return its page's URL.

        Documented-unstable. This publishes the vector on neuronpedia.org
        under the account that owns ``api_key``, through Neuronpedia's
        documented "Create Vector" endpoint (``POST /api/vector/new``,
        https://www.neuronpedia.org/api-doc). Every field is checked before
        the request; the key is sent in the ``x-api-key`` header and never
        written into a message.

        Parameters
        ----------
        api_key:
            Your Neuronpedia API key (under Settings on neuronpedia.org).
        scan:
            Neuronpedia's model id, for example ``"gemma-2-2b"``.
        vector_label:
            The name Neuronpedia shows for the vector.
        layer:
            Decoder layer (0-based). Defaults to the trailing index of the
            site label, as :meth:`to_repeng` reads it.
        hook_type:
            ``"resid_post"`` (a decoder block's output, the default) or
            ``"resid_pre"`` (its input).
        steer_strength:
            The default steering strength Neuronpedia offers, from -100 to 100.

        Returns
        -------
        str
            The URL of the vector's page on Neuronpedia.

        Raises
        ------
        AttributionError
            ``neuronpedia_upload_argument_invalid`` for a missing key or a
            field the endpoint refuses; ``neuronpedia_upload_http_error`` and
            ``neuronpedia_upload_response_invalid`` as for
            ``tl.attribution.graph.upload_to_neuronpedia``.
        InvalidArgumentError
            As :meth:`library_layer`, when ``layer`` is ``None`` and the site
            label ends in no index.
        """

        from ..attribution.graph._neuronpedia_upload import upload_vector

        return upload_vector(
            self.vector,
            api_key=api_key,
            scan=scan,
            layer=self.library_layer(layer),
            hook_type=hook_type,
            vector_label=vector_label,
            steer_strength=steer_strength,
        )

    def to_dialz(self, *, layer: int | None = None, model_type: str | None = None) -> Any:
        """Return a ``dialz.SteeringVector`` holding this vector.

        Parameters
        ----------
        layer:
            Decoder layer ``SteeringModel`` applies it to. Defaults to the
            trailing index of the site label.
        model_type:
            ``SteeringVector.model_type``; defaults to :attr:`model_type`.

        Returns
        -------
        Any
            A real ``dialz.SteeringVector``.
        """

        from ..bridge.dialz import from_direction

        return from_direction(self, layer=layer, model_type=model_type)

    def library_layer(self, layer: int | None = None) -> int:
        """Return the decoder layer index a steering library applies this at.

        Parameters
        ----------
        layer:
            Explicit layer index, or ``None`` to read it from the site label.

        Returns
        -------
        int
            The layer index.

        Raises
        ------
        InvalidArgumentError
            ``direction_layer_unknown`` when the site label ends in no index
            and no layer is given; ``direction_layer_mismatch`` when the given
            layer differs from the site's own index.
        """

        match = _TRAILING_INDEX.search(self.site)
        if layer is None:
            if match is None:
                raise InvalidArgumentError(
                    f"Site {self.site!r} names no decoder layer index",
                    code="direction_layer_unknown",
                    remedy="pass layer= (the decoder block the library should patch)",
                )
            return int(match.group(1))
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise InvalidArgumentError(
                f"layer must be an int; got {layer!r}",
                code="direction_layer_unknown",
                remedy="pass layer= as the decoder block index",
            )
        if match is not None and int(match.group(1)) != layer:
            raise InvalidArgumentError(
                f"layer={layer} differs from site {self.site!r}, which is layer {match.group(1)}",
                code="direction_layer_mismatch",
                remedy="omit layer= to patch the layer the direction was read at",
            )
        return layer


def direction_from_pairs(
    pos_traces: Any,
    neg_traces: Any,
    site: Any,
    method: str = "mean_diff",
    *,
    read_token_index: ReadIndex = -1,
) -> Direction:
    """Fit a contrastive direction from positive and negative traces.

    Parameters
    ----------
    pos_traces:
        One ``Trace`` or a sequence of traces of the positive prompts, one
        prompt per batch row. Rows are concatenated in order.
    neg_traces:
        The negative prompts, laid out the same way: row ``i`` of the
        negatives pairs with row ``i`` of the positives.
    site:
        Site label (or a layer record, whose label is resolved in every
        trace) whose saved out holds ``[n_prompts, n_tokens, hidden]``
        activations, or ``[n_prompts, hidden]`` with ``read_token_index=None``.
    method:
        ``"mean_diff"`` (default), ``"pca"`` or ``"logistic"``; see the module
        docstring for the exact math.
    read_token_index:
        Token position read from every prompt (default ``-1``, the last).
        When a trace saved its ``attention_mask`` input, positions count
        within each prompt's real tokens, so ``-1`` is the last real token of
        a padded prompt. A sequence gives one position per prompt across all
        traces; ``None`` reads the outs unsliced.

    Returns
    -------
    Direction
        The fitted direction with its provenance record.

    Raises
    ------
    InvalidArgumentError
        ``direction_method_unknown``, ``direction_pairs_empty``,
        ``direction_pairs_mismatch``, ``direction_pairs_identical``,
        ``direction_site_unsaved``, ``direction_read_token_rank``,
        ``direction_read_token_count``, ``direction_mask_shape``,
        ``direction_mask_empty_row`` or ``direction_logistic_unconverged``.
    """

    if method not in METHODS:
        raise InvalidArgumentError(
            f"Unknown method {method!r}",
            code="direction_method_unknown",
            remedy=f"pass method= one of {list(METHODS)}",
        )
    positives = _as_traces(pos_traces, "positive")
    negatives = _as_traces(neg_traces, "negative")
    label = _site_label(site) if _is_record(site) else site
    positive = _side_rows(positives, label, read_token_index, "positive")
    negative = _side_rows(negatives, label, read_token_index, "negative")
    _check_pairs(positive, negative)
    vector = _fit(positive, negative, method)
    rows_digest = hashlib.sha256()
    for rows in (positive, negative):
        rows_digest.update(repr(tuple(rows.shape)).encode())
        rows_digest.update(rows.detach().cpu().double().numpy().tobytes())
    site_name = str(label)
    origin = (
        f"direction_from_pairs(site={site_name!r}, method={method!r}, "
        f"n_pairs={positive.shape[0]}, read_token_index={read_token_index!r}, "
        f"pairs_sha256={rows_digest.hexdigest()})"
    )
    canonical = vector.detach().cpu().to(torch.float64).reshape(1, -1)
    provenance = BasisProvenance(
        origin=origin,
        method=method,
        sha256=_basis_digest(canonical),
        num_vectors=1,
        space_dim=int(canonical.shape[1]),
        dim=-1,
        atol=0.0,
    )
    return Direction(
        vector=vector,
        site=site_name,
        method=method,
        n_pairs=int(positive.shape[0]),
        read_token_index=read_token_index,
        model_type=_model_type(positives[0]),
        provenance=provenance,
    )


def _as_traces(traces: Any, side: str) -> list[Any]:
    """Return a non-empty list of traces from one trace or a sequence."""

    found = list(traces) if isinstance(traces, (list, tuple)) else [traces]
    if not found:
        raise InvalidArgumentError(
            f"No {side} traces were given",
            code="direction_pairs_empty",
            remedy=f"pass at least one trace of the {side} prompts",
        )
    return found


def _is_record(site: Any) -> bool:
    """Return whether ``site`` is a layer or op record rather than a spelling."""

    return hasattr(site, "out") and hasattr(site, "layer_label")


def _site_label(record: Any) -> str:
    """Return the label naming one record in another trace of the same model."""

    label = str(record.layer_label)
    num_passes = getattr(record, "num_passes", None)
    pass_index = getattr(record, "pass_index", None)
    if isinstance(num_passes, int) and num_passes > 1 and isinstance(pass_index, int):
        return f"{label}:{pass_index}"
    return label


def _saved_out(trace: Any, site: Any, side: str) -> torch.Tensor:
    """Return the saved tensor out of ``site`` in one trace."""

    from .errors import SiteResolutionError

    try:
        record = trace.resolve_sites(site, max_fanout=1).first()
    except SiteResolutionError:
        if not isinstance(site, str):
            raise
        record = trace[site]
    try:
        out = getattr(record, "out", None)
    except PayloadUnavailableError:
        out = None
    if not isinstance(out, torch.Tensor):
        raise InvalidArgumentError(
            f"The {side} site {site!r} has no saved tensor out",
            code="direction_site_unsaved",
            remedy="trace with the site saved (for example layers_to_save='all')",
        )
    return out


def _captured_mask(trace: Any) -> torch.Tensor | None:
    """Return the saved ``attention_mask`` input of a trace, if any."""

    for layer in getattr(trace, "layer_list", []):
        role = getattr(layer, "io_role", None)
        if not getattr(layer, "is_input", False) or not isinstance(role, str):
            continue
        out = getattr(layer, "out", None)
        if role.split(".")[-1] == "attention_mask" and isinstance(out, torch.Tensor):
            return out
    return None


def _side_rows(traces: list[Any], site: Any, index: ReadIndex, side: str) -> torch.Tensor:
    """Read and concatenate one row per prompt from every trace of one side."""

    outs = [_saved_out(trace, site, side) for trace in traces]
    total = sum(int(out.shape[0]) for out in outs)
    positions = None if index is None or isinstance(index, int) else list(index)
    if positions is not None and len(positions) != total:
        raise InvalidArgumentError(
            f"read_token_index lists {len(positions)} positions for {total} {side} prompts",
            code="direction_read_token_count",
            remedy="pass one int for every prompt, or one position per prompt",
        )
    rows, start = [], 0
    for trace, out in zip(traces, outs, strict=True):
        count = int(out.shape[0])
        chunk = index if positions is None else positions[start : start + count]
        rows.append(_read_rows(out, chunk, _captured_mask(trace), side))
        start += count
    shapes = {tuple(row.shape[1:]) for row in rows}
    if len(shapes) != 1:
        raise InvalidArgumentError(
            f"The {side} traces read rows of different widths {sorted(shapes)}",
            code="direction_pairs_mismatch",
            remedy="trace every prompt with the same model at the same site",
        )
    return torch.cat(rows, dim=0)


def _read_rows(
    out: torch.Tensor, index: ReadIndex, mask: torch.Tensor | None, side: str
) -> torch.Tensor:
    """Slice one token per prompt out of a ``[n, tokens, hidden]`` out."""

    out = out.detach()
    if index is None:
        return out
    if out.dim() < 3:
        raise InvalidArgumentError(
            f"The {side} out has shape {tuple(out.shape)}; reading a token needs "
            "[n_prompts, n_tokens, hidden]",
            code="direction_read_token_rank",
            remedy="pass read_token_index=None for outs that are already one row per prompt",
        )
    if isinstance(index, int):
        indices = torch.full((out.shape[0],), index, dtype=torch.long, device=out.device)
    else:
        indices = torch.as_tensor(list(index), dtype=torch.long, device=out.device)
    if mask is not None:
        indices = _adjust_for_padding(indices, mask, tuple(out.shape[:2]), side)
    return out[torch.arange(out.shape[0], device=out.device), indices]


def _adjust_for_padding(
    indices: torch.Tensor, mask: torch.Tensor, rows_tokens: tuple[int, ...], side: str
) -> torch.Tensor:
    """Map positions within each prompt's real tokens to absolute positions."""

    real = mask.to(indices.device) == 1
    if tuple(real.shape) != rows_tokens:
        raise InvalidArgumentError(
            f"The {side} attention_mask has shape {tuple(real.shape)}; the out "
            f"reads [n_prompts, n_tokens] = {rows_tokens}",
            code="direction_mask_shape",
            remedy="trace the prompts with the attention_mask they were tokenized with",
        )
    if not bool(real.any(dim=1).all()):
        raise InvalidArgumentError(
            f"A {side} prompt has no unmasked token in its attention_mask",
            code="direction_mask_empty_row",
            remedy="drop empty prompts before tracing",
        )
    positions = torch.arange(real.shape[1], device=indices.device).expand_as(real)
    start = torch.where(real, positions, real.shape[1]).min(dim=1).values
    end = torch.where(real, positions, -1).max(dim=1).values
    return torch.where(indices < 0, indices + (end - start + 1), indices) + start


def _check_pairs(positive: torch.Tensor, negative: torch.Tensor) -> None:
    """Refuse rows that do not pair up or are identical."""

    if positive.shape != negative.shape or positive.dim() != 2:
        raise InvalidArgumentError(
            f"Positive rows {tuple(positive.shape)} and negative rows "
            f"{tuple(negative.shape)} must be matching [n_pairs, hidden]",
            code="direction_pairs_mismatch",
            remedy="trace one negative prompt per positive prompt, reading one token each",
        )
    if torch.equal(positive, negative):
        raise InvalidArgumentError(
            "The positive and negative rows are identical, so the direction would be all zeros",
            code="direction_pairs_identical",
            remedy="check that neg_traces holds the negative prompts, not the positive ones",
        )


def _fit(positive: torch.Tensor, negative: torch.Tensor, method: str) -> torch.Tensor:
    """Fit the direction in the rows' dtype."""

    deltas = positive - negative
    if method == "mean_diff":
        return deltas.mean(dim=0)
    wide = deltas.to(torch.float64)
    if method == "pca":
        vector = torch.linalg.svd(wide, full_matrices=False).Vh[0]
        if float((wide @ vector).mean()) < 0:
            vector = -vector
    else:
        vector = _logistic_direction(wide / 2)
    return (vector / vector.norm()).to(positive.dtype)


def _logistic_direction(half: torch.Tensor) -> torch.Tensor:
    """Solve L2 logistic regression of ``half`` (label 1) against ``-half`` (label 0).

    The objective ``0.5 |w|^2 + sum log(1 + exp(-z_i w.x_i))`` over the
    stacked rows with ``z = +1`` for ``half`` and ``-1`` for ``-half`` is
    symmetric, so it equals ``0.5 |w|^2 + 2 sum log(1 + exp(-w.h_i))`` over
    the ``half`` rows alone. Newton steps solve in the smaller of the two
    dimensions (Woodbury when there are fewer rows than features).
    """

    rows, width = half.shape
    weight = torch.zeros(width, dtype=torch.float64)
    for _ in range(_LOGISTIC_MAX_ITER):
        margin = torch.sigmoid(-(half @ weight))
        grad = weight - 2 * half.T @ margin
        curvature = 2 * margin * (1 - margin)
        step = _newton_step(half, curvature, grad, rows < width)
        weight = _backtrack(half, weight, step)
        if float(step.abs().max()) <= _LOGISTIC_TOL * max(1.0, float(weight.abs().max())):
            return weight
    raise InvalidArgumentError(
        f"Logistic regression did not converge in {_LOGISTIC_MAX_ITER} Newton steps",
        code="direction_logistic_unconverged",
        remedy="use method='mean_diff' or 'pca', or check the rows for non-finite values",
    )


def _logistic_objective(half: torch.Tensor, weight: torch.Tensor) -> float:
    """Return ``0.5 |w|^2 + 2 sum softplus(-h_i . w)``."""

    return float(0.5 * weight @ weight + 2 * torch.nn.functional.softplus(-(half @ weight)).sum())


def _backtrack(half: torch.Tensor, weight: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Take the Newton step, halving it until the objective does not increase."""

    current = _logistic_objective(half, weight)
    scale = 1.0
    for _ in range(30):
        candidate = weight - scale * step
        if _logistic_objective(half, candidate) <= current:
            return candidate
        scale /= 2
    return weight - scale * step


def _newton_step(
    half: torch.Tensor, curvature: torch.Tensor, grad: torch.Tensor, woodbury: bool
) -> torch.Tensor:
    """Return ``(I + H^T diag(c) H)^-1 grad`` for the logistic Hessian."""

    if not woodbury:
        hessian = torch.eye(half.shape[1], dtype=half.dtype) + half.T @ (half * curvature[:, None])
        return torch.linalg.solve(hessian, grad)
    inner = torch.diag(1 / curvature.clamp_min(1e-300)) + half @ half.T
    return grad - half.T @ torch.linalg.solve(inner, half @ grad)


def _model_type(trace: Any) -> str | None:
    """Return the traced model's ``config.model_type``, or ``None``."""

    from ..data_classes._runtime_handles import source_model_from_trace

    found = getattr(getattr(source_model_from_trace(trace), "config", None), "model_type", None)
    return found if isinstance(found, str) else None


__all__ = ["Direction", "direction_from_pairs"]
