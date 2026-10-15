"""Logit-lens appliance over semantic facets.

``logit_lens`` pushes each transformer block's residual-stream state through
the model's OWN final normalization + unembedding, showing what the model
"would have said" at every depth. Every architecture-specific fact (which
child is the unembedding, which norm feeds it, the norm kind/parameters)
comes from the ``language_model_head`` facet recipe -- this module is generic
math over that facet vocabulary plus the existing block ``resid_*`` facets,
so new architectures extend coverage by registering recipes, never by
editing this appliance.

Vision models have no language-model head. When none exists, the default
lens falls back to a vision head facet recipe: a classifier head
(``classification_head``: final norm plus ``classifier`` over one token) or
CLIP's image-text similarity (``contrastive_image_head``: decode against the
captured text features of the prompts the capture ran). The fallback lens is
validated exactly like the language-model lens: re-projecting the last
block's captured ``resid_post`` must reproduce the model's captured logits.

A user lens (``lens=``) may record the model it was built for as a
``model_identity`` attribute (the record an extraction artifact carries in
``manifest["signature"]["model_identity"]``). A measured identity that
differs from the traced model refuses unless ``validate=False``; a lens with
no comparable identity gets a last-layer agreement check that warns when
its last-layer decode does not reproduce the model's captured logits.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from ..errors._base import TorchLensError, TorchLensWarning
from .facets import Facet, MissingFacet

__all__ = [
    "LogitLensEntry",
    "LogitLensError",
    "LogitLensPrediction",
    "LogitLensPredictions",
    "LogitLensResult",
    "logit_lens",
    "logit_lens_predictions",
]

LensFunc = Callable[[torch.Tensor], torch.Tensor]

#: Rank convention for requested-token ranks (one-based; recorded on results
#: so a consumer never has to guess how ties were broken).
TIE_CONVENTION = "rank = 1 + count(logits strictly greater); tied logits share the smallest rank"

#: Provenance vocabulary for prediction rows. Only a row served directly from
#: the model's captured output logits may claim to BE the native output;
#: every projected row says what it is.
PROVENANCE_NATIVE = "native output"
PROVENANCE_PROJECTED = "projected through final norm/head"

#: ``lens_source`` values for the default lens, by the head that built it.
LENS_SOURCE_MODEL_HEAD = "model_head"
LENS_SOURCE_CLASSIFIER_HEAD = "classifier_head"
LENS_SOURCE_CONTRASTIVE_HEAD = "contrastive_head"

#: The only identity-record level a trace can be compared against.
_COMPARABLE_IDENTITY_LEVEL = "measured"
_IDENTITY_HASHES = frozenset({"blake2b", "sha256"})


class LogitLensError(TorchLensError, RuntimeError):
    """Raised when a logit lens cannot be built, validated, or applied.

    Contracted refusals carry a stable ``fields["code"]``; the historical
    ``RuntimeError`` lineage is preserved for existing catch sites.
    """


@dataclass(frozen=True)
class LogitLensEntry:
    """Per-layer logit-lens projection.

    Parameters
    ----------
    address:
        Module address of the transformer block the hidden state came from.
    layer_index:
        Zero-based position of the block among the projected layers.
    facet:
        Hidden-state facet name that was projected (e.g. ``"resid_post"``).
    logits:
        Projected logits with the hidden state's leading dimensions preserved.
    """

    address: str
    layer_index: int
    facet: str
    logits: torch.Tensor

    def top_k(self, k: int = 5, *, position: int = -1, batch_index: int = 0) -> tuple[Any, Any]:
        """Return top-k probabilities and token ids at one sequence position.

        Parameters
        ----------
        k:
            Number of top tokens.
        position:
            Sequence position (last dimension before vocab), when present.
        batch_index:
            Batch element to inspect.

        Returns
        -------
        tuple[Any, Any]
            ``(probabilities, token_ids)`` tensors of length ``k``.
        """

        vector = _position_vector(self.logits, position=position, batch_index=batch_index)
        probabilities = torch.softmax(vector.float(), dim=-1)
        values, indices = probabilities.topk(min(k, probabilities.shape[-1]))
        return values, indices


@dataclass(frozen=True)
class LogitLensResult:
    """Result of a :func:`logit_lens` sweep across transformer blocks.

    Parameters
    ----------
    entries:
        Per-layer projections in block execution order.
    facet:
        Hidden-state facet name that was projected.
    lens_source:
        ``"model_head"`` when the lens was reconstructed from the
        ``language_model_head`` facets, ``"classifier_head"`` or
        ``"contrastive_head"`` when it came from a vision head recipe, and
        ``"user"`` when supplied via ``lens=``.
    validated:
        Whether the reconstructed lens was numerically validated against the
        captured last-block ``resid_post`` -> logits pair. Always ``False``
        for user lenses (TorchLens has no oracle for them; see the identity
        check under ``logit_lens(validate=...)``).
    final_logits:
        The model's captured real output logits when readable, else ``None``.
    """

    entries: tuple[LogitLensEntry, ...]
    facet: str
    lens_source: str
    validated: bool
    final_logits: torch.Tensor | None

    def stacked(self) -> torch.Tensor:
        """Return all per-layer logits stacked on a new leading layer axis.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_layers, *logits_shape)``.
        """

        shapes = {tuple(entry.logits.shape) for entry in self.entries}
        if len(shapes) > 1:
            raise LogitLensError(
                f"Cannot stack per-layer logits with differing shapes {sorted(shapes)!r}; "
                "read result.entries[i].logits individually."
            )
        return torch.stack([entry.logits for entry in self.entries])

    def top_tokens(
        self,
        k: int = 5,
        *,
        position: int = -1,
        batch_index: int = 0,
        tokenizer: Any | None = None,
    ) -> list[tuple[str, list[tuple[Any, float]]]]:
        """Return per-layer top-k tokens at one sequence position.

        Parameters
        ----------
        k:
            Number of top tokens per layer.
        position:
            Sequence position to inspect.
        batch_index:
            Batch element to inspect.
        tokenizer:
            Optional tokenizer with ``convert_ids_to_tokens`` or ``decode``;
            when omitted, raw token ids are returned.

        Returns
        -------
        list[tuple[str, list[tuple[Any, float]]]]
            One ``(address, [(token, probability), ...])`` row per layer.
        """

        rows: list[tuple[str, list[tuple[Any, float]]]] = []
        for entry in self.entries:
            values, indices = entry.top_k(k, position=position, batch_index=batch_index)
            ids = [int(index) for index in indices]
            tokens: Sequence[Any] = ids if tokenizer is None else _decode_ids(tokenizer, ids)
            rows.append((entry.address, list(zip(tokens, [float(value) for value in values]))))
        return rows

    def summary(
        self,
        *,
        position: int = -1,
        batch_index: int = 0,
        tokenizer: Any | None = None,
    ) -> str:
        """Return a human-readable per-layer top-1 table.

        Parameters
        ----------
        position:
            Sequence position to inspect.
        batch_index:
            Batch element to inspect.
        tokenizer:
            Optional tokenizer used to decode token ids.

        Returns
        -------
        str
            One line per layer with the top token and its probability.
        """

        lines = [f"logit lens ({self.facet}, lens={self.lens_source})"]
        for entry in self.entries:
            values, indices = entry.top_k(1, position=position, batch_index=batch_index)
            token: Any = int(indices[0])
            if tokenizer is not None:
                token = _decode_ids(tokenizer, [int(indices[0])])[0]
            lines.append(
                f"  [{entry.layer_index:>3}] {entry.address}: {token!r} ({float(values[0]):.4f})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class LogitLensPrediction:
    """Streaming per-layer prediction summary (one row per projected layer).

    Parameters
    ----------
    address:
        Module address of the transformer block (or the head module for the
        native-output row).
    layer_index:
        Zero-based position among the projected layers; ``None`` for the
        native-output row.
    facet:
        Hidden-state facet that was projected; ``None`` for the native row.
    provenance:
        ``"projected through final norm/head"`` for reconstructed rows;
        ``"native output"`` ONLY for the row served directly from the model's
        captured output logits.
    positions:
        Absolute sequence positions this row covers, in row order.
    top_ids / top_logits / top_probs:
        Top-k token ids, logit values, and probabilities shaped
        ``[batch, n_positions, k]``. Probabilities ALWAYS use the
        full-vocabulary softmax denominator.
    logsumexp:
        Full-vocabulary ``logsumexp`` per position, ``[batch, n_positions]``
        (float32) -- retains the denominator so any token's probability can
        be recovered later from its logit.
    token_ids / token_logits / token_probs / token_ranks:
        Requested-token values: ids as given, logits/probabilities shaped
        ``[batch, n_positions, n_tokens]``, and ONE-BASED ranks (int64, tie
        convention recorded on the parent result).
    """

    address: str
    layer_index: int | None
    facet: str | None
    provenance: str
    positions: tuple[int, ...]
    top_ids: torch.Tensor
    top_logits: torch.Tensor
    top_probs: torch.Tensor
    logsumexp: torch.Tensor
    token_ids: tuple[int, ...]
    token_logits: torch.Tensor
    token_probs: torch.Tensor
    token_ranks: torch.Tensor


@dataclass(frozen=True)
class LogitLensPredictions:
    """Result of a streaming :func:`logit_lens_predictions` sweep.

    Parameters
    ----------
    rows:
        Per-layer prediction summaries in block execution order, optionally
        followed by the native-output row.
    k:
        Requested top-k width.
    facet:
        Hidden-state facet that was projected.
    lens_source:
        ``"model_head"``, ``"classifier_head"``, ``"contrastive_head"`` or
        ``"user"`` (see :class:`LogitLensResult`).
    validated:
        Whether the reconstructed lens passed numeric validation.
    tie_convention:
        Recorded rank tie convention for ``token_ranks``.
    """

    rows: tuple[LogitLensPrediction, ...]
    k: int
    facet: str
    lens_source: str
    validated: bool
    tie_convention: str = TIE_CONVENTION


def logit_lens(
    trace: Any,
    *,
    facet: str = "resid_post",
    layers: Sequence[str] | None = None,
    lens: LensFunc | Mapping[str, LensFunc] | None = None,
    validate: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> LogitLensResult:
    """Project per-block hidden states through the model's output head.

    Parameters
    ----------
    trace:
        Completed TorchLens trace of a language model, a vision classifier
        (classification head), or a CLIP model run on images and label
        prompts, captured with the relevant activations saved (the default
        exhaustive save suffices).
    facet:
        Hidden-state facet to project (``"resid_post"``, ``"resid_pre"``, or
        ``"resid_mid"``).
    layers:
        Explicit module addresses to project; defaults to every module
        exposing ``facet``, in trace order. A vision head lens projects only
        the blocks of the tower that feeds its head (CLIP's vision tower,
        never its text tower).
    lens:
        Optional user lens: one callable ``hidden -> logits`` applied to every
        layer, or a mapping from module address to per-layer callables (e.g. a
        tuned lens). When omitted, the lens is reconstructed from the
        ``language_model_head`` facets (or, when the trace has no language
        model head, from a vision head recipe) and numerically validated. A
        user lens may carry a ``model_identity`` attribute (on the callable,
        the mapping, or each per-layer callable) holding the identity record
        of the model it was built for.
    validate:
        Whether to check the lens. For the reconstructed default lens: re-project
        the last block's captured ``resid_post`` and compare against the
        captured logits. Fail-closed: validation that cannot run (missing
        anchors) refuses rather than silently trusting the reconstruction.
        For a user lens: a recorded measured identity that differs from the
        traced model refuses (``logit_lens_identity_mismatch``); a lens with no
        comparable identity gets the same last-layer re-projection as a check
        that warns (``logit_lens_identity_unrecorded``) when it disagrees or
        cannot run. ``validate=False`` skips every check.
    rtol:
        Relative tolerance for lens validation.
    atol:
        Absolute tolerance for lens validation.

    Returns
    -------
    LogitLensResult
        Per-layer projections in block order.
    """

    setup = _prepare_lens(
        trace, facet=facet, layers=layers, lens=lens, validate=validate, rtol=rtol, atol=atol
    )
    entries = [
        LogitLensEntry(address=address, layer_index=layer_index, facet=facet, logits=logits)
        for layer_index, address, logits in _iter_layer_logits(trace, setup)
    ]
    return LogitLensResult(
        entries=tuple(entries),
        facet=facet,
        lens_source=setup.lens_source,
        validated=setup.validated,
        final_logits=setup.final_logits,
    )


def logit_lens_predictions(
    trace: Any,
    *,
    k: int = 5,
    positions: Sequence[int] | None = None,
    tokens: Sequence[int] | None = None,
    facet: str = "resid_post",
    layers: Sequence[str] | None = None,
    lens: LensFunc | Mapping[str, LensFunc] | None = None,
    validate: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    include_native: bool = True,
    mask_input: bool = False,
) -> LogitLensPredictions:
    """Stream per-layer top-k predictions without retaining full projections.

    :func:`logit_lens` retains every per-layer ``[batch, positions, vocab]``
    projection (~2.5 GB extrapolated at 1,024 positions on a 50k vocabulary);
    this extractor projects ONE layer at a time, reduces it to top-k values,
    the full-vocabulary ``logsumexp``, and requested-token values/ranks, then
    discards the layer's full projection. Probabilities always use the
    full-vocabulary softmax denominator. Only the row served directly from
    the model's CAPTURED output logits is labelled ``"native output"``;
    every reconstructed row is labelled ``"projected through final
    norm/head"``.

    Parameters
    ----------
    k:
        Top-k width per position (clamped to the vocabulary size).
    positions:
        Absolute sequence positions to keep (negative indices allowed);
        defaults to every position a row covers. Positions outside a row's
        covered range are dropped from THAT row (the native row of a
        ``logits_to_keep``-sliced capture covers only the last positions).
    tokens:
        Token ids whose logit/probability/rank are retained per position.
    facet / layers / lens / validate / rtol / atol:
        As in :func:`logit_lens`.
    include_native:
        Whether to append the native-output row when captured logits are
        readable.
    mask_input:
        ``True`` bans the model input's own token ids from every row's
        predictions: each batch row's input tokens get ``-inf`` logits
        before any reduction, so top-k, the ``logsumexp`` denominator,
        probabilities and ranks all see the masked vocabulary. This cannot
        be recomputed from the retained rows afterwards (their
        denominators are already reduced), which is why it is a knob here.
        The trace's first input must be a saved integer token-id tensor
        (``logit_lens_mask_input_unavailable`` otherwise). Off by default.

    Returns
    -------
    LogitLensPredictions
        Reduced per-layer rows in block order (+ optional native row).
    """

    if k < 1:
        raise LogitLensError(
            f"k must be a positive integer, got {k!r}. Remedy: pass k >= 1.",
            code="logit_lens_k_invalid",
        )
    setup = _prepare_lens(
        trace, facet=facet, layers=layers, lens=lens, validate=validate, rtol=rtol, atol=atol
    )
    head, final_logits = setup.head, setup.final_logits
    token_ids = tuple(int(token) for token in tokens) if tokens is not None else ()
    banned = _input_token_ids(trace) if mask_input else None
    rows: list[LogitLensPrediction] = []
    sequence_length: int | None = None
    with torch.no_grad():
        for layer_index, address, projected in _iter_layer_logits(trace, setup):
            if sequence_length is None and projected.ndim >= 3:
                sequence_length = int(projected.shape[-2])
            rows.append(
                _reduce_projection_row(
                    projected,
                    address=address,
                    layer_index=layer_index,
                    facet=facet,
                    provenance=PROVENANCE_PROJECTED,
                    position_base=0,
                    full_length=sequence_length,
                    k=k,
                    positions=positions,
                    token_ids=token_ids,
                    banned_ids=banned,
                )
            )
            del projected
        if include_native and final_logits is not None:
            native_length = int(final_logits.shape[-2]) if final_logits.ndim >= 3 else 1
            full_length = sequence_length if sequence_length is not None else native_length
            head_address = (head.address if head is not None else None) or "<head>"
            rows.append(
                _reduce_projection_row(
                    final_logits,
                    address=head_address,
                    layer_index=None,
                    facet=None,
                    provenance=PROVENANCE_NATIVE,
                    # HF logits_to_keep projects the LAST K positions, so the
                    # native row's absolute positions start at S - K.
                    position_base=max(0, full_length - native_length),
                    full_length=full_length,
                    k=k,
                    positions=positions,
                    token_ids=token_ids,
                    banned_ids=banned,
                )
            )
    return LogitLensPredictions(
        rows=tuple(rows),
        k=k,
        facet=facet,
        lens_source=setup.lens_source,
        validated=setup.validated,
    )


@dataclass(frozen=True)
class _LensSetup:
    """A resolved, checked lens for one trace (shared by every lens reader).

    Attributes
    ----------
    facet:
        The hidden-state facet each layer is read from.
    addresses:
        Block addresses to project, in block order.
    lens_by_address:
        The per-address lens callable.
    lens_source:
        ``"user"`` or the default head's kind.
    validated:
        Whether a default lens was numerically validated.
    head:
        The default head, when the trace has one.
    final_logits:
        The head's captured output logits, when readable.
    """

    facet: str
    addresses: tuple[str, ...]
    lens_by_address: Mapping[str, LensFunc]
    lens_source: str
    validated: bool
    head: _DefaultHead | None
    final_logits: torch.Tensor | None


def _prepare_lens(
    trace: Any,
    *,
    facet: str,
    layers: Sequence[str] | None,
    lens: LensFunc | Mapping[str, LensFunc] | None,
    validate: bool,
    rtol: float,
    atol: float,
) -> _LensSetup:
    """Resolve the head, the layers and the lens, and check the lens once.

    The one home of the lens rule for :func:`logit_lens`,
    :func:`logit_lens_predictions` and ``tl.stats.measure_trajectory``.
    """

    head = _find_default_head(trace)
    final_logits = _read_final_logits(head)
    lens_by_address: Mapping[str, LensFunc]
    if lens is not None:
        addresses = _resolve_layer_addresses(trace, facet=facet, layers=layers)
        lens_by_address = _user_lens_by_address(lens, addresses)
        if validate:
            _check_user_lens(trace, lens, lens_by_address, head, (rtol, atol))
        lens_source, validated = "user", False
    else:
        if head is None:
            raise LogitLensError(
                "No module in this trace exposes the 'unembed_weight' facet (or a vision "
                "head's 'classifier_weight' / 'image_projection_weight' facet), so the "
                "model's own lens cannot be reconstructed. Remedy: register a facet "
                "recipe producing the language_model_head facet names for this "
                "architecture (tl.facets.register), or pass lens= explicitly.",
                code="logit_lens_head_unavailable",
            )
        addresses = _resolve_layer_addresses(trace, facet=facet, layers=layers, tower=head.tower)
        default_lens = _build_head_lens(head)
        if validate:
            _validate_lens(trace, head, default_lens, rtol=rtol, atol=atol)
        lens_by_address = dict.fromkeys(addresses, default_lens)
        lens_source, validated = head.kind, validate
    return _LensSetup(
        facet=facet,
        addresses=tuple(addresses),
        lens_by_address=lens_by_address,
        lens_source=lens_source,
        validated=validated,
        head=head,
        final_logits=final_logits,
    )


def _iter_layer_logits(trace: Any, setup: _LensSetup) -> Iterator[tuple[int, str, torch.Tensor]]:
    """Yield ``(layer_index, address, logits)`` one layer at a time.

    Each layer is projected under ``torch.no_grad`` from its detached facet
    value; the caller reduces and drops it before the next layer, so at
    most one ``[batch, positions, vocab]`` projection is alive.
    """

    for layer_index, address in enumerate(setup.addresses):
        with torch.no_grad():
            hidden = _facet_tensor(trace.modules[address].facets[setup.facet], name=setup.facet)
            logits = setup.lens_by_address[address](hidden.detach())
        yield layer_index, address, logits


def _input_token_ids(trace: Any) -> torch.Tensor:
    """Return the trace's integer model input as ``[batch, positions]`` token ids.

    ``mask_input=True`` bans these ids; a trace whose first input is not a
    saved integer tensor refuses (``logit_lens_mask_input_unavailable``).
    """

    labels = list(getattr(trace, "input_layers", ()) or ())
    value: Any = None
    if labels:
        try:
            value = trace[labels[0]].out
        except Exception:  # noqa: BLE001 -- any unreadable input means "unavailable" below
            value = None
    if not isinstance(value, torch.Tensor) or value.is_floating_point() or value.is_complex():
        raise LogitLensError(
            "mask_input=True bans the input's own token ids, but this trace's first model "
            "input is not a saved integer token-id tensor (a model fed embeddings has no "
            "token ids to ban). Remedy: trace the model on token ids with activations "
            "saved, e.g. tl.trace(model, input_ids), or drop mask_input=True.",
            code="logit_lens_mask_input_unavailable",
        )
    ids = value.detach().long()
    return ids.unsqueeze(0) if ids.ndim == 1 else ids.reshape(ids.shape[0], -1)


def _mask_input_tokens(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Set every input token's logit to ``-inf`` in each batch row, before any reduction.

    ``logits`` is ``[batch, positions, vocab]``; ``token_ids`` is
    ``[batch, input_positions]``. Each row bans the tokens of its own input
    at every position (the tuned-lens ``mask_input`` meaning).
    """

    batch, vocab = int(logits.shape[0]), int(logits.shape[-1])
    if int(token_ids.shape[0]) != batch:
        raise LogitLensError(
            f"mask_input=True: the input has {int(token_ids.shape[0])} token rows but the "
            f"projected logits have {batch} batch rows. Remedy: pass one token row per batch "
            "row, or drop mask_input=True.",
            code="logit_lens_mask_input_unavailable",
        )
    ids = token_ids.to(logits.device)
    valid = (ids >= 0) & (ids < vocab)
    banned = torch.zeros(batch, vocab, dtype=torch.bool, device=logits.device)
    banned.scatter_(1, ids.clamp(0, vocab - 1), valid)
    return logits.masked_fill(banned.unsqueeze(-2), -math.inf)


def _reduce_projection_row(
    projected: torch.Tensor,
    *,
    address: str,
    layer_index: int | None,
    facet: str | None,
    provenance: str,
    position_base: int,
    full_length: int | None,
    k: int,
    positions: Sequence[int] | None,
    token_ids: tuple[int, ...],
    banned_ids: torch.Tensor | None = None,
) -> LogitLensPrediction:
    """Reduce one full projection to its retained prediction summary.

    The full ``[batch, positions, vocab]`` tensor is read exactly once here
    and never stored on the returned row.
    """

    work = projected.detach()
    if work.ndim < 2:
        raise LogitLensError(
            f"Projected logits for module {address!r} have shape {tuple(work.shape)}; "
            "prediction extraction needs at least (positions|batch, vocab). Remedy: "
            "pass a lens= returning at-least-rank-2 logits for this layer.",
            code="logit_lens_projection_rank_invalid",
        )
    if work.ndim == 2:
        work = work.unsqueeze(-2)
    if work.ndim > 3:
        work = work.reshape(-1, work.shape[-2], work.shape[-1])
    work = work.float()
    if banned_ids is not None:
        work = _mask_input_tokens(work, banned_ids)
    vocab = int(work.shape[-1])
    covered = _row_positions(
        n_row_positions=int(work.shape[-2]),
        position_base=position_base,
        full_length=full_length,
        requested=positions,
    )
    local_index = torch.tensor(
        [position - position_base for position in covered], dtype=torch.long, device=work.device
    )
    work = work.index_select(-2, local_index)
    logsumexp = torch.logsumexp(work, dim=-1)
    k_eff = min(k, vocab)
    top_logits, top_ids = work.topk(k_eff, dim=-1)
    top_probs = torch.exp(top_logits - logsumexp.unsqueeze(-1))
    if token_ids:
        bad = [token for token in token_ids if not 0 <= token < vocab]
        if bad:
            raise LogitLensError(
                f"tokens= contains ids {bad!r} outside the vocabulary of size {vocab}. "
                f"Remedy: pass token ids in [0, {vocab}).",
                code="logit_lens_token_id_invalid",
            )
        token_index = torch.tensor(token_ids, dtype=torch.long, device=work.device)
        token_logits = work.index_select(-1, token_index)
        token_probs = torch.exp(token_logits - logsumexp.unsqueeze(-1))
        # One token at a time: broadcasting all tokens at once materializes a
        # [batch, positions, n_tokens, vocab] bool -- the exact full-vocab
        # blowup this extractor exists to avoid.
        token_ranks = torch.stack(
            [
                (work > token_logits[..., index].unsqueeze(-1)).sum(dim=-1) + 1
                for index in range(len(token_ids))
            ],
            dim=-1,
        )
    else:
        empty_shape = (*work.shape[:-1], 0)
        token_logits = work.new_empty(empty_shape)
        token_probs = work.new_empty(empty_shape)
        token_ranks = torch.empty(empty_shape, dtype=torch.long, device=work.device)
    return LogitLensPrediction(
        address=address,
        layer_index=layer_index,
        facet=facet,
        provenance=provenance,
        positions=covered,
        top_ids=top_ids,
        top_logits=top_logits,
        top_probs=top_probs,
        logsumexp=logsumexp,
        token_ids=token_ids,
        token_logits=token_logits,
        token_probs=token_probs,
        token_ranks=token_ranks,
    )


def _row_positions(
    *,
    n_row_positions: int,
    position_base: int,
    full_length: int | None,
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return the absolute positions a row retains.

    Negative requested positions are normalized against the FULL sequence
    length; requested positions a row does not cover (e.g. early positions on
    a ``logits_to_keep``-sliced native row) are dropped from that row.
    """

    covered_range = range(position_base, position_base + n_row_positions)
    if requested is None:
        return tuple(covered_range)
    length = full_length if full_length is not None else position_base + n_row_positions
    normalized: list[int] = []
    for position in requested:
        absolute = int(position)
        if absolute < 0:
            absolute += length
        if not 0 <= absolute < length:
            raise LogitLensError(
                f"positions= entry {position!r} is outside the sequence of length {length}. "
                f"Remedy: pass positions in [-{length}, {length}).",
                code="logit_lens_position_invalid",
            )
        if absolute in covered_range and absolute not in normalized:
            normalized.append(absolute)
    return tuple(normalized)


def _facet_views(trace: Any) -> list[tuple[str, Any]]:
    """Return ``(address, facet view)`` for every single-call module.

    A module called more than once (a dropout reused inside a block, a
    weight-shared layer) serves facets per call only, and its ``facets``
    view refuses; it can be neither a projected block nor a head here.
    """

    views: list[tuple[str, Any]] = []
    for module in trace.modules:
        if len(tuple(getattr(module, "calls", ()) or ())) > 1:
            continue
        views.append((str(getattr(module, "address", "")), module.facets))
    return views


def _in_tower(address: str, tower: str | None) -> bool:
    """Return whether a module address lies inside a head's tower."""

    if tower is None or tower == "self":
        return True
    return address == tower or address.startswith(f"{tower}.")


def _resolve_layer_addresses(
    trace: Any, *, facet: str, layers: Sequence[str] | None, tower: str | None = None
) -> list[str]:
    """Return the block addresses to project, validating availability.

    ``tower`` (vision heads only) restricts the default to the blocks of the
    tower feeding the head and refuses explicit layers outside it: a CLIP
    image head decodes the vision tower's residual stream, never the text
    tower's.
    """

    available = [
        address for address, view in _facet_views(trace) if address != "self" and view.has(facet)
    ]
    if layers is None:
        in_tower = [address for address in available if _in_tower(address, tower)]
        if not in_tower:
            raise LogitLensError(
                f"No modules in this trace expose facet {facet!r}. Logit lens needs "
                "transformer blocks with residual-stream facets; capture with the "
                "default exhaustive save, or register a block facet recipe for this "
                "architecture."
            )
        return in_tower
    missing = [address for address in layers if address not in available]
    if missing:
        raise LogitLensError(
            f"Modules {tuple(missing)!r} do not expose facet {facet!r} in this trace. "
            f"Available: {tuple(available)!r}."
        )
    outside = [str(address) for address in layers if not _in_tower(str(address), tower)]
    if outside:
        raise LogitLensError(
            f"Modules {tuple(outside)!r} lie outside {tower!r}, the tower whose blocks "
            "feed this trace's vision head, so the head's lens cannot decode them. "
            f"Remedy: pass layers inside {tower!r}, or pass lens= for those layers.",
            code="logit_lens_layer_outside_head",
        )
    return [str(address) for address in layers]


@dataclass(frozen=True)
class _DefaultHead:
    """The head a default lens is reconstructed from."""

    kind: str
    view: Any
    address: str | None
    tower: str | None
    logits_facet: str = "logits"


#: Fallback vision heads, in priority order: the facet that identifies the
#: head, its ``lens_source``, and the facet holding its captured output.
_VISION_HEAD_KEYS = (
    ("classifier_weight", LENS_SOURCE_CLASSIFIER_HEAD, "logits"),
    ("image_projection_weight", LENS_SOURCE_CONTRASTIVE_HEAD, "logits_per_image"),
)


def _find_default_head(trace: Any) -> _DefaultHead | None:
    """Return the language-model head, else the first vision head, else ``None``."""

    views = _facet_views(trace)
    for address, view in views:
        if view.has("unembed_weight"):
            return _DefaultHead(LENS_SOURCE_MODEL_HEAD, view, address or None, None)
    for key, kind, logits_facet in _VISION_HEAD_KEYS:
        for address, view in views:
            if view.has(key):
                tower = view.get("tower_address")
                return _DefaultHead(
                    kind,
                    view,
                    address or None,
                    tower if isinstance(tower, str) else None,
                    logits_facet,
                )
    return None


def _read_final_logits(head: _DefaultHead | None) -> torch.Tensor | None:
    """Return the head's captured real output logits when readable."""

    if head is None:
        return None
    value = head.view.get(head.logits_facet)
    if value is None or isinstance(value, MissingFacet):
        return None
    try:
        return _facet_tensor(value, name="logits").detach()
    except LogitLensError:
        return None


def _user_lens_by_address(
    lens: LensFunc | Mapping[str, LensFunc], addresses: Sequence[str]
) -> dict[str, LensFunc]:
    """Return a per-address lens mapping from the user ``lens=`` argument."""

    if callable(lens):
        return dict.fromkeys(addresses, lens)
    if isinstance(lens, Mapping):
        missing = [address for address in addresses if address not in lens]
        if missing:
            raise LogitLensError(
                f"lens= mapping is missing entries for modules {tuple(missing)!r}; "
                "provide one callable per projected layer or a single callable."
            )
        return {address: lens[address] for address in addresses}
    raise LogitLensError(
        "lens= must be a callable hidden -> logits or a mapping from module "
        f"address to such callables, got {type(lens).__name__}."
    )


def _optional_tensor(head_view: Any, name: str) -> torch.Tensor | None:
    """Return a facet's tensor, or ``None`` when the facet is absent."""

    value = head_view.get(name)
    if value is None or isinstance(value, MissingFacet):
        return None
    return _facet_tensor(value, name=name).detach()


def _final_norm_fn(head_view: Any) -> LensFunc:
    """Reconstruct the head's final normalization from its facets."""

    kind = head_view.get("final_norm_kind")
    if not isinstance(kind, str):
        raise LogitLensError(
            "The final normalization feeding the unembedding head was not "
            "identified (facet 'final_norm_kind' is unavailable), so the model's "
            "own lens cannot be reconstructed. Pass lens= explicitly, or register "
            "a facet recipe classifying this model's final norm."
        )
    gamma = _facet_tensor(head_view["final_norm_gamma"], name="final_norm_gamma").detach()
    beta = _optional_tensor(head_view, "final_norm_beta")
    eps_value = head_view.get("final_norm_eps")
    eps = float(eps_value) if isinstance(eps_value, (int, float)) else 1e-5

    def _normalize(hidden: torch.Tensor) -> torch.Tensor:
        """Apply the reconstructed final norm to a hidden state."""

        if kind == "layer_norm":
            return F.layer_norm(hidden, gamma.shape, gamma, beta, eps)
        if kind == "rms_norm":
            return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma
        raise LogitLensError(f"Unknown final_norm_kind {kind!r}; pass lens= explicitly.")

    return _normalize


def _build_default_lens(head_view: Any) -> LensFunc:
    """Reconstruct the model's own final-norm + unembed lens from facets."""

    normalize = _final_norm_fn(head_view)
    weight = _facet_tensor(head_view["unembed_weight"], name="unembed_weight").detach()
    bias = _optional_tensor(head_view, "unembed_bias")

    def _apply(hidden: torch.Tensor) -> torch.Tensor:
        """Apply the reconstructed final norm + unembedding to a hidden state."""

        logits = normalize(hidden) @ weight.transpose(-2, -1)
        if bias is not None:
            logits = logits + bias
        return logits

    return _apply


def _read_token(hidden: torch.Tensor, position: int) -> torch.Tensor:
    """Return the one sequence position a vision head reads, ``[batch, hidden]``."""

    if hidden.ndim != 3:
        raise LogitLensError(
            f"A vision head lens reads one token of a [batch, sequence, hidden] state, "
            f"but the projected hidden state has shape {tuple(hidden.shape)}. Remedy: "
            "project a block residual facet such as 'resid_post', or pass lens=.",
            code="logit_lens_hidden_rank_invalid",
        )
    return hidden[:, position, :]


def _vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Return the last-axis L2 norm, computed the way CLIP computes it."""

    return torch.sum(torch.pow(tensor, 2), dim=-1, keepdim=True) ** 0.5


def _build_head_lens(head: _DefaultHead) -> LensFunc:
    """Reconstruct the default lens of a language-model or vision head."""

    if head.kind == LENS_SOURCE_MODEL_HEAD:
        return _build_default_lens(head.view)
    view = head.view
    normalize = _final_norm_fn(view)
    position = view.get("pooled_position")
    if not isinstance(position, int) or isinstance(position, bool):
        raise LogitLensError(
            "The vision head's token read was not identified (facet 'pooled_position' "
            "is unavailable), so the model's own lens cannot be reconstructed. Remedy: "
            "pass lens= explicitly, or register a facet recipe for this head.",
            code="logit_lens_head_unavailable",
        )
    if head.kind == LENS_SOURCE_CLASSIFIER_HEAD:
        weight = _facet_tensor(view["classifier_weight"], name="classifier_weight").detach()
        bias = _optional_tensor(view, "classifier_bias")

        def _classify(hidden: torch.Tensor) -> torch.Tensor:
            """Read the head's token, normalize it, and apply the classifier."""

            logits = normalize(_read_token(hidden, position)) @ weight.transpose(-2, -1)
            return logits if bias is None else logits + bias

        return _classify
    projection = _facet_tensor(
        view["image_projection_weight"], name="image_projection_weight"
    ).detach()
    text = _facet_tensor(view["text_features"], name="text_features").detach()
    scale = _facet_tensor(view["logit_scale"], name="logit_scale").detach().exp()
    text_unit = text / _vector_norm(text)

    def _similarity(hidden: torch.Tensor) -> torch.Tensor:
        """Score the head's token against the captured label-prompt features."""

        image = normalize(_read_token(hidden, position)) @ projection.transpose(-2, -1)
        image = image / _vector_norm(image)
        return (text_unit @ image.transpose(-2, -1) * scale).transpose(-2, -1)

    return _similarity


def _last_block_address(trace: Any, tower: str | None) -> str | None:
    """Return the last block exposing ``resid_post`` inside the head's tower."""

    resid_addresses = [
        address
        for address, view in _facet_views(trace)
        if address != "self" and view.has("resid_post") and _in_tower(address, tower)
    ]
    return resid_addresses[-1] if resid_addresses else None


def _validate_lens(
    trace: Any, head: _DefaultHead, lens: LensFunc, *, rtol: float, atol: float
) -> None:
    """Validate a reconstructed lens against captured last-block -> logits data.

    Applying the reconstructed final norm + unembedding to the LAST block's
    captured ``resid_post`` must reproduce the model's captured real logits:
    that is exactly the path the real forward took, so the check validates the
    full norm + unembed reconstruction end-to-end. This is a tripwire, not a
    formality: an unconventional head (extra projection, nonstandard norm
    scaling such as a ``(1 + weight)`` RMSNorm, dropout before the head) makes
    the reconstruction WRONG, and a wrong lens is a confidently mislabelled
    reading of someone's model. Fail-closed: when the anchors needed to
    validate were not captured, this refuses rather than silently trusting the
    reconstruction (pass ``validate=False`` to opt out explicitly). A vision
    head validates against the last block of its own tower.
    """

    captured = head.view.get(head.logits_facet)
    if captured is None:
        raise LogitLensError(
            "Cannot validate the reconstructed lens: the model's captured head "
            "logits were not saved in this trace. Re-capture with the default "
            "exhaustive save (or a save= predicate including the head output), or "
            "pass validate=False to trust the reconstruction without a numeric check."
        )
    reference_address = _last_block_address(trace, head.tower)
    if reference_address is None:
        raise LogitLensError(
            "Cannot validate the reconstructed lens: no module exposes a captured "
            "'resid_post' facet to replay through it. Re-capture with the default "
            "exhaustive save, or pass validate=False to trust the reconstruction "
            "without a numeric check."
        )
    reference_hidden = _facet_tensor(
        trace.modules[reference_address].facets["resid_post"], name="resid_post"
    ).detach()
    reference_logits = _facet_tensor(captured, name="logits").detach()
    with torch.no_grad():
        reconstructed = lens(reference_hidden)
    comparable = _align_to_captured_slice(reconstructed, reference_logits)
    if comparable is None or not torch.allclose(comparable, reference_logits, rtol=rtol, atol=atol):
        raise LogitLensError(
            "Reconstructed lens failed validation: applying the reconstructed "
            "final norm + unembedding to the last block's captured resid_post "
            f"(shape {tuple(reconstructed.shape)}) does not reproduce the captured "
            f"logits (shape {tuple(reference_logits.shape)}, rtol={rtol}, "
            f"atol={atol}). The head does something the reconstruction does not "
            "capture (nonstandard norm scaling, an extra projection, or dropout "
            "between the last block and the head). Pass lens= explicitly or "
            "register a corrected facet recipe."
        )


def _identity_records(lens: Any, lens_by_address: Mapping[str, LensFunc]) -> list[Any]:
    """Return every ``model_identity`` attribute on the lens and its per-layer callables."""

    owners = [lens] if callable(lens) else [lens, *lens_by_address.values()]
    return [
        getattr(owner, "model_identity")
        for owner in owners
        if getattr(owner, "model_identity", None) is not None
    ]


def _recorded_lens_identity(
    lens: Any, lens_by_address: Mapping[str, LensFunc]
) -> dict[str, Any] | None:
    """Return the one model-identity record a user lens carries, or ``None``.

    The record may sit on the lens itself (a callable or the mapping) or on
    each per-layer callable; every record found must be the same.
    """

    distinct: list[dict[str, Any]] = []
    for record in _identity_records(lens, lens_by_address):
        if not isinstance(record, Mapping) or not isinstance(record.get("level"), str):
            raise LogitLensError(
                "lens.model_identity must be a model-identity record (a mapping with a "
                f"'level' entry, as in an extraction manifest), got {type(record).__name__}. "
                "Remedy: attach the record from manifest['signature']['model_identity'], "
                "or drop the attribute.",
                code="logit_lens_identity_invalid",
            )
        if dict(record) not in distinct:
            distinct.append(dict(record))
    if len(distinct) > 1:
        raise LogitLensError(
            f"The user lens carries {len(distinct)} different model-identity records "
            "across its per-layer callables, so it cannot belong to one model. Remedy: "
            "rebuild the lens from one model, or pass validate=False to use it anyway.",
            code="logit_lens_identity_mismatch",
        )
    return distinct[0] if distinct else None


def _compare_lens_identity(trace: Any, record: Mapping[str, Any]) -> tuple[str, str]:
    """Compare a lens's identity record against the traced model.

    Returns
    -------
    tuple[str, str]
        ``("match", "")``, ``("mismatch", why)`` or ``("unverifiable", why)``.
    """

    level = record.get("level")
    if level != _COMPARABLE_IDENTITY_LEVEL:
        return (
            "unverifiable",
            f"its recorded identity has level {level!r}, and only a 'measured' state "
            "digest can be compared against the traced model",
        )
    hash_name = record.get("hash", "blake2b")
    if hash_name not in _IDENTITY_HASHES:
        raise LogitLensError(
            f"lens.model_identity names hash {hash_name!r}; a measured identity record "
            f"uses one of {sorted(_IDENTITY_HASHES)}. Remedy: attach the unmodified record "
            "from manifest['signature']['model_identity'].",
            code="logit_lens_identity_invalid",
        )
    from .._data_substrate import compare_model_identity, compute_model_identity
    from .._runnable_transaction import _require_live_source_model
    from ..errors.runnable import RunCapabilityUnavailableError

    try:
        model = _require_live_source_model(trace)
    except RunCapabilityUnavailableError:
        return (
            "unverifiable",
            "the trace no longer holds its source model, so the lens's measured "
            "identity cannot be compared",
        )

    current = compute_model_identity(model, level="measured", hash_name=str(hash_name))
    verdict = compare_model_identity(record, current)
    if verdict is None:
        return "match", ""
    if verdict == "unavailable":
        return (
            "unverifiable",
            "the traced model's state cannot be measured (meta-device or offloaded entries)",
        )
    return "mismatch", verdict


def _check_user_lens(
    trace: Any,
    lens: Any,
    lens_by_address: Mapping[str, LensFunc],
    head: _DefaultHead | None,
    tolerance: tuple[float, float],
) -> None:
    """Refuse a user lens built for another model; warn when it is unverified.

    ``tolerance`` is the ``(rtol, atol)`` pair of the caller's lens validation.
    """

    record = _recorded_lens_identity(lens, lens_by_address)
    if record is None:
        reason = "it records no model identity"
    else:
        verdict, detail = _compare_lens_identity(trace, record)
        if verdict == "match":
            return
        if verdict == "mismatch":
            raise LogitLensError(
                "The user lens records a model identity that differs from the traced "
                f"model ({detail} differs), so its rows would be a confident reading of a "
                "different model. Remedy: use a lens built for this model, or pass "
                "validate=False to apply it anyway.",
                code="logit_lens_identity_mismatch",
            )
        reason = detail
    disagreement = _last_layer_disagreement(trace, lens, lens_by_address, head, tolerance)
    if disagreement is None:
        return
    warnings.warn(
        TorchLensWarning(
            f"logit_lens could not confirm the user lens belongs to this model: {reason}, "
            f"and {disagreement}. Remedy: attach the model's identity record as "
            "lens.model_identity (an extraction manifest's "
            "manifest['signature']['model_identity']), or pass validate=False to skip "
            "the check",
            code="logit_lens_identity_unrecorded",
        ),
        stacklevel=3,
    )


def _last_layer_pair(
    trace: Any, lens: Any, lens_by_address: Mapping[str, LensFunc], head: _DefaultHead | None
) -> tuple[torch.Tensor, torch.Tensor, str] | str:
    """Return ``(decoded, captured, address)`` for the agreement check, or why it cannot run."""

    captured = _read_final_logits(head)
    if captured is None:
        return "the trace has no captured head logits to check its last-layer decode against"
    address = _last_block_address(trace, None if head is None else head.tower)
    if address is None:
        return "no block exposes a captured 'resid_post' to check its last-layer decode on"
    layer_lens = lens if callable(lens) else lens_by_address.get(address)
    if layer_lens is None:
        return f"it has no entry for the last block {address!r}, so its last layer is unchecked"
    hidden = _facet_tensor(trace.modules[address].facets["resid_post"], name="resid_post")
    with torch.no_grad():
        decoded = layer_lens(hidden.detach())
    if not isinstance(decoded, torch.Tensor):
        return f"its last-layer decode returned {type(decoded).__name__}, not logits"
    return decoded, captured, address


def _last_layer_disagreement(
    trace: Any,
    lens: Any,
    lens_by_address: Mapping[str, LensFunc],
    head: _DefaultHead | None,
    tolerance: tuple[float, float],
) -> str | None:
    """Return why the lens's last-layer decode fails to agree, or ``None``.

    The check is the default lens's validation run as a check: the lens applied
    to the last block's captured ``resid_post`` must reproduce the model's
    captured logits.
    """

    pair = _last_layer_pair(trace, lens, lens_by_address, head)
    if isinstance(pair, str):
        return pair
    decoded, captured, address = pair
    rtol, atol = tolerance
    comparable = _align_to_captured_slice(decoded, captured)
    if comparable is None:
        return (
            f"its last-layer decode (shape {tuple(decoded.shape)}) is not comparable with "
            f"the captured logits (shape {tuple(captured.shape)})"
        )
    comparable = comparable.to(captured.dtype)
    if torch.allclose(comparable, captured, rtol=rtol, atol=atol):
        return None
    agree = (comparable.argmax(dim=-1) == captured.argmax(dim=-1)).float().mean().item()
    max_diff = (comparable - captured).abs().max().item()
    return (
        f"its last-layer decode on {address!r} does not reproduce the captured logits "
        f"(max |difference| {max_diff:.4g}, top-1 agreement {agree:.1%}, rtol={rtol}, "
        f"atol={atol})"
    )


def _align_to_captured_slice(
    reconstructed: torch.Tensor, captured: torch.Tensor
) -> torch.Tensor | None:
    """Return the reconstructed slice comparable to sequence-sliced captured logits.

    Current HF causal LMs (transformers 5.x ``logits_to_keep``) project only
    the LAST ``K`` sequence positions through the head, so the captured
    ``logits`` facet is ``[..., K, vocab]`` while a full-sequence
    reconstruction is ``[..., S, vocab]`` with ``K <= S``. The HF slice is
    ``hidden[:, -K:, :]``, so the comparable reconstruction is the matching
    suffix -- validation still checks real reconstructed values against real
    captured values, just on the positions the model actually projected.

    Returns
    -------
    torch.Tensor | None
        ``reconstructed`` itself on exact shape match, its sequence-suffix
        slice in the ``logits_to_keep`` case, or ``None`` when the shapes are
        not comparable either way (the caller refuses).
    """

    if reconstructed.shape == captured.shape:
        return reconstructed
    if reconstructed.ndim != captured.ndim or reconstructed.ndim < 3:
        return None
    if reconstructed.shape[:-2] != captured.shape[:-2]:
        return None
    if reconstructed.shape[-1] != captured.shape[-1]:
        return None
    kept = captured.shape[-2]
    if not 0 < kept <= reconstructed.shape[-2]:
        return None
    return reconstructed[..., -kept:, :]


def _position_vector(logits: torch.Tensor, *, position: int, batch_index: int) -> torch.Tensor:
    """Return the vocab vector at one batch element / sequence position."""

    if logits.ndim >= 3:
        return logits[batch_index, ..., position, :]
    if logits.ndim == 2:
        return logits[batch_index]
    return logits


def _decode_ids(tokenizer: Any, ids: Sequence[int]) -> list[Any]:
    """Decode token ids with whichever tokenizer surface is available."""

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        return list(convert(list(ids)))
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return [decode([token_id]) for token_id in ids]
    return list(ids)


def _facet_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Return a tensor from a facet-like value or refuse with a clear message."""

    if isinstance(value, Facet):
        value = value.value
    if isinstance(value, MissingFacet):
        raise LogitLensError(f"Facet {name!r} is unavailable: {value.reason}")
    if not isinstance(value, torch.Tensor):
        raise LogitLensError(
            f"Facet {name!r} did not produce a tensor (got {type(value).__name__})."
        )
    return value
