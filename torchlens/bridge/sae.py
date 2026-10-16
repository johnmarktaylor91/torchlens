"""SAE splice-experiment helper (SAE-library-agnostic).

Splicing swaps a sparse autoencoder's reconstruction back into the model at
the site the SAE was trained on and replays the downstream computation, so
the causal footprint of the SAE's features can be measured against the clean
run. This module composes existing TorchLens primitives (fork, replay,
``tl.splice_module``) into the one-call experiment; it never adds capture
machinery of its own.

Every spelling in this module is DOCUMENTED-UNSTABLE: whether the splice
experiment earns a stable public name is a UI-sprint question, and the
surface here is deliberately minimal until that ruling.

The SAE object is duck-typed: anything exposing ``encode(x) -> latents`` and
``decode(latents) -> reconstruction`` works (an SAE Lens ``SAE``, or any
hand-built module). No third-party SAE package is required.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._errors import InvalidArgumentError
from ..errors import TorchLensWarning
from . import _sae_metrics
from ._sae_export import write_delphi, write_sae_dashboard
from ._sae_latents import logged_splice_trace, register, unregister
from ._utils import resolve_one_site

if TYPE_CHECKING:
    from collections.abc import Callable

#: Kept-latents size above which ``splice`` warns once per session: the latents
#: are always kept by default, and past this size the warning names the cheaper
#: spelling. The same 256 MiB guard ``tl.stats`` uses for per-unit state.
_LATENTS_WARN_BYTES = 256 * 1024**2

#: Warning codes already emitted this session (each code fires once).
_WARNED_CODES: set[str] = set()

_DOCS = "docs/recipes/pretrained_saes.md"


def _require_codec(sae: Any) -> None:
    """Refuse SAE objects without an encode/decode pair.

    Parameters
    ----------
    sae:
        Candidate SAE object.

    Raises
    ------
    TypeError
        If ``sae`` does not expose callable ``encode`` and ``decode``.
    """

    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise TypeError(
            "SAE splice requires an SAE exposing encode(...) and decode(...); "
            f"got {type(sae).__name__!r}. Any object with that pair works -- "
            "an SAE Lens SAE, or a hand-built module."
        )


def _keep_leading_positions(
    replacement: torch.Tensor, out: torch.Tensor, start: int
) -> torch.Tensor:
    """Return ``replacement`` with sequence positions before ``start`` taken from ``out``.

    Parameters
    ----------
    replacement:
        Tensor replacing the site activation, ``(batch, sequence, ...)``.
    out:
        Clean site activation of the same shape.
    start:
        Number of leading sequence positions kept clean (0 keeps none).

    Returns
    -------
    torch.Tensor
        ``replacement`` itself when ``start`` is 0, otherwise a patched copy.
    """

    if start == 0:
        return replacement
    patched = replacement.clone()
    patched[:, :start] = out[:, :start]
    return patched


class _SAEReconstruction(nn.Module):
    """Shape-preserving encode/decode splice around a duck-typed SAE.

    Parameters
    ----------
    sae:
        Object exposing ``encode`` and ``decode``.
    latents_edit:
        Optional transform applied to the encoded latents before decoding.
    start:
        Leading sequence positions left clean (1 keeps the BOS position).
    """

    def __init__(
        self,
        sae: Any,
        latents_edit: Callable[[torch.Tensor], torch.Tensor] | None,
        start: int = 0,
    ) -> None:
        """Store the SAE, optional latent transform and kept positions."""

        super().__init__()
        self._sae = sae
        self._latents_edit = latents_edit
        self._start = start

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        """Return the (optionally latent-edited) SAE reconstruction of ``out``.

        Parameters
        ----------
        out:
            Site activation being replaced.

        Returns
        -------
        torch.Tensor
            Reconstruction cast back to the site's dtype and device.
        """

        _latents, _edited, reconstruction = _encode_decode(
            self._sae, out, self._latents_edit, self._start
        )
        return reconstruction

    def logged(
        self, out: torch.Tensor, keep_latents: Callable[[torch.Tensor], None]
    ) -> torch.Tensor:
        """Return the reconstruction of ``out``, handing the encoder output to ``keep_latents``.

        The same arithmetic as ``forward``, run under capture by the logged
        splice so every op is a trace node.
        """

        latents, _edited, reconstruction = _encode_decode(
            self._sae, out, self._latents_edit, self._start
        )
        keep_latents(latents)
        return reconstruction


class _ZeroAblation(nn.Module):
    """Replace the site with zeros, keeping leading sequence positions clean.

    Parameters
    ----------
    start:
        Leading sequence positions left clean.
    """

    def __init__(self, start: int) -> None:
        """Store the kept positions."""

        super().__init__()
        self._start = start

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        """Return zeros shaped like ``out`` (leading positions clean).

        Parameters
        ----------
        out:
            Site activation being replaced.

        Returns
        -------
        torch.Tensor
            Zero-ablated activation.
        """

        return _keep_leading_positions(torch.zeros_like(out), out, self._start)


def _encode_decode(
    sae: Any,
    out: torch.Tensor,
    latents_edit: Callable[[torch.Tensor], torch.Tensor] | None,
    start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return the latents, the edited latents and the reconstruction of ``out``.

    Parameters
    ----------
    sae:
        Duck-typed SAE.
    out:
        Site activation being reconstructed.
    latents_edit:
        Optional transform applied to the encoded latents before decoding.
    start:
        Leading sequence positions whose clean value replaces the
        reconstruction.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
        Encoder output exactly as returned, the latents after
        ``latents_edit`` (``None`` without one), and the reconstruction cast
        to the site's dtype and device.

    Raises
    ------
    TypeError
        If ``encode`` or ``decode`` returns a non-tensor.
    """

    latents = sae.encode(out)
    edited = latents_edit(latents) if latents_edit is not None else None
    reconstruction = sae.decode(edited if edited is not None else latents)
    if not isinstance(reconstruction, torch.Tensor):
        raise TypeError(
            f"SAE decode(...) must return a torch.Tensor; got {type(reconstruction).__name__!r}"
        )
    if not isinstance(latents, torch.Tensor):
        raise TypeError(
            f"SAE encode(...) must return a torch.Tensor; got {type(latents).__name__!r}"
        )
    reconstruction = reconstruction.to(device=out.device, dtype=out.dtype)
    return latents, edited, _keep_leading_positions(reconstruction, out, start)


@dataclass(frozen=True)
class SpliceResult:
    """Settled record of one SAE splice experiment.

    Reconstruction metrics follow the SAELens evals and the Gemma Scope
    paper (Lieberum et al., 2024): fraction of variance explained with
    per-dimension centring, and cross-entropy loss recovered against a
    zero-ablated site. Every float is computed in float64. A metric that
    cannot be computed is ``None`` (``nan`` for the variance-explained
    floats), with the reason in ``unavailable_metrics``; none is a silent
    0.0.

    Parameters
    ----------
    site:
        Resolved pass-qualified layer label the reconstruction replaced.
    clean:
        Clean capture the experiment forked from.
    spliced:
        Replayed fork carrying the spliced computation.
    clean_site_out:
        Clean activation at the splice site.
    reconstruction:
        SAE reconstruction that replaced it (after any latent edit; with
        ``exclude_bos=True`` the BOS position holds the clean value).
    reconstruction_mse:
        Mean squared error between reconstruction and clean activation.
    fraction_variance_explained:
        ``1 - SSE / SST`` with SST centred on the single scalar mean of every
        element of the clean activation; ``nan`` when SST is zero.
    clean_outputs:
        Clean model-output tensors in output order.
    spliced_outputs:
        Spliced model-output tensors in the same order.
    output_delta_l2:
        L2 norm of the concatenated output difference.
    output_delta_max:
        Largest absolute per-element output difference.
    latents:
        The SAE encoder output for the clean site activation, exactly as
        ``sae.encode`` returned it (every position, BOS included);
        ``None`` with ``keep_latents=False``.
    edited_latents:
        The latents after ``latents_edit``, which the decoder consumed;
        ``None`` without an edit or with ``keep_latents=False``.
    fraction_variance_explained_pooled:
        The standard reading: ``1 - sum_d SSE_d / sum_d SST_d`` where each
        feature dimension ``d`` (the last axis) is centred on its own mean
        over tokens. This is SAELens's explained variance and one minus
        the Gemma Scope fraction of variance unexplained. ``nan`` when no
        dimension varies (for example a single token).
    fraction_variance_explained_by_dim:
        ``1 - SSE_d / SST_d`` for each feature dimension, float64, ``nan``
        where that dimension does not vary.
    zero_variance:
        ``True`` when ``fraction_variance_explained`` or
        ``fraction_variance_explained_pooled`` is ``nan`` because its
        denominator is zero.
    ce_loss_clean:
        Mean next-token cross-entropy of the clean run (``tokens=`` given).
    ce_loss_spliced:
        The same with the SAE reconstruction spliced in.
    ce_loss_zero_ablated:
        The same with the site zero-ablated.
    ce_loss_recovered:
        ``(CE_zero - CE_spliced) / (CE_zero - CE_clean)``: 1.0 when the
        reconstruction loses nothing, 0.0 when it is no better than zeros.
    kl_divergence:
        Mean ``KL(P_clean || P_spliced)`` of the next-token distributions,
        in nats.
    unavailable_metrics:
        Metric name to the reason it is ``None`` or ``nan``.
    latents_site:
        With ``log_latents=True``, the label of the latents node in
        ``spliced`` (the encoder output, between the encoder and decoder
        ops); ``None`` otherwise.
    """

    site: str
    clean: Any
    spliced: Any
    clean_site_out: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_mse: float
    fraction_variance_explained: float
    clean_outputs: tuple[torch.Tensor, ...]
    spliced_outputs: tuple[torch.Tensor, ...]
    output_delta_l2: float
    output_delta_max: float
    latents: torch.Tensor | None
    edited_latents: torch.Tensor | None
    fraction_variance_explained_pooled: float
    fraction_variance_explained_by_dim: torch.Tensor
    zero_variance: bool
    ce_loss_clean: float | None
    ce_loss_spliced: float | None
    ce_loss_zero_ablated: float | None
    ce_loss_recovered: float | None
    kl_divergence: float | None
    unavailable_metrics: Mapping[str, str]
    latents_site: str | None = None


def _tensor_outputs(log: Any) -> tuple[torch.Tensor, ...]:
    """Return every saved tensor model output of a trace in output order.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Saved output tensors.

    Raises
    ------
    ValueError
        If no model output carries a saved tensor payload.
    """

    outputs = tuple(
        op.out for op in log.output_ops if isinstance(getattr(op, "out", None), torch.Tensor)
    )
    if not outputs:
        raise ValueError(
            "SAE splice needs saved tensor model outputs to compare runs; "
            "this capture retained none. Capture with the default exhaustive "
            "save (tl.trace(model, x, capture=tl.options.CaptureOptions("
            "intervention_ready=True)))."
        )
    return outputs


def _warn_once(code: str, message: str, site: str, *, stacklevel: int) -> None:
    """Emit one coded ``TorchLensWarning`` per session per code.

    Parameters
    ----------
    code:
        Stable warning code.
    message:
        Full message, ending in ``Remedy: ...``.
    site:
        Splice site label the warning is about.
    stacklevel:
        Frames up to the caller of ``splice``, counted from here.
    """

    if code in _WARNED_CODES:
        return
    _WARNED_CODES.add(code)
    warnings.warn(
        TorchLensWarning(message, code=code, affected_sites=[site]), stacklevel=stacklevel
    )


def _warn_large_latents(site: str, latents: torch.Tensor, edited: torch.Tensor | None) -> None:
    """Warn once per session when the kept latents pass ``_LATENTS_WARN_BYTES``.

    Parameters
    ----------
    site:
        Splice site label.
    latents:
        Kept encoder output.
    edited:
        Kept edited latents, if any.
    """

    code = "sae_splice_latents_large"
    total = sum(t.numel() * t.element_size() for t in (latents, edited) if t is not None)
    if total <= _LATENTS_WARN_BYTES:
        return
    _warn_once(
        code,
        f"splice at {site!r} keeps the SAE latents in its result: "
        f"{total / 1024**2:.1f} MiB, above the {_LATENTS_WARN_BYTES // 1024**2} MiB guard; "
        "every latent is kept, nothing is dropped. This warning fires once per session; "
        f"see {_DOCS}. Remedy: splice(..., keep_latents=False), then encode only what you "
        "read with sae.encode(result.clean_site_out[...])",
        site,
        stacklevel=4,
    )


def _warn_zero_variance(site: str, reasons: Mapping[str, str]) -> None:
    """Warn once per session that a variance-explained reading is ``nan``.

    Parameters
    ----------
    site:
        Splice site label.
    reasons:
        Metric name to the reason it is ``nan``.
    """

    code = "sae_splice_zero_variance"
    _warn_once(
        code,
        f"splice at {site!r}: {'; '.join(reasons.values())}, so {' and '.join(reasons)} "
        "is nan and result.zero_variance is True. This warning fires once per session; "
        f"see {_DOCS}. Remedy: splice a batch of several distinct inputs (or a multi-token "
        "sequence)",
        site,
        stacklevel=4,
    )


def _sequence_start(site: str, out: torch.Tensor, exclude_bos: bool) -> int:
    """Return the first sequence position the metrics count (0, or 1 for BOS).

    Parameters
    ----------
    site:
        Splice site label.
    out:
        Clean site activation.
    exclude_bos:
        Whether position 0 of the sequence axis is left out.

    Returns
    -------
    int
        ``1`` with ``exclude_bos``, else ``0``.

    Raises
    ------
    InvalidArgumentError
        If ``exclude_bos`` is set on a site without a ``(batch, sequence,
        features)`` layout or with a single position.
    """

    if not exclude_bos:
        return 0
    if out.ndim < 3 or out.shape[1] < 2:
        raise InvalidArgumentError(
            f"splice(..., exclude_bos=True) at {site!r} needs a (batch, sequence, features) "
            f"activation with at least two positions; this site has shape {tuple(out.shape)}",
            code="sae_splice_bos_needs_sequence_axis",
            remedy="drop exclude_bos=True for sites without a sequence axis, "
            f"or splice a token-level site; see {_DOCS}",
        )
    return 1


def _clean_trace(model_or_log: Any, inputs: Any) -> Any:
    """Return the clean intervention-ready trace the experiment forks from.

    Parameters
    ----------
    model_or_log:
        Live module or captured trace.
    inputs:
        Model inputs (module case only).

    Returns
    -------
    Any
        Clean ``Trace``.

    Raises
    ------
    TypeError
        If ``inputs`` disagrees with the ``model_or_log`` kind.
    """

    if isinstance(model_or_log, nn.Module):
        if inputs is None:
            raise TypeError("SAE splice from a live model requires `inputs`.")
        import torchlens as tl

        return tl.trace(
            model_or_log,
            inputs,
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )
    if inputs is not None:
        raise TypeError(
            "SAE splice from an existing Trace takes no `inputs`; the fork replays the captured "
            "inputs."
        )
    return model_or_log


def _replayed(clean: Any, site: str, name: str, module: nn.Module) -> Any:
    """Fork ``clean``, replace ``site`` with ``module``'s output and push downstream.

    Parameters
    ----------
    clean:
        Clean trace.
    site:
        Resolved site label.
    name:
        Fork name.
    module:
        Shape-preserving replacement module.

    Returns
    -------
    Any
        Replayed fork.
    """

    from torchlens.intervention import label as label_selector, splice_module

    fork = clean.fork(name)
    fork.attach_hooks(label_selector(site), splice_module(module, input="out"))
    fork.push()
    return fork


def _logits_index(site: str, outputs: tuple[torch.Tensor, ...], tokens: torch.Tensor) -> int:
    """Return the index of the first model output shaped ``tokens.shape + (vocab,)``.

    Parameters
    ----------
    site:
        Splice site label.
    outputs:
        Clean model outputs.
    tokens:
        Caller token ids.

    Returns
    -------
    int
        Output index read as next-token logits.

    Raises
    ------
    InvalidArgumentError
        If ``tokens`` is not an integer tensor or no output matches its shape.
    """

    shapes = [tuple(out.shape) for out in outputs]
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.dtype.is_floating_point
        or tokens.dtype.is_complex
        or tokens.dtype == torch.bool
        or tokens.ndim < 1
    ):
        problem = f"splice(..., tokens=...) at {site!r} got {type(tokens).__name__}"
        if isinstance(tokens, torch.Tensor):
            problem += f" of dtype {tokens.dtype} and shape {tuple(tokens.shape)}"
        index = None
    else:
        index = next(
            (i for i, out in enumerate(outputs) if tuple(out.shape[:-1]) == tuple(tokens.shape)),
            None,
        )
        problem = (
            f"splice(..., tokens=...) at {site!r}: no model output is shaped tokens.shape + "
            f"(vocab,) = {tuple(tokens.shape)} + (vocab,); the outputs are {shapes}"
        )
    if index is None:
        raise InvalidArgumentError(
            problem,
            code="sae_splice_tokens_mismatch",
            remedy="pass the integer token ids the model ran on (for a Hugging Face causal LM, "
            f"the input_ids), shaped like the logits without their vocab axis; see {_DOCS}",
        )
    return index


def _next_token_metrics(
    clean: Any,
    site: str,
    tokens: torch.Tensor | None,
    start: int,
    outputs: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Return the CE triple, CE loss recovered and KL, with reasons for any ``None``.

    Parameters
    ----------
    clean:
        Clean trace (forked again for the zero-ablation baseline).
    site:
        Resolved site label.
    tokens:
        Caller token ids, or ``None``.
    start:
        First predicting position (1 skips BOS).
    outputs:
        Clean and spliced model outputs.

    Returns
    -------
    tuple[dict[str, float | None], dict[str, str]]
        Metric values and the reasons for those that are ``None``.
    """

    names = (
        "ce_loss_clean",
        "ce_loss_spliced",
        "ce_loss_zero_ablated",
        "ce_loss_recovered",
        "kl_divergence",
    )
    values: dict[str, float | None] = dict.fromkeys(names)
    if tokens is None:
        reason = (
            "no tokens= given; pass the token ids the model ran on to compute next-token "
            "cross-entropy and KL from the logits output"
        )
        return values, dict.fromkeys(names, reason)
    index = _logits_index(site, outputs[0], tokens)
    clean_logits, spliced_logits = outputs[0][index], outputs[1][index]
    values["kl_divergence"] = _sae_metrics.next_token_kl(clean_logits, spliced_logits, start)
    reasons: dict[str, str] = {}
    if tokens.shape[-1] < start + 2:
        reason = (
            f"a next-token loss needs at least {start + 2} positions; tokens has {tokens.shape[-1]}"
        )
        return values, dict.fromkeys(names[:4], reason)
    zero_outputs = _tensor_outputs(
        _replayed(clean, site, "sae_zero_ablation", _ZeroAblation(start))
    )
    ce_clean = _sae_metrics.next_token_ce(clean_logits, tokens, start)
    ce_spliced = _sae_metrics.next_token_ce(spliced_logits, tokens, start)
    ce_zero = _sae_metrics.next_token_ce(zero_outputs[index], tokens, start)
    values.update(
        ce_loss_clean=ce_clean,
        ce_loss_spliced=ce_spliced,
        ce_loss_zero_ablated=ce_zero,
        ce_loss_recovered=_sae_metrics.ce_loss_recovered(ce_clean, ce_spliced, ce_zero),
    )
    if values["ce_loss_recovered"] is None:
        reasons["ce_loss_recovered"] = (
            "zero-ablating the site leaves the loss unchanged, so the recovered fraction is undefined"
        )
    return values, reasons


def _output_deltas(
    clean_outputs: tuple[torch.Tensor, ...], spliced_outputs: tuple[torch.Tensor, ...]
) -> tuple[float, float]:
    """Return the L2 norm and max absolute value of the output difference."""

    deltas = [
        (spliced_out.detach().float() - clean_out.detach().float()).flatten()
        for clean_out, spliced_out in zip(clean_outputs, spliced_outputs)
    ]
    flat_delta = torch.cat(deltas) if deltas else torch.zeros(0)
    peak = float(flat_delta.abs().max()) if flat_delta.numel() else 0.0
    return float(flat_delta.norm()), peak


def splice(  # noqa: PLR0913 -- the public experiment door: each keyword is one documented splice input (site, SAE, latent edit, next-token ids, BOS handling, latent retention, latent logging)
    model_or_log: Any,
    inputs: Any = None,
    *,
    site: Any,
    sae: Any,
    latents_edit: Callable[[torch.Tensor], torch.Tensor] | None = None,
    tokens: torch.Tensor | None = None,
    exclude_bos: bool = False,
    keep_latents: bool = True,
    log_latents: bool = False,
) -> SpliceResult:
    """Run the standard SAE splice experiment at one site.

    Captures (or reuses) a clean intervention-ready trace, forks it, replaces
    the site activation with the SAE's reconstruction via
    ``tl.splice_module``, replays the downstream computation, and reports
    reconstruction fidelity, output-level causal effect and, given
    ``tokens=``, next-token cross-entropy loss recovered and KL divergence.

    Parameters
    ----------
    model_or_log:
        Live ``nn.Module`` (captured here with ``intervention_ready=True``)
        or an already-captured intervention-ready ``Trace``.
    inputs:
        Model inputs, required when ``model_or_log`` is a module and refused
        when it is a trace.
    site:
        Layer label, selector, or layer object naming the splice site.
    sae:
        Duck-typed SAE exposing ``encode`` and ``decode``.
    latents_edit:
        Optional transform applied to the encoded latents before decoding --
        the causal-testing knob (ablate or steer individual features).
    tokens:
        Integer token ids the model ran on. The first model output shaped
        ``tokens.shape + (vocab,)`` is read as next-token logits; one more
        replay with the site zero-ablated gives the CE baseline. Without it
        the CE and KL fields are ``None`` with a reason.
    exclude_bos:
        Leave sequence position 0 out: it keeps its clean value in the
        spliced and zero-ablated runs and is excluded from every metric
        (the Gemma Scope evaluation convention). Needs a ``(batch,
        sequence, features)`` site.
    keep_latents:
        Keep the latents in the result (default). Past 256 MiB a warning
        fires once per session; ``keep_latents=False`` drops them.
    log_latents:
        Off by default. On, ``spliced`` is a fresh capture of the live model
        with the SAE run inside it as ordinary ops, so its encoder ops, the
        latents and its decoder ops are trace nodes between the site and its
        consumers (``result.latents_site`` names the latents node). The
        spliced outputs are the same as with it off. Needs the live-model
        door (``splice(model, inputs, ...)``) and a site some module call
        returns.

    Returns
    -------
    SpliceResult
        Frozen experiment record; ``result.spliced`` is the replayed fork.

    Raises
    ------
    TypeError
        If ``sae`` lacks an encode/decode pair, or ``inputs`` disagrees with
        the ``model_or_log`` kind.
    InvalidArgumentError
        If ``tokens`` matches no model output (``sae_splice_tokens_mismatch``)
        or ``exclude_bos`` meets a site without a sequence axis
        (``sae_splice_bos_needs_sequence_axis``); with ``log_latents=True``,
        a Trace door (``sae_splice_logged_needs_live_model``) or a site no
        single module call returns (``bridge_module_site_*``).
    """

    _require_codec(sae)
    if log_latents and not isinstance(model_or_log, nn.Module):
        raise InvalidArgumentError(
            "splice(..., log_latents=True) captures the live model again with the SAE inside "
            "it, so it needs the model and its inputs; it got a Trace",
            code="sae_splice_logged_needs_live_model",
            remedy="call splice(model, inputs, site=..., sae=..., log_latents=True) with the "
            f"live model; see {_DOCS}",
        )
    clean = _clean_trace(model_or_log, inputs)
    site_label = str(resolve_one_site(clean, site).label)
    clean_site_out = clean[site_label].out
    start = _sequence_start(site_label, clean_site_out, exclude_bos)
    with torch.no_grad():
        latents, edited, reconstruction = _encode_decode(sae, clean_site_out, latents_edit, start)
    counted = slice(start, None)
    variance = _sae_metrics.variance_metrics(
        clean_site_out[:, counted] if start else clean_site_out,
        reconstruction[:, counted] if start else reconstruction,
    )
    if variance.zero_variance:
        _warn_zero_variance(site_label, variance.reasons)
    latents_site = None
    if log_latents:
        spliced, latents_site = logged_splice_trace(
            model_or_log, inputs, clean, site_label, _SAEReconstruction(sae, latents_edit, start)
        )
    else:
        spliced = _replayed(
            clean, site_label, "sae_splice", _SAEReconstruction(sae, latents_edit, start)
        )
    clean_outputs = _tensor_outputs(clean)
    spliced_outputs = _tensor_outputs(spliced)
    next_token, token_reasons = _next_token_metrics(
        clean, site_label, tokens, start, (clean_outputs, spliced_outputs)
    )
    if keep_latents:
        _warn_large_latents(site_label, latents, edited)
    delta_l2, delta_max = _output_deltas(clean_outputs, spliced_outputs)
    return SpliceResult(
        site=site_label,
        clean=clean,
        spliced=spliced,
        clean_site_out=clean_site_out,
        reconstruction=reconstruction,
        reconstruction_mse=variance.mse,
        fraction_variance_explained=variance.global_fve,
        clean_outputs=clean_outputs,
        spliced_outputs=spliced_outputs,
        output_delta_l2=delta_l2,
        output_delta_max=delta_max,
        latents=latents if keep_latents else None,
        edited_latents=edited if keep_latents else None,
        fraction_variance_explained_pooled=variance.pooled_fve,
        fraction_variance_explained_by_dim=variance.fve_by_dim,
        zero_variance=variance.zero_variance,
        unavailable_metrics={**variance.reasons, **token_reasons},
        latents_site=latents_site,
        **next_token,
    )


__all__ = [
    "SpliceResult",
    "register",
    "splice",
    "unregister",
    "write_delphi",
    "write_sae_dashboard",
]
