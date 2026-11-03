"""Captum bridge helpers."""

from __future__ import annotations

import contextlib
import functools
import inspect
import weakref
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from .._errors import InvalidArgumentError
from .._selection.selectors import site as site_selector
from .._user_funcs import trace as _trace
from ..intervention.spec import when as _when

# ``source_model`` stays importable from this module (audit notebook 16 calls
# ``bridge.captum.source_model``).
from ._utils import module_for_site, resolve_one_site, source_model  # noqa: F401

#: ``module_for_site`` refusals that mean "no live module owns exactly this
#: call"; each is served by splicing a hookable probe at the op instead.
_PROBE_FALLBACK_CODES = frozenset(
    {
        "bridge_module_site_unresolved",
        "bridge_module_site_ambiguous",
        "bridge_module_site_multi_call",
    }
)


def _require_captum() -> None:
    """Raise the bridge's ImportError when Captum is not installed."""

    try:
        import captum  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Captum bridge requires the `captum` extra: install torchlens[captum]."
        ) from exc


class _SiteProbe(nn.Module):
    """Identity module TorchLens splices at one op site so Captum can hook it.

    Captum's layer methods hook an ``nn.Module``; a functional op (``out +
    identity`` in a residual block) or one pass of a module that runs several
    times has no module whose hook sees exactly that tensor. The probe is that
    module: :func:`attribute` runs the model through a TorchLens capture that
    splices the probe at the op's structural site key, so a Captum forward
    hook on it reads and replaces the tensor the rest of the forward consumes.

    Parameters
    ----------
    model:
        Live source model of the trace (kept by weak reference).
    site_key:
        Structural ``site_key_v1`` of the op, valid on every forward.
    label:
        Pass-qualified layer label of the op in the originating trace.
    ordinal:
        1-based position of the op among the ops sharing its site key in one
        forward, in execution order (a site key is pass-free, so the second
        call of a repeated module shares its key with the first).
    """

    def __init__(self, model: nn.Module, site_key: str, label: str, ordinal: int = 1) -> None:
        """Remember the model and the site the probe stands at."""

        super().__init__()
        self._model_ref = weakref.ref(model)
        self.site_key = site_key
        self.label = label
        self.ordinal = ordinal

    def forward(self, out: Any) -> Any:
        """Return the site tensor unchanged (Captum's hook does the work)."""

        return out

    def extra_repr(self) -> str:
        """Name the site in ``repr``."""

        return f"label={self.label!r}, site_key={self.site_key!r}, ordinal={self.ordinal}"

    def model(self) -> nn.Module:
        """Return the live source model, refusing once it was collected."""

        model = self._model_ref()
        if model is None:
            raise ValueError(
                "This Captum site probe outlived its source model. Re-run trace and keep "
                "the model object alive while attributing."
            )
        return model


def attribute(
    trace: Any,
    method: Any,
    target: Any | None = None,
    *,
    inputs: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Captum attribution method using inputs retained by a TorchLens trace.

    When ``method.layer`` is a site probe from :func:`layer`, the forward runs
    through a TorchLens capture that splices the probe at that site, so
    Captum's layer hooks fire on exactly that tensor.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` from the model being attributed.
    method:
        Captum attribution object exposing ``attribute``.
    target:
        Captum target, forwarded as ``target=`` only when given (the
        activation and neuron classes take no ``target`` and reject one).
    inputs:
        Optional explicit Captum input (a tensor or a tuple of tensors); when
        omitted, every logged forward input is forwarded in ``forward`` order.
    **kwargs:
        Additional keyword arguments forwarded to Captum (``baselines``,
        ``neuron_selector``, ``additional_forward_args``, ...).

    Returns
    -------
    Any
        Captum attribution result.

    Raises
    ------
    ImportError
        If Captum is unavailable.
    TypeError
        If ``method`` does not expose ``attribute``.
    InvalidArgumentError
        ``bridge_captum_inputs_unresolved`` (logged inputs not replayable
        positionally) or ``bridge_captum_multi_probe_unsupported`` (more than
        one site probe in ``method.layer``).
    """

    _require_captum()
    if not hasattr(method, "attribute"):
        raise TypeError("Captum bridge expected a method object with an attribute(...) method.")

    captum_inputs = _logged_inputs(trace) if inputs is None else inputs
    if target is not None:
        kwargs["target"] = target
    probe = _single_probe(method)
    if probe is None:
        return method.attribute(captum_inputs, **kwargs)
    with _forward_through_probe(method, probe):
        return method.attribute(captum_inputs, **kwargs)


def _logged_inputs(log: Any) -> Any:
    """Return every logged forward input in the shape Captum's ``inputs`` takes.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with saved input tensors.

    Returns
    -------
    Any
        The input tensor for a one-input forward, else the tuple of input
        tensors in ``forward``'s positional order.

    Raises
    ------
    InvalidArgumentError
        ``bridge_captum_inputs_unresolved`` when an input was passed by
        keyword only, sits inside a container, was not saved, or a positional
        parameter between two logged inputs has no tensor.
    """

    tensors, problem = _positional_inputs(log)
    if problem is not None:
        raise InvalidArgumentError(
            f"Captum bridge cannot forward the logged inputs: {problem}.",
            code="bridge_captum_inputs_unresolved",
            remedy=(
                "pass inputs= explicitly (a tensor or a tuple of the positional "
                "tensors) and the rest through additional_forward_args="
            ),
        )
    return tensors[0] if len(tensors) == 1 else tensors


def _positional_inputs(log: Any) -> tuple[tuple[torch.Tensor, ...], str | None]:
    """Collect the logged inputs in ``forward`` order, or say why they cannot be."""

    by_name: dict[str, torch.Tensor] = {}
    for layer in log._layer_list:
        if not getattr(layer, "is_input", False):
            continue
        role = str(getattr(layer, "io_role", "") or "")
        parts = role.split(".")
        out = getattr(layer, "out", None)
        if len(parts) != 2 or parts[0] != "input" or not isinstance(out, torch.Tensor):
            return (), f"input {role or layer.layer_label!r} is nested or unsaved"
        by_name[parts[1]] = out
    if not by_name:
        return (), "the trace holds no saved input tensor"
    positional = _positional_names(source_model(log))
    order = [name for name in positional if name in by_name]
    if set(order) != set(by_name) or order != positional[: len(order)]:
        return (), (
            f"logged inputs {sorted(by_name)} are not the leading positional "
            f"parameters {positional}"
        )
    return tuple(by_name[name] for name in order), None


def _positional_names(model: nn.Module) -> list[str]:
    """Return ``model.forward``'s positional parameter names in order."""

    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    try:
        parameters = inspect.signature(model.forward).parameters.values()
    except (TypeError, ValueError):
        return []
    return [param.name for param in parameters if param.kind in kinds]


def layer(trace: Any, site: Any, *, pass_index: int | None = None) -> nn.Module:
    """Resolve a TorchLens site to the module Captum's layer methods hook.

    A module address returns that live module; an op label whose tensor a
    single-call module returns gives that module too (Grad-CAM's resolver:
    the outermost owner). Every other site -- an op no module returns, one
    whose owner runs more than once, or an explicit ``pass_index`` -- gives a
    hookable identity probe that :func:`attribute` splices into the forward
    at that op's structural site, so any op you pick out in TorchLens is a
    Captum layer.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` with a live source model reference.
    site:
        Module address, module pass label, op label, layer selector, or
        layer object.
    pass_index:
        1-based call of a module ``site`` that runs more than once
        (``"layer4.1.relu"`` with ``pass_index=2`` is ``"layer4.1.relu:2"``),
        or 1-based pass of a label that names several passes.

    Returns
    -------
    nn.Module
        The live PyTorch module, or a site probe to be used through
        :func:`attribute`.

    Raises
    ------
    ImportError
        If Captum is unavailable.
    InvalidArgumentError
        ``bridge_pass_index_invalid`` for a ``pass_index`` that is not a
        positive integer, is past the last pass, or is missing for a site
        that names several passes; ``bridge_pass_index_needs_module_address``
        when ``pass_index`` accompanies a single-pass site that is not a
        plain module address.
    """

    _require_captum()
    site, pass_index = _apply_pass_index(trace, site, pass_index)
    if pass_index is None:
        try:
            return module_for_site(trace, site, bridge="Captum")
        except InvalidArgumentError as exc:
            if exc.fields.get("code") not in _PROBE_FALLBACK_CODES:
                raise
    resolved = _pick_pass(resolve_one_site(trace, site), site, pass_index)
    site_key = str(resolved.site_key)
    return _SiteProbe(
        source_model(trace),
        site_key,
        str(resolved.label),
        _site_ordinal(trace, resolved, site_key),
    )


def _pick_pass(resolved: Any, site: Any, pass_index: int | None) -> Any:
    """Return the one op ``site`` names: ``resolved`` itself, or its ``pass_index``-th pass.

    Parameters
    ----------
    resolved:
        Op or layer record from the selector resolver.
    site:
        Site as passed to :func:`layer`, for messages.
    pass_index:
        1-based pass to pick when ``resolved`` carries several ops, or
        ``None``.

    Returns
    -------
    Any
        A single-pass record.

    Raises
    ------
    InvalidArgumentError
        ``bridge_pass_index_invalid`` or ``bridge_pass_index_needs_module_address``.
    """

    ops = getattr(resolved, "ops", None)
    count = len(ops) if ops is not None and hasattr(ops, "__len__") else 0
    if count < 2:
        if pass_index is not None:
            raise InvalidArgumentError(
                f"Captum layer pass_index={pass_index} needs a plain module address or a "
                f"label naming several passes as the site; got {site!r}, which names one op",
                code="bridge_pass_index_needs_module_address",
                remedy="pass a module address such as 'layer4.1.relu' with pass_index=, or "
                "the op label or 'address:pass' spelling alone",
            )
        return resolved
    if pass_index is None or pass_index > count:
        raise InvalidArgumentError(
            f"Captum site {site!r} names {count} passes; pass_index={pass_index!r} does not "
            "pick one of them",
            code="bridge_pass_index_invalid",
            remedy=f"pass pass_index= between 1 and {count}, or the pass-qualified label",
        )
    return ops[pass_index - 1]


def _site_ordinal(trace: Any, resolved: Any, site_key: str) -> int:
    """Return the 1-based execution-order rank of ``resolved`` among ops at ``site_key``.

    Parameters
    ----------
    trace:
        Trace the op was resolved in.
    resolved:
        Resolved op (or single-pass layer) record.
    site_key:
        Its structural site key.

    Returns
    -------
    int
        1 when the record carries no raw index or no earlier op shares the key.
    """

    target = getattr(resolved, "raw_index", None)
    if not isinstance(target, int):
        return 1
    earlier = 0
    for record in getattr(trace, "_layer_list", ()) or ():
        for op in getattr(record, "ops", None) or (record,):
            index = getattr(op, "raw_index", None)
            if (
                isinstance(index, int)
                and index < target
                and getattr(op, "site_key", None) == site_key
            ):
                earlier += 1
    return earlier + 1


def _apply_pass_index(trace: Any, site: Any, pass_index: int | None) -> tuple[Any, int | None]:
    """Fold ``pass_index`` into a pass-qualified module address when ``site`` is one.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` with a live source model reference.
    site:
        Site as passed to :func:`layer`.
    pass_index:
        1-based module call index, or ``None``.

    Returns
    -------
    tuple[Any, int | None]
        ``(f"{site}:{pass_index}", None)`` for a module address; otherwise
        ``(site, pass_index)`` for :func:`_pick_pass` to apply.

    Raises
    ------
    InvalidArgumentError
        ``bridge_pass_index_invalid`` when ``pass_index`` is not a positive integer.
    """

    if pass_index is None:
        return site, None
    if isinstance(pass_index, bool) or not isinstance(pass_index, int) or pass_index < 1:
        raise InvalidArgumentError(
            f"Captum layer pass_index must be a positive integer; got {pass_index!r}",
            code="bridge_pass_index_invalid",
            remedy="pass the 1-based call index of the module, such as pass_index=2",
        )
    if isinstance(site, str) and site in dict(source_model(trace).named_modules()):
        return f"{site}:{pass_index}", None
    return site, pass_index


def _single_probe(method: Any) -> _SiteProbe | None:
    """Return the one site probe among ``method.layer``, or ``None``.

    Parameters
    ----------
    method:
        Captum attribution object (``layer`` may be a module or a list).

    Returns
    -------
    _SiteProbe | None
        The probe when exactly one is present.

    Raises
    ------
    InvalidArgumentError
        ``bridge_captum_multi_probe_unsupported`` for two or more probes.
    """

    layers = getattr(method, "layer", None)
    candidates = layers if isinstance(layers, (list, tuple)) else [layers]
    probes = [candidate for candidate in candidates if isinstance(candidate, _SiteProbe)]
    if len(probes) > 1:
        raise InvalidArgumentError(
            f"Captum bridge attribute() supports one spliced site per call; method.layer "
            f"holds {len(probes)} site probes ({[probe.label for probe in probes]!r})",
            code="bridge_captum_multi_probe_unsupported",
            remedy="attribute each spliced site with its own method object",
        )
    return probes[0] if probes else None


def _spliced_forward(probe: _SiteProbe) -> Any:
    """Build the forward function that splices ``probe`` at its site.

    The returned ``forward(*args, **kwargs)`` runs the source model through
    the capture lane (``tl.trace`` retaining nothing) with the probe called on
    the ``probe.ordinal``-th op at its site key, and returns the model's own
    output with its autograd graph intact. The same lane minted the site key
    the probe carries, so the key resolves exactly as it did in the
    originating trace (``tl.record`` tracks no module stack and ``spec.bind``
    names a ``+=`` ``add`` where capture names it ``iadd``, so neither can
    serve a nested or in-place site). The forward refuses
    ``bridge_captum_site_not_reached`` when it fires the site fewer times than
    the probe's ordinal.
    """

    model = probe.model()
    seen = {"count": 0}

    def splice(out: Any, *, hook: Any) -> Any:
        """Route the ``probe.ordinal``-th firing of the site through the probe.

        The downstream forward gets a clone of what the probe returned: an
        in-place consumer (``nn.ReLU(inplace=True)`` after a residual add)
        would otherwise overwrite the tensor Captum's hook saved or injected.
        """

        seen["count"] += 1
        if seen["count"] != probe.ordinal:
            return out
        result = probe(out)
        return result.clone() if isinstance(result, torch.Tensor) else result

    spec = _when(site_selector(probe.site_key), splice)
    return functools.partial(_run_spliced, model, probe, spec, seen)


def _run_spliced(
    model: nn.Module, probe: _SiteProbe, spec: Any, seen: dict[str, int], *args: Any, **kwargs: Any
) -> Any:
    """Run one capture with the site counter reset; refuse a forward that misses the site."""

    seen["count"] = 0
    raw_output: dict[str, Any] = {}

    def keep_output(_module: nn.Module, _inputs: Any, output: Any) -> None:
        """Keep the model's own return value (autograd graph intact)."""

        raw_output["value"] = output

    handle = model.register_forward_hook(keep_output)
    try:
        captured = _trace(
            model,
            list(args) if args else None,
            input_kwargs=dict(kwargs) if kwargs else None,
            save=_save_nothing,
            intervene=spec,
        )
    finally:
        handle.remove()
    captured.cleanup()
    if seen["count"] < probe.ordinal:
        _refuse_unreached(probe, seen["count"])
    return raw_output["value"]


def _refuse_unreached(probe: _SiteProbe, count: int) -> None:
    """Raise ``bridge_captum_site_not_reached`` for a forward that fired the site ``count`` times."""

    raise InvalidArgumentError(
        f"Captum site probe {probe.label!r} (site_key {probe.site_key!r}, call "
        f"{probe.ordinal}) was never reached: this forward ran the site {count} time(s)",
        code="bridge_captum_site_not_reached",
        remedy="attribute with inputs that take the same path as the trace the "
        "site was picked from, or pick a site on the path these inputs take",
    )


def _save_nothing(ctx: Any) -> bool:
    """Save predicate for the spliced forward: retain no activation."""

    return False


@contextlib.contextmanager
def _forward_through_probe(method: Any, probe: _SiteProbe) -> Iterator[None]:
    """Point ``method`` (and nested Captum objects sharing its forward) at the spliced forward.

    Parameters
    ----------
    method:
        Captum attribution object whose ``forward_func`` is swapped for the
        duration of the block.
    probe:
        Site probe to splice.
    """

    original = method.forward_func
    spliced = _spliced_forward(probe)
    holders = [method] + [
        value
        for value in vars(method).values()
        if value is not method and getattr(value, "forward_func", None) is original
    ]
    for holder in holders:
        holder.forward_func = spliced
    try:
        yield
    finally:
        for holder in holders:
            holder.forward_func = original


__all__ = ["attribute", "layer"]
