"""Captum bridge helpers."""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterator
from typing import Any

from torch import nn

from .._errors import InvalidArgumentError

# ``source_model`` stays importable from this module (audit notebook 16 calls
# ``bridge.captum.source_model``).
from ._utils import (  # noqa: F401
    first_input_tensor,
    module_for_site,
    resolve_one_site,
    source_model,
)

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
    log: Any,
    method: Any,
    target: Any,
    *,
    inputs: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Captum attribution method using inputs retained by a TorchLens log.

    When ``method.layer`` is a site probe from :func:`layer` (an op that no
    module returns, or one pass of a repeated module), the method's forward
    function is routed through a TorchLens capture that splices the probe at
    that site for the duration of the call, so Captum's layer hooks fire on
    exactly that tensor. A live module from :func:`layer` runs the model
    directly, as before.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` from the model being attributed.
    method:
        Captum attribution object exposing ``attribute``.
    target:
        Captum target forwarded to ``method.attribute``.
    inputs:
        Optional explicit Captum input. When omitted, the first logged input
        tensor is used.
    **kwargs:
        Additional keyword arguments forwarded to Captum.

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
        ``bridge_captum_multi_probe_unsupported`` when ``method.layer`` lists
        more than one site probe (one spliced site per call).
    """

    _require_captum()

    if not hasattr(method, "attribute"):
        raise TypeError("Captum bridge expected a method object with an attribute(...) method.")

    captum_inputs = first_input_tensor(log) if inputs is None else inputs
    probe = _single_probe(method)
    if probe is None:
        return method.attribute(captum_inputs, target=target, **kwargs)
    with _forward_through_probe(method, probe):
        return method.attribute(captum_inputs, target=target, **kwargs)


def layer(log: Any, site: Any, *, pass_index: int | None = None) -> nn.Module:
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
    log:
        TorchLens ``Trace`` with a live source model reference.
    site:
        Module address, module pass label, op label, layer selector, or
        layer object.
    pass_index:
        Which call of a module ``site`` names (1-based) when it runs more
        than once; ``"layer4.1.relu"`` with ``pass_index=2`` is the same as
        the pass-qualified address ``"layer4.1.relu:2"``.

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
        positive integer; ``bridge_pass_index_needs_module_address`` when
        ``pass_index`` accompanies anything but a plain module address.
    """

    _require_captum()
    site = _apply_pass_index(log, site, pass_index)
    try:
        return module_for_site(log, site, bridge="Captum")
    except InvalidArgumentError as exc:
        if exc.fields.get("code") not in _PROBE_FALLBACK_CODES:
            raise
    resolved = resolve_one_site(log, site)
    site_key = str(resolved.site_key)
    return _SiteProbe(
        source_model(log), site_key, str(resolved.label), _site_ordinal(log, resolved, site_key)
    )


def _site_ordinal(log: Any, resolved: Any, site_key: str) -> int:
    """Return the 1-based execution-order rank of ``resolved`` among ops at ``site_key``.

    Parameters
    ----------
    log:
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
    for record in getattr(log, "layer_list", ()) or ():
        for op in getattr(record, "ops", None) or (record,):
            index = getattr(op, "raw_index", None)
            if (
                isinstance(index, int)
                and index < target
                and getattr(op, "site_key", None) == site_key
            ):
                earlier += 1
    return earlier + 1


def _apply_pass_index(log: Any, site: Any, pass_index: int | None) -> Any:
    """Fold ``pass_index`` into a pass-qualified module address.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with a live source model reference.
    site:
        Site as passed to :func:`layer`.
    pass_index:
        1-based module call index, or ``None``.

    Returns
    -------
    Any
        ``site`` unchanged, or ``f"{site}:{pass_index}"``.

    Raises
    ------
    InvalidArgumentError
        ``bridge_pass_index_invalid`` or ``bridge_pass_index_needs_module_address``.
    """

    if pass_index is None:
        return site
    if isinstance(pass_index, bool) or not isinstance(pass_index, int) or pass_index < 1:
        raise InvalidArgumentError(
            f"Captum layer pass_index must be a positive integer; got {pass_index!r}",
            code="bridge_pass_index_invalid",
            remedy="pass the 1-based call index of the module, such as pass_index=2",
        )
    modules = dict(source_model(log).named_modules())
    if not isinstance(site, str) or site not in modules:
        raise InvalidArgumentError(
            f"Captum layer pass_index={pass_index} needs a plain module address as the "
            f"site; got {site!r}, which is not a module of the source model (an op label "
            "or pass-qualified address already names one call)",
            code="bridge_pass_index_needs_module_address",
            remedy="pass a module address such as 'layer4.1.relu' with pass_index=, or the "
            "op label or 'address:pass' spelling alone",
        )
    return f"{site}:{pass_index}"


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

    Parameters
    ----------
    probe:
        Site probe to splice.

    Returns
    -------
    Callable
        ``forward(*args, **kwargs)`` running the source model through the
        sparse ``tl.record`` lane (the capture wrappers, nothing saved) with
        the probe called on the ``probe.ordinal``-th op at its site key, and
        returning the model's own output with its autograd graph intact. The
        same lane minted the site key the probe carries, so the key resolves
        exactly as it did in the originating trace.

    Raises
    ------
    InvalidArgumentError
        ``bridge_captum_site_not_reached`` (from the returned forward) when a
        forward ran fewer ops at the site than the probe's ordinal, so the
        probe never saw its tensor.
    """

    import torchlens as tl

    from ..intervention.selectors import site as site_selector
    from ..intervention.site_keys import armed_live_minter

    model = probe.model()
    seen = {"count": 0}

    def splice(out: Any, *, hook: Any) -> Any:
        """Route the ``probe.ordinal``-th firing of the site through the probe."""

        seen["count"] += 1
        return probe(out) if seen["count"] == probe.ordinal else out

    spec = tl.when(site_selector(probe.site_key), splice)

    def forward(*args: Any, **kwargs: Any) -> Any:
        """Run one forward with the site counter reset; refuse a forward that misses the site."""

        seen["count"] = 0
        # ``tl.record`` leaves the live site-key minter unarmed by design; the
        # probe addresses the op by site key, so arm one for this forward.
        with armed_live_minter():
            output, _recording = tl.record(
                model,
                list(args) if args else None,
                dict(kwargs) if kwargs else None,
                save=_save_nothing,
                intervene=spec,
                on_predicate_error="fail-fast",
                return_output=True,
            )
        if seen["count"] < probe.ordinal:
            raise InvalidArgumentError(
                f"Captum site probe {probe.label!r} (site_key {probe.site_key!r}, call "
                f"{probe.ordinal}) was never reached: this forward ran the site "
                f"{seen['count']} time(s)",
                code="bridge_captum_site_not_reached",
                remedy="attribute with inputs that take the same path as the trace the "
                "site was picked from, or pick a site on the path these inputs take",
            )
        return output

    return forward


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
