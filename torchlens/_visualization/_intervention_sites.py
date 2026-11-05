"""Intervention-site lookup shared by the draw path and the interventions view.

Resolves a trace's intervention spec targets to the forward ops they mark.
It reaches the selector resolver through ``Trace._resolve_sites`` (the same
call ``_selection.resolver._resolve_sites`` makes), so the interventions view
and its census can use it without importing the render stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .._utils._multipass_access import get_multipass_attr

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def intervention_sites_for_log(trace: Trace) -> list[Op]:
    """Resolve intervention spec targets to layer-pass records.

    Parameters
    ----------
    trace:
        Log whose intervention spec should be inspected.

    Returns
    -------
    list[Op]
        Distinct intervention sites in execution order.
    """

    spec = getattr(trace, "_intervention_spec", None)
    if spec is None:
        return []
    targets: list[Any] = []
    targets.extend(getattr(spec, "targets", ()) or ())
    targets.extend(
        value_spec.site_target for value_spec in getattr(spec, "target_value_specs", ()) or ()
    )
    targets.extend(hook_spec.site_target for hook_spec in getattr(spec, "hook_specs", ()) or ())
    if not targets:
        return []

    by_label: dict[str, Op] = {}
    for target in targets:
        table = trace._resolve_sites(target, max_fanout=max(1, len(trace._layer_list)))
        for site in table:
            forward_site = _forward_site(site)
            # Key by the pass-qualified label (op.label, e.g. relu_1_1:2) so
            # recurrent passes stay DISTINCT sites. The old aggregate layer_label
            # collapsed :1/:2/:3 into one site, so resolving pass 2 later colored
            # every pass. Fall back to layer_label only when no pass label exists.
            site_label = get_multipass_attr(forward_site, "label", None, multipass=None)
            if not isinstance(site_label, str):
                site_label = getattr(forward_site, "layer_label", None)
            if isinstance(site_label, str):
                by_label.setdefault(site_label, cast("Op", forward_site))
    execution_order = {_site_sort_key(op): index for index, op in enumerate(trace._layer_list)}
    return sorted(by_label.values(), key=lambda site: execution_order.get(_site_sort_key(site), 0))


def _forward_site(site: Any) -> Any:
    """Return the forward op a resolved intervention site marks, at its own pass.

    A GradFn differentiates one pass: resolve that pass, not the multi-pass
    Layer its layer-label ``op`` names. Other sites mark their ``op`` (a hook
    site) or themselves.
    """

    from .._data_classes.grad_fn import GradFn
    from .._ir._selector_universe import _grad_fn_forward_op

    if isinstance(site, GradFn):
        return _grad_fn_forward_op(site)
    return getattr(site, "op", site)


def _site_sort_key(op: Any) -> str:
    """Return an op's pass-qualified label for stable per-pass site ordering."""

    label = get_multipass_attr(op, "label", None, multipass=None)
    if isinstance(label, str):
        return label
    return str(getattr(op, "layer_label", ""))
