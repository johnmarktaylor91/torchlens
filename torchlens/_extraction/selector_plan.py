"""Selector-valued ``layers=``: resolve on batch zero, freeze, attest (D12).

``layers=`` accepts capture's selector/predicate algebra plus extraction's
string lookups and selector-valued mappings. Batch zero resolves in
EXECUTION order and is harvested from the same trace; the ordered
structural site keys and pass-qualified labels FREEZE; every later batch
re-runs the selector and must attest the exact frozen plan. Zero-site,
missing, extra, reordered, and excess-fanout results refuse typed. Mapping
selectors namespace children (``key/label`` — collision-free because user
keys refuse ``/`` at call time). Partial-unit Selections teach the
postprocess remedy. Full-forward stays the default (early halt can suppress
later model side effects); this panel supplies the frozen-plan contract,
brainpipe/capture own any faster producer.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError

__tl_layer__ = "L5"

__all__ = ["SelectorPlan", "freeze_selector_plan", "is_selector_request", "selector_request_record"]

#: Fan-out ceiling per request: a selector matching more sites than this is
#: almost certainly under-constrained, and each site becomes a stored,
#: exported, manifest-described member.
_MAX_SELECTOR_SITES = 1024


@dataclasses.dataclass(frozen=True)
class SelectorPlan:
    """The frozen batch-zero selector resolution (extract D12).

    Attributes
    ----------
    entries:
        Ordered ``(output_key, pass_qualified_label, site_key)`` rows in
        execution order.
    request_record:
        JSON-portable disclosure of the caller's selector request.
    """

    entries: tuple[tuple[str, str, str | None], ...]
    request_record: dict[str, Any]

    @property
    def output_keys(self) -> list[str]:
        """Return the plan's output keys in execution order.

        Returns
        -------
        list[str]
            One key per frozen entry.
        """

        return [key for key, _label, _site in self.entries]

    def to_record(self) -> dict[str, Any]:
        """Return the manifest/signature form of the frozen plan.

        Returns
        -------
        dict[str, Any]
            ``request`` plus the frozen ``entries`` rows.
        """

        return {
            "request": self.request_record,
            "entries": [
                {"key": key, "label": label, "site_key": site} for key, label, site in self.entries
            ],
        }


def _is_selector(value: Any) -> bool:
    """Return whether one ``layers=`` value is a capture selector.

    Parameters
    ----------
    value:
        Candidate value.

    Returns
    -------
    bool
        ``True`` for :class:`~torchlens.intervention.selectors.BaseSelector`
        instances.
    """

    from ..intervention.selectors import BaseSelector

    return isinstance(value, BaseSelector)


def _refuse_partial_units(value: Any) -> None:
    """Refuse element-level Selections with the teaching remedy (D12).

    Parameters
    ----------
    value:
        The offending Selection.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_selector_partial_units``, always.
    """

    raise InvalidArgumentError(
        f"layers= received a Selection ({value!r}); extraction stores WHOLE "
        "sites (row i of every shard is stimulus i's full activation), so an "
        "element-level selection cannot become a stored key.",
        code="extraction_selector_partial_units",
        remedy=(
            "pass the site's selector (e.g. tl.func/tl.in_module) or its "
            "label, then slice elements downstream with transform= or read "
            "the artifact and index it"
        ),
        selection=repr(value)[:200],
    )


def is_selector_request(layers: Any) -> bool:
    """Return whether the ``layers=`` request needs selector resolution.

    Parameters
    ----------
    layers:
        The caller's ``layers=`` value.

    Returns
    -------
    bool
        ``True`` for a bare selector or a mapping holding any selector
        value.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_selector_partial_units`` for element-level Selections
        (bare or as mapping values);
        ``extraction_selector_mixed_unsupported`` for mappings mixing
        string lookups and selectors (one request resolves through one
        door).
    """

    from ..selection import Selection

    if isinstance(layers, Selection):
        _refuse_partial_units(layers)
    if _is_selector(layers):
        return True
    if isinstance(layers, Mapping):
        kinds = set()
        for value in layers.values():
            if isinstance(value, Selection):
                _refuse_partial_units(value)
            kinds.add("selector" if _is_selector(value) else "string")
        if kinds == {"selector"}:
            return True
        if "selector" in kinds:
            raise InvalidArgumentError(
                "layers= mixes selector values and string lookups in one "
                "mapping; one request resolves through one door, so the "
                "frozen plan stays attestable field-by-field.",
                code="extraction_selector_mixed_unsupported",
                remedy=(
                    "split the extraction into one selector-valued call and "
                    "one string-valued call, or express the string lookups "
                    "as selectors"
                ),
            )
    return False


def selector_request_record(layers: Any) -> dict[str, Any]:
    """Build the JSON-portable disclosure of a selector request.

    The record is the RESUME identity of the request (compared
    field-by-field), so it uses the selector's stable query repr.

    Parameters
    ----------
    layers:
        Bare selector or selector-valued mapping.

    Returns
    -------
    dict[str, Any]
        ``{"kind": "selector", "query": repr}`` or
        ``{"kind": "selector_mapping", "queries": {key: repr}}``.
    """

    if isinstance(layers, Mapping):
        return {
            "kind": "selector_mapping",
            "queries": {str(key): repr(value) for key, value in layers.items()},
        }
    return {"kind": "selector", "query": repr(layers)}


def _resolve_one_selector(
    model: Any, envelope_args: tuple[Any, ...], envelope_kwargs: dict[str, Any], selector: Any
) -> list[tuple[str, str | None, torch.Tensor]]:
    """Run one selective capture and harvest its saved sites in execution order.

    Parameters
    ----------
    model:
        Model to trace.
    envelope_args:
        Positional forward args.
    envelope_kwargs:
        Keyword forward args.
    selector:
        The capture selector (``save=`` predicate).

    Returns
    -------
    list[tuple[str, str | None, torch.Tensor]]
        Ordered ``(pass_qualified_label, site_key, out)`` saved-site rows.
    """

    import torchlens as tl

    trace = tl.trace(model, envelope_args, envelope_kwargs or None, save=selector)
    return _saved_sites(trace)


def _saved_sites(trace: Any, selector: Any = None) -> list[tuple[str, str | None, torch.Tensor]]:
    """List a trace's saved sites, optionally filtered by one selector.

    Parameters
    ----------
    trace:
        A finished selective capture.
    selector:
        Optional selector re-evaluated per saved op (used by the mapping
        fast path: ONE union capture, per-key post-hoc filtering).

    Returns
    -------
    list[tuple[str, str | None, torch.Tensor]]
        Ordered ``(pass_qualified_label, site_key, out)`` rows.
    """

    from ..ir.selector_eval import evaluate

    sites: list[tuple[str, str | None, torch.Tensor]] = []
    for op in trace.saved_ops:
        if getattr(op, "is_input", False) or getattr(op, "is_output", False):
            # Input/output boundary pseudo-ops alias real sites and are
            # always retained; a selector plan freezes USER-addressed sites
            # only (the aliased producer op is already in the plan).
            continue
        if selector is not None and not evaluate(selector, op, lifecycle="site"):
            continue
        site_key: str | None
        try:
            site_key = str(op.site_key)
        except Exception:  # noqa: BLE001 - keyless legacy grouping stays plan-comparable by label
            site_key = None
        sites.append((str(op.label), site_key, op.out))
    return sites


def freeze_selector_plan(
    model: Any,
    envelope_args: tuple[Any, ...],
    envelope_kwargs: dict[str, Any],
    layers: Any,
) -> tuple[SelectorPlan, dict[str, torch.Tensor]]:
    """Resolve a selector request against batch zero and FREEZE it (D12).

    Parameters
    ----------
    model:
        Model to trace.
    envelope_args:
        Batch zero's positional args.
    envelope_kwargs:
        Batch zero's keyword args.
    layers:
        Bare selector or selector-valued mapping.

    Returns
    -------
    tuple[SelectorPlan, dict[str, torch.Tensor]]
        The frozen plan and batch zero's harvested outputs keyed by plan
        output key (the resolution trace is harvested — batch zero never
        runs twice).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_selector_no_sites`` when a selector matches nothing;
        ``extraction_selector_excess_fanout`` above the site ceiling.
    """

    request = selector_request_record(layers)
    entries: list[tuple[str, str, str | None]] = []
    outputs: dict[str, torch.Tensor] = {}
    if isinstance(layers, Mapping):
        per_key_sites = _resolve_mapping_sites(model, envelope_args, envelope_kwargs, layers)
        for key, sites in per_key_sites.items():
            if not sites:
                _refuse_no_sites(key, layers[key])
            _refuse_excess_fanout(key, sites)
            namespaced = len(sites) > 1
            for label, site_key, out in sites:
                out_key = f"{key}/{label}" if namespaced else key
                entries.append((out_key, label, site_key))
                outputs[out_key] = out
        return SelectorPlan(tuple(entries), request), outputs
    sites = _resolve_one_selector(model, envelope_args, envelope_kwargs, layers)
    if not sites:
        _refuse_no_sites(None, layers)
    _refuse_excess_fanout(None, sites)
    for label, site_key, out in sites:
        entries.append((label, label, site_key))
        outputs[label] = out
    return SelectorPlan(tuple(entries), request), outputs


def _resolve_mapping_sites(
    model: Any,
    envelope_args: tuple[Any, ...],
    envelope_kwargs: dict[str, Any],
    layers: Mapping[str, Any],
) -> dict[str, list[tuple[str, str | None, torch.Tensor]]]:
    """Resolve every mapping key's sites with as few captures as possible.

    Fast path: ONE union capture (selectors OR-composed) with per-key
    post-hoc filtering through the selector evaluator. Selectors that
    refuse OR composition (``tl.followed_by``) fall back to one capture
    per key — same plan, more forwards.

    Parameters
    ----------
    model:
        Model to trace.
    envelope_args:
        Positional forward args.
    envelope_kwargs:
        Keyword forward args.
    layers:
        Selector-valued mapping.

    Returns
    -------
    dict[str, list[tuple[str, str | None, torch.Tensor]]]
        Per-key ordered site rows.
    """

    import torchlens as tl

    selectors = {str(key): value for key, value in layers.items()}
    if len(selectors) == 1:
        key, selector = next(iter(selectors.items()))
        return {key: _resolve_one_selector(model, envelope_args, envelope_kwargs, selector)}
    union: Any = None
    try:
        for selector in selectors.values():
            union = selector if union is None else (union | selector)
    except Exception:  # noqa: BLE001 - OR-refusing selectors fall back to per-key captures
        union = None
    if union is not None:
        trace = tl.trace(model, envelope_args, envelope_kwargs or None, save=union)
        return {key: _saved_sites(trace, selector) for key, selector in selectors.items()}
    return {
        key: _resolve_one_selector(model, envelope_args, envelope_kwargs, selector)
        for key, selector in selectors.items()
    }


def _refuse_no_sites(key: str | None, selector: Any) -> None:
    """Raise the zero-site refusal (D12).

    Parameters
    ----------
    key:
        Mapping output key, when the request was a mapping.
    selector:
        The selector that matched nothing.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_selector_no_sites``, always.
    """

    where = f"for output key {key!r} " if key is not None else ""
    raise InvalidArgumentError(
        f"The layers= selector {where}matched ZERO saved sites on batch "
        f"zero ({selector!r}); an empty plan would complete an artifact "
        "with no data.",
        code="extraction_selector_no_sites",
        remedy=(
            "broaden the selector (check tl.trace(model, x, "
            "save=selector).saved_ops interactively), or name layers by label"
        ),
        key=key,
        selector=repr(selector)[:200],
    )


def _refuse_excess_fanout(
    key: str | None, sites: list[tuple[str, str | None, torch.Tensor]]
) -> None:
    """Raise the excess-fanout refusal above the site ceiling (D12).

    Parameters
    ----------
    key:
        Mapping output key, when the request was a mapping.
    sites:
        Resolved sites.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_selector_excess_fanout`` above the ceiling.
    """

    if len(sites) <= _MAX_SELECTOR_SITES:
        return
    where = f"for output key {key!r} " if key is not None else ""
    raise InvalidArgumentError(
        f"The layers= selector {where}matched {len(sites)} sites (ceiling "
        f"{_MAX_SELECTOR_SITES}); every site becomes a stored, exported, "
        "manifest-described member, so an under-constrained selector is "
        "refused rather than silently exploding the artifact.",
        code="extraction_selector_excess_fanout",
        remedy="narrow the selector (compose with tl.in_module or type filters)",
        key=key,
        n_sites=len(sites),
        ceiling=_MAX_SELECTOR_SITES,
    )


def attest_selector_batch(
    model: Any,
    call: tuple[tuple[Any, ...], dict[str, Any]],
    layers: Any,
    plan: SelectorPlan,
    batch_index: int,
) -> dict[str, torch.Tensor]:
    """Re-run the selector on one batch and attest the frozen plan (D12).

    Parameters
    ----------
    model:
        Model to trace.
    call:
        This batch's ``(positional args, keyword args)`` pair.
    layers:
        The original selector request.
    plan:
        The frozen batch-zero plan.
    batch_index:
        Zero-based batch index for refusal messages.

    Returns
    -------
    dict[str, torch.Tensor]
        This batch's outputs keyed by the plan's output keys.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_selector_plan_violated`` when this batch's resolution
        differs from the frozen plan (missing, extra, or reordered sites —
        the exact diff is named).
    """

    envelope_args, envelope_kwargs = call
    fresh_plan, outputs = freeze_selector_plan(model, envelope_args, envelope_kwargs, layers)
    if fresh_plan.entries != plan.entries:
        frozen = [list(entry) for entry in plan.entries]
        observed = [list(entry) for entry in fresh_plan.entries]
        missing = [entry for entry in frozen if entry not in observed]
        extra = [entry for entry in observed if entry not in frozen]
        raise InvalidArgumentError(
            f"Batch {batch_index} resolved a DIFFERENT site plan than the "
            f"frozen batch-zero plan ({len(missing)} missing, {len(extra)} "
            f"extra{', reordered' if not missing and not extra else ''}); "
            "rows written under a drifting plan would silently mix sites.",
            code="extraction_selector_plan_violated",
            remedy=(
                "make the model's control flow batch-invariant for the "
                "selected sites, or extract the varying sites with "
                "batch_size=1 into separate artifacts"
            ),
            batch_index=batch_index,
            missing=missing[:10],
            extra=extra[:10],
            n_frozen=len(frozen),
            n_observed=len(observed),
        )
    return outputs
