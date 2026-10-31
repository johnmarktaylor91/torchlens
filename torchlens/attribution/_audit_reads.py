"""Grounding a one-backward read table for the causal audit.

``tl.attribution.read`` scores every site (or every element of every site)
by activation times gradient of one target from one backward pass. Summed
over a site, that score is the first-order estimate of what zero-ablating the
site does to the target (attribution patching: Nanda, "Attribution Patching",
2023; Syed et al., "Attribution Patching Outperforms Automated Circuit
Discovery", 2023), and it is exact when the target is linear in the site.
The audit tests the estimate by doing the ablation: each ``ACT`` row's site
(site grain) or each element of it (element grain) is ablated on the audit's
intervention-ready trace and the target the read was taken for is measured
on the fork.

The read's target is an edge seed (``tl.attribution.seed(site, index=...)``
or ``cotangent=...``); the caller passes the same seed as ``target=`` and it
must match the table's recorded target. Rows whose status is not ``ok``
carry no score and are reported apart. A table of ``EDGE`` rows is not
grounded here: an edge is tested by patching it (``_audit_edges``).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch import Tensor

from torchlens.attribution._audit_claims import _AUDIT_DOCS_POINTER, _AuditRequest
from torchlens.attribution._audit_sites import _site_edits, _SitesPlan, _SiteUnit
from torchlens.attribution._result import AttributionError


def _is_read_table(claim: Any) -> bool:
    """Return whether ``claim`` is a one-backward ``ReadTable``."""

    from torchlens.attribution.onebackward._table import ReadTable

    return isinstance(claim, ReadTable)


def _recorded_target(claim: Any, target_id: str | None) -> str | None:
    """Return the table's recorded repr of ``target_id`` (``"t3"`` is the fourth target)."""

    reprs = claim.provenance.target_reprs
    if not isinstance(target_id, str) or not target_id[1:].isdigit():
        return None
    position = int(target_id[1:])
    return reprs[position] if position < len(reprs) else None


def _seed_spelling(recorded: str | None) -> str:
    """Return a runnable ``target=`` spelling for a recorded index seed, else ``target``."""

    if recorded is not None and recorded.startswith("seed(") and "index=" in recorded:
        return f"tl.attribution.{recorded}"
    return "target"


def _read_remedy(seed: str) -> str:
    """Return the runnable audit spelling with an activation-times-gradient site read."""

    return (
        f"tl.attribution.causal_audit(model, x, claim=tl.attribution.read(tl.trace(model, x), "
        f"target={seed}), target={seed}, k=1)"
    )


def _read_refusal(why: str, remedy: str, *, code: str) -> AttributionError:
    """Build a coded refusal for a read table the audit cannot measure."""

    return AttributionError(
        f"causal_audit cannot audit this read table: {why}. Remedy: {remedy}. "
        f"{_AUDIT_DOCS_POINTER}",
        code=code,
    )


def _audited_rows(claim: Any) -> tuple[str | None, list[tuple[int, Any]], list[tuple[str, str]]]:
    """Split the table into its one target, the ok ACT rows and the rows reported apart."""

    rows = claim.rows()
    targets = sorted({row.target_id for row in rows if row.target_id is not None})
    first = targets[0] if targets else None
    seed = _seed_spelling(_recorded_target(claim, first))
    if len(targets) != 1:
        raise _read_refusal(
            f"it holds rows for {len(targets)} targets {targets!r}; the audit measures one target",
            f"tl.attribution.causal_audit(model, x, claim=claim.target({first!r}), "
            f"target={seed}, k=1)"
            if first is not None
            else _read_remedy("target"),
            code="attribution_audit_claim_unsupported",
        )
    audited: list[tuple[int, Any]] = []
    apart: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        if row.status != "ok":
            apart.append((_row_site(row), row.status))
            continue
        if row.method != "activation_x_grad" or (row.grain == "site" and row.reduction != "sum"):
            raise _read_refusal(
                f"row {_row_site(row)!r} is a {row.method!r} read reduced by {row.reduction!r}; "
                "the audit compares an ablation with activation_x_grad scores summed over a "
                "site (reduce='sum') or kept per element (reduce=None)",
                _read_remedy(seed),
                code="attribution_audit_claim_unsupported",
            )
        audited.append((index, row))
    return first, audited, apart


def _row_site(row: Any) -> str:
    """Return the pass-qualified op label of an ACT row's address."""

    layer, pass_index = row.address[0], row.address[1] if len(row.address) > 1 else None
    return f"{layer}:{pass_index}" if pass_index is not None else str(layer)


def _seed_scorer(trace: Any, target: Any, recorded: str | None, seed: str) -> Any:
    """Return ``trace -> target value`` for the read's seed, or refuse a different target."""

    from torchlens.attribution.onebackward._targets import SeedTarget, _site_label

    remedy = _read_remedy(seed)
    if not isinstance(target, SeedTarget):
        raise _read_refusal(
            "target= must be the tl.attribution.seed(...) the table was read for; got "
            f"{type(target).__name__}",
            remedy,
            code="attribution_audit_claim_misaligned",
        )
    raw = _site_label(target.site)
    try:
        label = str(trace[raw].label) if raw is not None else None
    except (KeyError, ValueError, IndexError, AttributeError):
        label = None
    if label is None:
        mine = f"seed({raw!r}, ...)"
    elif target.index is not None:
        mine = f"seed({label!r}, index={target.index!r})"
    else:
        mine = f"seed({label!r}, cotangent=<{tuple(torch.as_tensor(target.cotangent).shape)}>)"
    if label is None or mine != recorded:
        raise _read_refusal(
            f"the table was read for {recorded} but target= is {mine}",
            remedy,
            code="attribution_audit_claim_misaligned",
        )
    if target.index is not None:
        index = target.index
        return lambda run: float(run[label].out.detach()[index].to(torch.float64))
    cotangent = torch.as_tensor(target.cotangent).detach().to(torch.float64)
    return lambda run: math.fsum(
        (run[label].out.detach().to(torch.float64) * cotangent).flatten().tolist()
    )


def _row_units(index: int, row: Any, site: str, value: Tensor) -> list[_SiteUnit]:
    """Return the units of one ok ACT row: the whole site, or each element after the batch axis."""

    if row.grain == "site":
        return [_SiteUnit((index,), site, (), site, float(row.score))]
    claim = torch.as_tensor(row.value).detach().to(torch.float64).sum(dim=0)
    return [
        _SiteUnit((index, *element), site, element, site, float(claim[element]))
        for element in itertools.product(*(range(size) for size in value.shape[1:]))
    ]


def _plan_read_audit(request: _AuditRequest, trace: Any) -> _SitesPlan:
    """Ground a one-backward read table on the audit's trace."""

    claim = request.claim
    target_id, audited, apart = _audited_rows(claim)
    recorded = _recorded_target(claim, target_id)
    seed = _seed_spelling(recorded)
    score = _seed_scorer(trace, request.target, recorded, seed)
    units: list[_SiteUnit] = []
    sites: dict[str, Tensor] = {}
    for index, row in audited:
        site = _row_site(row)
        try:
            value = trace[site].out
        except (KeyError, ValueError, IndexError, AttributeError):
            value = None
        if not isinstance(value, Tensor) or (
            row.shape is not None and tuple(row.shape) != tuple(value.shape)
        ):
            found = None if not isinstance(value, Tensor) else tuple(value.shape)
            raise _read_refusal(
                f"row {site!r} (shape {row.shape}) does not match the audit's trace of the "
                f"model (shape {found}), so the table was read on other inputs",
                _read_remedy(seed),
                code="attribution_audit_claim_misaligned",
            )
        sites[site] = value
        units.extend(_row_units(index, row, site, value))
    name, edits = _site_edits(
        request.ablation,
        sites,
        remedy=f"tl.attribution.causal_audit(model, x, claim=claim, target={seed}, edit=0.0)",
    )
    grain = audited[0][1].grain if audited else "site"
    return _SitesPlan(
        trace=trace,
        units=tuple(unit.key for unit in units),
        rows={unit.key: unit for unit in units},
        shapes={site: tuple(int(size) for size in value.shape) for site, value in sites.items()},
        ablation_name=name,
        edits=edits,
        score=score,
        claim_kind="read_table",
        claim_method="activation_x_grad",
        target_repr=str(recorded),
        extra={"target_id": target_id, "grain": grain, "rows_not_audited": tuple(apart)},
    )
