"""Per-query-head patching under grouped-query attention (an op-level interceptor).

Under grouped-query attention the ``k``/``v`` facets hold one tensor per KV
group, shared by several query heads, so a ``k``/``v``-site grid can only
patch a whole group. Each query head still reads its own COPY of its group:
the attention code expands the grouped tensor to the query-head count
(``repeat_kv`` in Hugging Face models: index, expand, reshape) before the
scores or the value mix. Patching one query head's slice of that expanded op
measures that head alone and leaves the other heads of its group untouched.

The interceptor finds the expansion op in the captured graph (the first
query-head-shaped descendant of the facet that is a single-parent reshape
whose head blocks are exact copies), selects that one call in the rerun, and
patches through the ordinary rerun route of :func:`torchlens.mechinterp.patch_grid`:
the same fire ledger (exactly one fire per cell, or a refusal), the same
pre-run cost line and ``budget=``, the same transactional model and RNG
restore. Fused kernels that expand inside the kernel (torch SDPA with
``enable_gqa=True``) expose no expansion op and refuse typed.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from .._semantic.patching import (
    _baseline_traces,
    _CounterfactualStateGuard,
    _metric_scalar,
    _modules_with_facet,
    _run_patch,
    _teardown,
    _warn_if_campaign_all_identical,
)
from ..intervention.selectors import (
    func as func_selector,
    in_module as in_module_selector,
    where as where_selector,
)
from ._errors import refuse
from ._grids import PatchGrid, _baseline_metrics, _disclose_cost_and_enforce_budget

__all__ = ["patch_query_heads_grid"]

#: Single-parent ops that produce a grouped tensor's query-head copies.
_EXPANSION_FUNCS = frozenset({"reshape", "view", "flatten", "repeat_interleave", "repeat"})

_SITE_WORDS = {"k": "key", "v": "value"}

#: Suffix live op records carry on their capture-order raw label.
_RAW_SUFFIX = "_raw"

_UNAVAILABLE_REMEDY = (
    'load the model with attn_implementation="eager", or switch it to the registered '
    '"torchlens" attention backend (tl.bridge.transformers.register_attention(model)); both expand '
    "the grouped key/value tensor as a captured op"
)


@dataclass(frozen=True)
class _Expansion:
    """One attention module's located expansion op and its clean value."""

    address: str
    op_label: str
    raw_label: str
    func_name: str
    clean_value: torch.Tensor
    n_q_heads: int
    n_kv_heads: int


def _effect_site_wording(site: str) -> str:
    """Return the receipt's ``effect_site`` wording for a ``k``/``v`` interceptor grid."""

    return f"expanded {_SITE_WORDS[site]} copy read by one query head (op-level interceptor)"


def patch_query_heads_grid(  # noqa: PLR0913 -- mirrors the grid driver's signature
    model: Any,
    clean_input: Any,
    corrupted_input: Any,
    metric: Any,
    *,
    site: str = "v",
    heads: Any = None,
    budget: int | None = None,
    trace_kwargs: Any = None,
) -> PatchGrid:
    """Run a ``[layer, head]`` patching grid over QUERY heads at a k/v site.

    Each cell patches the CLEAN value of one query head's copy of its KV
    group into a CORRUPTED rerun (the denoising direction of
    :func:`torchlens.mechinterp.patch_grid`) and scores the run with ``metric``.

    Parameters
    ----------
    model:
        Grouped-query attention model to trace and rerun (state restored).
    clean_input / corrupted_input:
        The two prompts; equal lengths are required.
    metric:
        Callable ``Trace -> scalar tensor``.
    site:
        ``"v"`` (default) patches the expanded value copy, ``"k"`` the
        expanded key copy.
    heads:
        Query-head indices (default: all query heads).
    budget:
        Hard cell-count ceiling; exceeding it refuses before any rerun.
    trace_kwargs:
        Extra kwargs forwarded to ``tl.trace``.

    Returns
    -------
    PatchGrid
        Axes ``("layer", "head")``; head coordinates are query heads. The
        receipts add ``effect_site`` (the interceptor wording), ``site``,
        ``n_q_heads``, ``n_kv_heads`` and ``expansion_ops`` (module address
        to the clean trace's expansion op label). ``fires`` equals the cell
        count: a cell whose rerun fires other than once refuses. As for
        every grid, ``values`` are patched-run metrics and the receipts carry
        ``clean_metric`` and ``corrupted_metric`` (the unpatched baselines).

    Raises
    ------
    MechInterpError
        ``mi_kv_heads_not_grouped`` when query and KV head counts are equal,
        ``mi_kv_expansion_unavailable`` when no expansion op exists (fused
        kernels), ``mi_kv_expansion_ambiguous`` when more than one op (or op
        call) matches, ``mi_grid_axes_invalid`` for a site other than k/v,
        ``mi_head_geometry_unavailable`` for an out-of-range head, and the
        grid driver's own codes (budget, length mismatch, missing payloads).
    """

    _validate_site(site)
    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        started = time.perf_counter()
        clean_log, corrupted_log = _baseline_traces(
            model,
            clean_input,
            corrupted_input,
            trace_kwargs=dict(trace_kwargs or {}),
            guard=guard,
        )
        baseline_seconds = (time.perf_counter() - started) / 2.0
        baselines = _baseline_metrics(clean_log, corrupted_log, metric)
        expansions = _locate_all(clean_log, corrupted_log, site)
        n_q = expansions[0].n_q_heads
        head_list = _head_list(heads, n_q)
        n_cells = _disclose_cost_and_enforce_budget(
            [len(expansions), len(head_list)], baseline_seconds, budget
        )
        values = torch.empty((len(expansions), len(head_list)), dtype=torch.float32)
        campaign: dict[str, int] = {"fires": 0, "identical": 0}
        for row, expansion in enumerate(expansions):
            for col, head in enumerate(head_list):
                values[row, col] = _run_cell(
                    model,
                    corrupted_input,
                    corrupted_log,
                    expansion,
                    (head,),
                    metric,
                    guard,
                    campaign,
                    like=values,
                )
        _warn_if_campaign_all_identical(campaign, f"query-head grid over site {site!r}")
        return PatchGrid(
            axes=("layer", "head"),
            coordinates={
                "layer": tuple(expansion.address for expansion in expansions),
                "head": tuple(head_list),
            },
            values=values,
            receipts={
                "engine": "rerun",
                "fires": campaign["fires"],
                "identical_fires": campaign["identical"],
                "n_cells": n_cells,
                "baseline_seconds_per_rerun": baseline_seconds,
                "metric_provenance": getattr(metric, "__name__", repr(metric)),
                "effect_site": _effect_site_wording(site),
                "site": site,
                "n_q_heads": n_q,
                "n_kv_heads": expansions[0].n_kv_heads,
                "expansion_ops": {e.address: e.op_label for e in expansions},
                **baselines,
            },
        )
    finally:
        _teardown(guard, clean_log, corrupted_log)


def _patch_query_heads_once(  # noqa: PLR0913 -- one cell of the grid, several heads
    model: Any,
    clean_input: Any,
    corrupted_input: Any,
    metric: Any,
    *,
    layer: str,
    heads: tuple[int, ...],
    site: str = "v",
    trace_kwargs: Any = None,
) -> float:
    """Patch several query heads' copies of one layer together; return the metric.

    Private: the consistency check that all query heads of one KV group
    together reproduce the kv-group cell of ``grid(..., sites="v")``.
    """

    _validate_site(site)
    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = _baseline_traces(
            model,
            clean_input,
            corrupted_input,
            trace_kwargs=dict(trace_kwargs or {}),
            guard=guard,
        )
        expansions = {e.address: e for e in _locate_all(clean_log, corrupted_log, site)}
        value = _run_cell(
            model,
            corrupted_input,
            corrupted_log,
            expansions[layer],
            heads,
            metric,
            guard,
            {"fires": 0, "identical": 0},
            like=torch.empty((), dtype=torch.float32),
        )
        return float(value)
    finally:
        _teardown(guard, clean_log, corrupted_log)


def _validate_site(site: str) -> None:
    """Refuse a site other than the key or value operand."""

    if site not in _SITE_WORDS:
        refuse(
            code="mi_grid_axes_invalid",
            message=f"site={site!r}: the query-head interceptor patches the expanded key "
            "or value copy only.",
            remedy='pass site="v" (value copy) or site="k" (key copy)',
            sites=site,
        )


def _head_list(heads: Any, n_q: int) -> list[int]:
    """Return the requested query heads, refusing out-of-range indices."""

    head_list = list(range(n_q)) if heads is None else [int(head) for head in heads]
    outside = [head for head in head_list if not 0 <= head < n_q]
    if outside or not head_list:
        refuse(
            code="mi_head_geometry_unavailable",
            message=f"heads={head_list!r}: query-head indices must lie in [0, {n_q}).",
            remedy=f"pass query-head indices between 0 and {n_q - 1}",
            heads=head_list,
        )
    return head_list


def _locate_all(clean_log: Any, corrupted_log: Any, site: str) -> list[_Expansion]:
    """Locate every grouped attention module's expansion op, or refuse."""

    addresses = _modules_with_facet(clean_log, site)
    if not addresses:
        refuse(
            code="mi_payload_missing",
            message=f"No module exposes facet {site!r} on the clean baseline.",
            remedy="check tl.facets.coverage(trace) for the vocabulary",
            analysis="patching.query_heads_grid",
            missing_sites=[],
        )
    expansions = []
    for address in addresses:
        facets = clean_log.modules[address].facets
        n_q, n_kv = facets.get("n_q_heads", None), facets.get("n_kv_heads", None)
        if not isinstance(n_q, int) or not isinstance(n_kv, int) or n_q <= n_kv:
            refuse(
                code="mi_kv_heads_not_grouped",
                message=f"Module {address!r} has {n_q} query heads and {n_kv} key/value "
                "heads: nothing is shared, so every query head already has its own "
                f"{_SITE_WORDS[site]} slice.",
                remedy="use mechinterp.patch_heads_grid, which measures each head directly",
                module=address,
                num_q_heads=n_q,
                n_kv_heads=n_kv,
            )
        expansions.append(_locate_expansion(clean_log, corrupted_log, address, site, n_q, n_kv))
    return expansions


def _locate_expansion(  # noqa: PLR0913 -- the module's identity plus its head counts
    clean_log: Any, corrupted_log: Any, address: str, site: str, n_q: int, n_kv: int
) -> _Expansion:
    """Return the module's single expansion op, refusing none or several."""

    home_label = str(clean_log.modules[address].facets[site].spec.home_label)
    candidates = _expansion_candidates(clean_log, home_label, address, n_q, n_kv)
    if not candidates:
        refuse(
            code="mi_kv_expansion_unavailable",
            message=f"Module {address!r} exposes no op that expands its grouped "
            f"{_SITE_WORDS[site]} tensor to the {n_q} query heads: the expansion happens "
            "inside a fused kernel (for example torch scaled_dot_product_attention with "
            "enable_gqa=True), so no query head's copy exists to patch.",
            remedy=_UNAVAILABLE_REMEDY,
            module=address,
            site=site,
        )
    if len(candidates) > 1:
        refuse(
            code="mi_kv_expansion_ambiguous",
            message=f"Module {address!r} has {len(candidates)} ops that each expand its "
            f"grouped {_SITE_WORDS[site]} tensor to the query heads "
            f"({[str(op.label) for op in candidates]}); patching one would leave the "
            "others unpatched.",
            remedy="patch the whole group with mechinterp.patch_grid(..., sites=site, "
            "axes=('layer', 'head')), or simplify the module so it expands once",
            module=address,
            candidates=[str(op.label) for op in candidates],
        )
    op = candidates[0]
    twins = [
        row
        for row in corrupted_log.layer_list
        if str(row.label) == str(op.label)
        and str(row.raw_label) == str(op.raw_label)
        and row.site_key == op.site_key
    ]
    if len(twins) != 1:
        refuse(
            code="mi_kv_expansion_ambiguous",
            message=f"The expansion op {op.label!r} of module {address!r} has "
            f"{len(twins)} counterparts in the corrupted trace; a per-query-head cell "
            "needs exactly one.",
            remedy="use equal-length prompts that take the same path through the model",
            module=address,
            candidates=[str(row.label) for row in twins],
        )
    return _Expansion(
        address=address,
        op_label=str(op.label),
        raw_label=str(op.raw_label),
        func_name=str(op.func_name),
        clean_value=op.out.detach().clone(),
        n_q_heads=n_q,
        n_kv_heads=n_kv,
    )


def _expansion_candidates(
    log: Any, home_label: str, address: str, n_q: int, n_kv: int
) -> list[Any]:
    """Walk the facet's descendants inside ``address`` to the first query-head ops.

    A path stops at its first op shaped ``[batch, n_q, ...]``; that op is a
    candidate only when it has one parent, is a reshape-like op, and its
    query-head blocks are exact copies of ``n_kv`` groups (the pure
    expansion). Anything else that reaches the query-head count (a fused
    attention kernel, a scores matmul) mixes heads and is not a copy.
    """

    index = _op_index(log)
    queue = deque(index.get(home_label, []))
    seen: set[str] = set()
    candidates = []
    while queue:
        op = queue.popleft()
        if str(op.label) in seen:
            continue
        seen.add(str(op.label))
        if _reaches_query_heads(op, n_q) and str(op.label) != home_label:
            if _is_pure_expansion(op, n_kv):
                candidates.append(op)
            continue
        queue.extend(_children_inside(op, index, address))
    return candidates


def _op_index(log: Any) -> dict[str, list[Any]]:
    """Map every final and layer label to its op records (several for multi-pass layers)."""

    index: dict[str, list[Any]] = {}
    for op in log.layer_list:
        for key in {str(op.label), str(op.layer_label)}:
            index.setdefault(key, []).append(op)
    return index


def _reaches_query_heads(op: Any, n_q: int) -> bool:
    """Whether ``op``'s output is head-major at the query-head count."""

    shape = tuple(op.shape or ())
    return len(shape) >= 3 and shape[1] == n_q


def _children_inside(op: Any, index: dict[str, list[Any]], address: str) -> list[Any]:
    """Return ``op``'s child op records that ran inside the module ``address``."""

    return [
        child
        for child_label in op.children
        for child in index.get(str(child_label), [])
        if _inside(child, address)
    ]


def _inside(op: Any, address: str) -> bool:
    """Whether ``op`` ran inside the module ``address`` (any pass, any submodule)."""

    module = str(getattr(op, "module", "") or "").split(":", 1)[0]
    return module == address or module.startswith(f"{address}.")


def _is_pure_expansion(op: Any, n_kv: int) -> bool:
    """Whether ``op`` copies ``n_kv`` groups into contiguous query-head blocks."""

    if str(op.func_name) not in _EXPANSION_FUNCS or len(list(op.parents)) != 1:
        return False
    value = getattr(op, "out", None)
    if not isinstance(value, torch.Tensor) or value.shape[1] % n_kv:
        return False
    groups = (value.shape[0], n_kv, value.shape[1] // n_kv, *value.shape[2:])
    blocks = value.detach().reshape(groups)
    return bool(torch.equal(blocks, blocks[:, :, :1].expand_as(blocks)))


def _expansion_selector(expansion: _Expansion) -> Any:
    """Select exactly the expansion op's call in a live rerun.

    Finalized labels and site keys do not exist while a rerun's forward runs,
    so the op is matched by its function, its module and its capture-order
    raw label (equal in every capture of the same path). The predicate reads
    the raw label from either shape of subject: a finished trace's op
    (``raw_label``, when the hook is attached) or the live op record
    (``_label_raw``, suffixed ``_raw``, when the rerun fires it). The
    per-cell fire count check refuses anything but one match.
    """

    raw_label = expansion.raw_label

    def _is_expansion(subject: Any) -> bool:
        """Whether an op (finished trace) or a live op record is the expansion call."""

        for name in ("raw_label", "_label_raw"):
            label = getattr(subject, name, None)
            if isinstance(label, str) and label:
                return label.removesuffix(_RAW_SUFFIX) == raw_label
        return False

    return (
        func_selector(expansion.func_name)
        & in_module_selector(expansion.address)
        & where_selector(_is_expansion, name_hint=f"expansion op {expansion.op_label}")
    )


def _head_hook(clean_value: torch.Tensor, heads: tuple[int, ...]) -> Any:
    """Build the hook that copies the clean ``heads`` slices into the expansion output."""

    def _hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Replace the chosen query heads' copies with their clean values."""

        del ctx
        patched = out.clone(memory_format=torch.preserve_format)
        index = list(heads)
        patched[:, index] = clean_value[:, index].to(dtype=patched.dtype, device=patched.device)
        return patched

    return _hook


def _run_cell(  # noqa: PLR0913 -- the shared per-cell rerun of the grid
    model: Any,
    corrupted_input: Any,
    corrupted_log: Any,
    expansion: _Expansion,
    heads: tuple[int, ...],
    metric: Any,
    guard: Any,
    campaign: dict[str, int],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Run one patched rerun and return its metric; exactly one fire or refuse."""

    ledger: dict[str, int] = {"fires": 0, "identical": 0}
    patched_log = _run_patch(
        model,
        corrupted_input,
        corrupted_log,
        _expansion_selector(expansion),
        _head_hook(expansion.clean_value, heads),
        name=f"query_heads_{expansion.address}_{'_'.join(map(str, heads))}",
        guard=guard,
        where=f"expansion op {expansion.op_label!r} of module {expansion.address!r}",
        campaign_ledger=ledger,
    )
    try:
        if ledger["fires"] != 1:
            refuse(
                code="mi_kv_expansion_ambiguous",
                message=f"The expansion op {expansion.op_label!r} of module "
                f"{expansion.address!r} fired {ledger['fires']} times in one rerun; a "
                "per-query-head cell needs exactly one fire.",
                remedy="patch the whole group with mechinterp.patch_grid(..., sites=site, "
                "axes=('layer', 'head')) for modules called more than once per forward",
                module=expansion.address,
                fires=ledger["fires"],
            )
        campaign["fires"] += ledger["fires"]
        campaign["identical"] += ledger["identical"]
        return _metric_scalar(metric(patched_log), like=like)
    finally:
        patched_log.cleanup()
