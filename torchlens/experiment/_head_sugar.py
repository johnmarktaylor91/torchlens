"""Head-ablation sugar (F03 item 7b): the canonical mechinterp move, guarded.

The sugar lowers "ablate head i of block B" onto the WRITE-CAPABLE ``v``
facet — exact for standard multi-head attention because attn_out is linear
in V_i (proven against a hand-written pre-projection z-slice hook in the
same test file; the ORACLE LAW: any sugar encapsulating architecture
knowledge owes a hand-written-hook oracle beside it, because the oracle is
what caught the panel's own flagship mis-spelling).

Two hard guards:

- MANDATORY MODULE SCOPING: the bare ``tl.head(i)`` spelling silently
  broadcasts across every layer and q/k/v (the measured 36-site defect);
  the sugar cannot emit an unscoped candidate.
- EQUIVALENCE DECLARATION: grouped-query / multi-query attention shares one
  KV head across several query heads, so zeroing its ``v`` is NOT a
  single-head ablation. When the module's facet geometry declares
  ``n_kv_heads != n_q_heads`` — or the geometry is underivable — the sugar
  refuses typed toward the facet-recipe fix rather than mis-lowering.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..errors.episode import BundleExperimentError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

__all__ = ["head_ablation_candidates"]


def _attention_geometry(baseline: Trace, module: str) -> tuple[int | None, int | None]:
    """(n_q_heads, n_kv_heads) from the module's facet view, or (None, None)."""

    from ..semantic.facets import FacetView

    try:
        record = baseline.modules[module]
    except Exception as exc:  # noqa: BLE001 - lookup failure -> typed refusal below
        raise BundleExperimentError(
            f"head_ablation_candidates could not resolve module {module!r} on "
            f"the baseline trace: {type(exc).__name__}: {exc}",
            code="head_ablation_module_unresolved",
            module=module,
        ) from exc
    try:
        view = FacetView(record)
        n_q = view.get("n_q_heads", view.get("n_heads"))
        n_kv = view.get("n_kv_heads", n_q)
    except Exception:  # noqa: BLE001 - geometry underivable reads as unknown
        return None, None
    return (n_q if isinstance(n_q, int) else None, n_kv if isinstance(n_kv, int) else None)


def head_ablation_candidates(
    baseline: Trace,
    module: str,
    *,
    heads: int | Sequence[int],
    facet: str = "v",
) -> dict[str, Any]:
    """Build the scoped per-head candidate plan for one attention block.

    Parameters
    ----------
    baseline:
        The un-edited capture the sweep will fork from.
    module:
        REQUIRED named module address (e.g. ``"transformer.h.5.attn"``) —
        scoping is mandatory; there is no all-layers spelling here.
    heads:
        Head indices: an int ``n`` means ``range(n)``; or an explicit
        sequence of indices.
    facet:
        The write-capable facet lowered to; ``"v"`` (default) is the
        equivalence-proven lowering for standard MHA.

    Returns
    -------
    dict[str, FacetSelector]
        ``{"h<i>": tl.head(i, facet).in_module(module)}`` in head order —
        exactly the candidate plan ``site_sweep`` takes.

    Raises
    ------
    BundleExperimentError
        ``head_ablation_module_unresolved`` when the module is not on the
        trace; ``head_ablation_equivalence_undeclared`` when the module's
        geometry is GQA/MQA (one KV head serves several query heads) or
        cannot be derived — the v-lowering may not claim single-head
        semantics it cannot prove.
    """

    from ..intervention.selectors import head as head_selector

    n_q, n_kv = _attention_geometry(baseline, module)
    if n_q is None or n_kv is None or n_q != n_kv:
        geometry = (
            "underivable" if n_q is None or n_kv is None else f"n_q_heads={n_q}, n_kv_heads={n_kv}"
        )
        raise BundleExperimentError(
            f"head_ablation_candidates cannot declare the v-facet lowering "
            f"equivalent to a single-head ablation at {module!r}: attention "
            f"geometry is {geometry}. For standard MHA (equal query and KV "
            "head counts) attn_out is linear in V_i, so zeroing head i's v IS "
            "the head ablation; under GQA/MQA one KV head serves several "
            "query heads and the claim is false. Remedy: use a facet recipe "
            "that serves per-query-head write access, or ablate the z/result "
            "facet with your own oracle.",
            code="head_ablation_equivalence_undeclared",
            module=module,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
        )
    indices = list(range(heads)) if isinstance(heads, int) else [int(i) for i in heads]
    if not indices:
        raise BundleExperimentError(
            "head_ablation_candidates received an empty head set",
            code="head_ablation_heads_invalid",
        )
    out_of_range = [i for i in indices if i < 0 or i >= n_q]
    if out_of_range:
        raise BundleExperimentError(
            f"head indices {out_of_range} are outside this module's {n_q}-head geometry",
            code="head_ablation_heads_invalid",
            module=module,
            n_q_heads=n_q,
        )
    return {f"h{i}": head_selector(i, facet).in_module(module) for i in indices}
