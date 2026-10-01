"""THE JAX SITE DIALECT for ``site_key_v1`` minting.

torch/preview ops carry a module ADDRESS stack, but jaxpr structure lives in
the equation ``source_path`` (nested ``pjit``/``scan``/``while`` call and
control sites with iteration markers). The site axis is therefore the
CONTAINING source path with iteration markers stripped
(:func:`~torchlens.backends.jax.jaxpr.normalize_jax_source_path`'s component
rule -- the prior art the L1 design memo names), and the pass-qualified call
instance is the iteration-QUALIFIED containing path, so corresponding
equations in two ``scan``/``while`` iterations share one site with
per-iteration ordinal restarts (property P2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...data_classes._site_key import SiteKeyMinter

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def _jax_site_components(op_log: Any) -> tuple[tuple[str, ...], str]:
    """Return (iteration-stripped site axis, iteration-qualified instance)."""

    from .jaxpr import _normalize_jax_source_path_component

    raw_path = str((getattr(op_log, "annotations", {}) or {}).get("jax_source_path", "") or "")
    containing = tuple(raw_path.split("/")[:-1]) if raw_path else ()
    module_site = tuple(
        normalized
        for component in containing
        if (normalized := _normalize_jax_source_path_component(component, is_leaf=False))
        is not None
    )
    call_instance = "/".join(containing) if containing else "<root>"
    return module_site, call_instance


def jax_site_keys(trace: Trace) -> dict[str, str]:
    """Mint policy-independent ``site_key_v1`` strings for all retained ops.

    Parameters
    ----------
    trace
        Trace containing materialized raw JAX ops in execution order.

    Returns
    -------
    dict[str, str]
        Rendered site key per retained raw label (orphans excluded -- they
        consume no ordinals and carry no keys, the SF-63 ruling).
    """

    minter = SiteKeyMinter()
    keys: dict[str, str] = {}
    for label, op_log in trace._raw_graph_ws.raw_layer_dict.items():
        if getattr(op_log, "is_orphan", False):
            continue
        module_site, call_instance = _jax_site_components(op_log)
        # The shared ``module_attribution``/site invariants (I-S2,
        # ``_check_site_key_uniqueness``) key per-instance uniqueness on
        # ``op.module_call_stack[-1]`` -- a torch/object-module concept a bare
        # jax function_root capture never populates. Site keys are
        # INTENTIONALLY shared across scan/while iterations (the whole point
        # of ``site_key_v1`` is to bridge recurring passes), so without a
        # call-instance marker two different iterations' same-site ops (e.g.
        # two ``scan_read`` reads) collide on ``(site_key, "<root>")`` and
        # trip I-S2. Expose this module's own already-computed
        # iteration-qualified ``call_instance`` through that same field (a
        # single pseudo "address:1" entry, parseable by
        # ``normalize_op_module_calls`` so a later pytree_module finalize
        # pass over this field never raises) whenever it is not the root.
        if call_instance != "<root>":
            op_log.module_call_stack = (f"{call_instance}:1",)
        keys[label] = minter.mint_at(
            module_site,
            call_instance,
            str(getattr(op_log, "type", "") or ""),
            (
                getattr(op_log, "multi_output_index", None)
                if getattr(op_log, "in_multi_output", False)
                else None
            ),
        )
    return keys
