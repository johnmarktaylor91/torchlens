"""Lowered counterfactuals (mikit D11) -- the TorchLens-original capability.

Any patch whose effect on ``attn_out`` is LINEAR in the patched quantity
lowers EXACTLY to an add at the real post-projection op:

- head delta: ``(z_clean - z_corrupt)_h @ W_O_h``
- pattern delta: ``((p_clean - p_corrupt)_h @ v_corrupt_group(h)) @ W_O_h``

This is by-head and by-pattern patching on DEFAULT-loaded fused-SDPA models
with no eager reload -- measured 4.578e-05 against a direct eager edit,
BELOW the 6.104e-05 unpatched eager-vs-SDPA baseline. It ships as its own
NAMED intervention kind: the fused ``pattern``/``z``/``result`` facets never
change from read-only, and the receipt says exactly what was lowered where
(never a bare ``replaced=True`` at the virtual site).

The disclosed envelope (contract, enforced):
- eval mode only (in-kernel SDPA dropout is unreproducible) -- refuses;
- q/k are NOT lowerable (softmax intervenes) -- typed refusal teaching the
  eager recapture;
- single-site cells in v1; length-mismatched prompts refuse, never
  broadcast; external ``output_attentions`` tensors are unaffected
  (disclosed in the receipt).
- reading an invalidated virtual facet on the patched result REFUSES typed,
  naming the lowering -- the trace never lies, it refuses.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..intervention.selectors import FuncSelector, InModuleSelector
from ..semantic.patching import (
    _baseline_traces,
    _CounterfactualStateGuard,
    _facet_tensor,
    _run_patch,
    _teardown,
)
from ._errors import refuse
from ._params import head_weight_views

__all__ = ["LoweredResult", "lowered_counterfactual"]

FORMULA_VERSION = "mikit_d11_v1"

#: Virtual facets a lowering invalidates at its layer on the patched rerun.
_INVALIDATED = ("z", "pattern", "scores", "result", "q", "k", "v")


@dataclass(frozen=True)
class LoweredResult:
    """The receipted product of one lowered counterfactual.

    ``read_facet`` is the guarded read surface: invalidated virtual facets
    at the lowered layer refuse typed (the trace never lies); everything
    else reads from the patched rerun.
    """

    metric_value: torch.Tensor | None
    receipt: dict[str, Any]
    _patched_trace: Any = field(repr=False)

    def read_facet(self, module_address: str, name: str) -> Any:
        """Read a facet from the patched rerun, refusing stale virtual reads."""

        lowered_layer = str(self.receipt["requested_virtual_site"]["module_address"])
        if name in _INVALIDATED and (
            module_address == lowered_layer or module_address.startswith(lowered_layer)
        ):
            refuse(
                code="mi_lowered_stale_facet_read",
                message=f"Facet {name!r} at {module_address!r} was INVALIDATED by the "
                f"lowered counterfactual (the edit landed at the real post-projection "
                f"site {self.receipt['lowered_at_real_site']['op_label']!r}; the "
                "virtual attention interior of the patched run was never recomputed).",
                remedy="recapture EAGER and edit z/pattern directly, or read facets at "
                "other layers / downstream sites",
                facet=name,
                module_address=module_address,
            )
        return self._patched_trace.modules[module_address].facets[name]

    def logits(self) -> torch.Tensor:
        """Return the patched run's output logits."""

        from ._anchors import resolve_lm_head

        return resolve_lm_head(self._patched_trace).facets()["logits"].value


def _head_facet(log: Any, address: str, name: str) -> torch.Tensor:
    """Read one attention facet tensor from a baseline trace."""

    return _facet_tensor(log.modules[address].facets[name]).detach()


def _delta_for(  # noqa: PLR0913 -- one cell's full evidence set (kind/head/two traces/site/views)
    kind: str,
    head: int,
    clean_log: Any,
    corrupted_log: Any,
    address: str,
    views: Any,
) -> torch.Tensor:
    """Compute the lowered delta ``[b, pos, d_model]`` for one cell."""

    w_o_h = views.w_o[head].to(torch.float32)  # [d_head, d_model]
    if kind == "head":
        z_clean = _head_facet(clean_log, address, "z")[:, head].to(torch.float32)
        z_corrupt = _head_facet(corrupted_log, address, "z")[:, head].to(torch.float32)
        if z_clean.shape != z_corrupt.shape:
            refuse(
                code="mi_lowered_length_mismatch",
                message=f"Clean/corrupt z shapes differ ({tuple(z_clean.shape)} vs "
                f"{tuple(z_corrupt.shape)}); lowered cells never broadcast.",
                remedy="use equal-length prompts",
            )
        return (z_clean - z_corrupt) @ w_o_h  # [b, pos, d_model]
    pattern_clean = _head_facet(clean_log, address, "pattern")[:, head].to(torch.float32)
    pattern_corrupt = _head_facet(corrupted_log, address, "pattern")[:, head].to(torch.float32)
    if pattern_clean.shape != pattern_corrupt.shape:
        refuse(
            code="mi_lowered_length_mismatch",
            message="Clean/corrupt pattern shapes differ; lowered cells never broadcast.",
            remedy="use equal-length prompts",
        )
    v = _head_facet(corrupted_log, address, "v").to(torch.float32)  # [b, src, H_kv? , d]
    group = max(1, views.n_q_heads // max(1, views.n_kv_heads))
    v_head = v[:, :, head // group if v.shape[2] == views.n_kv_heads else head, :]
    return ((pattern_clean - pattern_corrupt) @ v_head) @ w_o_h


def lowered_counterfactual(  # noqa: PLR0913 -- the memo-normative D11 signature
    model: Any,
    clean_input: Any,
    corrupted_input: Any,
    *,
    layer: Any,
    head: int,
    kind: str = "head",
    metric: Any = None,
    trace_kwargs: Any = None,
) -> LoweredResult:
    """Patch one head (or its pattern) on a fused model via exact lowering.

    Parameters
    ----------
    model:
        The model (eval mode; SDPA or eager -- the lowering is
        implementation-independent, which is what makes it verifiable).
    clean_input / corrupted_input:
        Equal-length prompts; the delta patches CLEAN behavior into the
        CORRUPTED run.
    layer:
        Attention module address (or index into executed attention order).
    head:
        Query head index.
    kind:
        ``"head"`` (z-delta) or ``"pattern"`` (pattern-delta against the
        corrupted values). ``"q"``/``"k"`` refuse: softmax intervenes and no
        exact lowering exists.
    metric:
        Optional ``Trace -> scalar``; evaluated on the patched rerun.
    trace_kwargs:
        Extra kwargs for the baseline traces.
    """

    if kind in ("q", "k"):
        refuse(
            code="mi_lowered_site_not_lowerable",
            message=f"{kind!r} edits cannot lower to the post-projection site: the "
            "softmax between q/k and attn_out makes the effect nonlinear.",
            remedy="recapture with attn_implementation='eager' and edit the real q/k "
            "projection outputs directly (they are real, writable ops)",
            kind=kind,
        )
    if kind not in ("head", "pattern"):
        refuse(
            code="mi_lowered_site_not_lowerable",
            message=f"Unknown lowering kind {kind!r}.",
            remedy='pass kind="head" or kind="pattern"',
        )
    if getattr(model, "training", False):
        refuse(
            code="mi_lowered_dropout_active",
            message="The model is in train mode: in-kernel SDPA dropout is "
            "unreproducible, so the lowering's exactness claim cannot hold.",
            remedy="model.eval() before lowering (eval mode only, by contract)",
        )

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
        from ._heads import _attention_modules, _select_layers

        module = _select_layers(_attention_modules(corrupted_log), layer)[0]
        address = str(module.address)
        views = head_weight_views(corrupted_log, address)
        if not 0 <= head < views.n_q_heads:
            refuse(
                code="mi_head_geometry_unavailable",
                message=f"head {head} outside [0, {views.n_q_heads}).",
                remedy="pass a query-head index",
                head=head,
            )
        delta = _delta_for(kind, head, clean_log, corrupted_log, address, views)

        o_module = corrupted_log.modules[views.sources["o"].module_address]
        o_param = next(p for p in o_module.params if p.name == "weight")
        real_site_label = str(o_param.used_by_ops[0])
        real_op = corrupted_log.ops[real_site_label]

        def _add_delta(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            """Add the lowered delta at the real post-projection op."""

            del hook
            return out + delta.reshape(out.shape).to(out.dtype)

        live_selector = FuncSelector(str(real_op.func_name)) & InModuleSelector(
            str(views.sources["o"].module_address)
        )
        patched_log = _run_patch(
            model,
            corrupted_input,
            corrupted_log,
            live_selector,
            _add_delta,
            name=f"lowered_{kind}_{address}_h{head}",
            guard=guard,
            facet_name=kind,
            address=address,
        )
        receipt = {
            "capability": "lowered_counterfactual",
            "formula_version": FORMULA_VERSION,
            "kind": kind,
            "requested_virtual_site": {
                "module_address": address,
                "facet": "z" if kind == "head" else "pattern",
                "head": head,
            },
            "lowered_at_real_site": {
                "op_label": str(real_op.label),
                "site_key": getattr(real_op, "site_key", None),
                "module_address": views.sources["o"].module_address,
            },
            "source_sites": {
                "clean": f"{address}/{'z' if kind == 'head' else 'pattern'}",
                "corrupt": f"{address}/{'z' if kind == 'head' else 'pattern+v'}",
            },
            "invalidated_virtual_facets": [f"{address}/{name}" for name in _INVALIDATED],
            "disclosures": [
                "external output_attentions tensors are unaffected by the lowering",
                "single-site cell (v1); eval mode only",
            ],
        }
        metric_value = None
        if metric is not None:
            metric_value = metric(patched_log)
        return LoweredResult(metric_value=metric_value, receipt=receipt, _patched_trace=patched_log)
    finally:
        _teardown(guard, clean_log, corrupted_log)
