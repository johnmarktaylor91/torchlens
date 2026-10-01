"""Shared dataflow anchors: the lm-head / final-norm resolution (internal).

One resolution used by residual decomposition, DLA, ``apply_norm_scale``,
prompts, and the alias layer: find the unembedding head by its facet
vocabulary, anchor the final norm by the lm-head recipe's own verified
dataflow walk, and expose the captured pieces (never recompute).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..semantic._norm_reconstruction import NormReconstruction, reconstruct_norm
from ..semantic.facets import AbsenceReason
from ..semantic.recipes._helpers import child_module, config_value
from ..semantic.recipes.lm_head import (
    _final_norm_walk,
    _has_lm_head,
    _head_child_name,
    _norm_input_spec,
)
from ._errors import refuse

__all__ = ["LMHeadAnchor", "resolve_lm_head"]


@dataclass(frozen=True)
class LMHeadAnchor:
    """The resolved unembedding/final-norm anchor for one trace.

    Parameters
    ----------
    owner:
        Module record owning the conventional unembedding child.
    head:
        The unembedding-head module record itself.
    norm:
        The final-norm module record (dataflow-anchored, payload-verified).
    target_op:
        The op producing the tensor the final norm consumes (the residual
        decomposition target).
    """

    owner: Any
    head: Any
    norm: Any
    target_op: Any

    def facets(self) -> Any:
        """Return the owner's lm-head facet view."""

        return self.owner.facets

    def norm_reconstruction(self) -> NormReconstruction:
        """Build the validated final-norm reconstruction (mikit D8).

        Input = the captured tensor entering the norm; output = the norm
        module's captured output; gamma/beta = live parameter reads as the
        recipe evidenced them; eps from module metadata. Validation happens
        inside :func:`reconstruct_norm` before anything is handed out.
        """

        norm_input = getattr(self.target_op, "out", None)
        output_ops = list(getattr(self.norm, "output_ops", ()) or ())
        norm_output = None
        if output_ops:
            try:
                norm_output = self.norm.trace.ops[str(output_ops[0])].out
            except (KeyError, ValueError, RuntimeError):
                norm_output = None
        if not isinstance(norm_input, torch.Tensor) or not isinstance(norm_output, torch.Tensor):
            refuse(
                code="mi_payload_missing",
                message="The final norm's input/output payloads were not saved, so the "
                "frozen-scale linearization has no operating point.",
                remedy="recapture with the residual stream saved (retention_plan or "
                "layers_to_save='all')",
                analysis="norm_reconstruction",
                missing_sites=[
                    {
                        "op_label": str(getattr(self.target_op, "label", "")),
                        "site_key": getattr(self.target_op, "site_key", None),
                    }
                ],
            )
        # Param spellings: HF nn.LayerNorm/RMSNorm use weight/bias; TLens's own
        # LayerNorm/RMSNorm use w/b -- both are real evidence, never guessed.
        params = list(getattr(self.norm, "params", ()) or ())
        gamma_param = next((p for p in params if p.name in ("weight", "w", "gamma")), None)
        beta_param = next((p for p in params if p.name in ("bias", "b", "beta")), None)
        eps = config_value(self.norm, "eps", "variance_epsilon")
        return reconstruct_norm(
            input=norm_input,
            output=norm_output,
            gamma=gamma_param.value if gamma_param is not None else None,
            beta=beta_param.value if beta_param is not None else None,
            eps=None if isinstance(eps, AbsenceReason) else float(eps),
            module_address=str(getattr(self.norm, "address", "")),
            class_name=str(getattr(self.norm, "class_name", "")),
        )


def resolve_lm_head(trace: Any) -> LMHeadAnchor:
    """Resolve the trace's unembedding head + final norm, by dataflow.

    Refuses typed when no head exists or the norm walk cannot verify --
    every consumer shares this one refusal site.
    """

    for module in trace.modules:
        if not _has_lm_head(module):
            continue
        head_name = _head_child_name(module)
        head = child_module(module, head_name) if head_name is not None else None
        if head is None:
            continue
        anchored = _final_norm_walk(head)
        if isinstance(anchored, AbsenceReason):
            refuse(
                code="mi_target_unresolvable",
                message=f"The final norm could not be anchored: {anchored.detail}.",
                remedy="pass an explicit target= op label on the residual stream",
            )
        norm, _walk = anchored
        spec = _norm_input_spec(norm)
        if isinstance(spec, AbsenceReason):
            refuse(
                code="mi_target_unresolvable",
                message=f"The final norm's input could not be resolved: {spec.detail}.",
                remedy="capture with the residual stream saved and retry",
            )
        return LMHeadAnchor(owner=module, head=head, norm=norm, target_op=spec.home)
    refuse(
        code="mi_target_unresolvable",
        message="No module in the trace exposes an unembedding head, so the final "
        "residual state cannot be anchored by dataflow.",
        remedy="pass an explicit target= op label (e.g. the last block output add)",
    )
    raise AssertionError("unreachable")
