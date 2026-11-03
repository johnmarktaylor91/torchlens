"""Provenance disclosure and source variable names for the postprocess pipeline.

The lookup-key step ends with :func:`_warn_unattributed_tensor_args`, the one
aggregate warning for tensor arguments the capture could not attribute to a
graph or source; the following step is :func:`_populate_var_names`, which
resolves assignment names from the AST index when code context was kept.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member
from .ast_branches import resolve_var_names

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _unattributed_op_offenders(ops: Any) -> list[str]:
    """Return ``"label (positions)"`` rows for non-output ops with source-less args."""

    offenders: list[str] = []
    for op in ops:
        if getattr(op, "type", None) == "output":
            continue
        positions = tuple(record_member(op, "unattributed_tensor_args", ()) or ())
        if not positions:
            continue
        label = getattr(op, "label", None) or getattr(op, "layer_label", None) or op._label_raw
        offenders.append(f"{label} ({', '.join(positions)})")
    return offenders


def _pop_boundary_adoptions(self: Trace, field_name: str, suffix: str) -> list[str]:
    """Pop one module-boundary adoption queue into human-readable offender rows."""

    return [
        f"{label} (adopted at module {boundary} {module_address}{suffix})"
        for label, boundary, module_address in (self.__dict__.pop(field_name, None) or ())
    ]


def _warn_unattributed_tensor_args(self: Trace) -> None:
    """Warn once for tensor arguments without graph/source provenance.

    Parameters
    ----------
    self:
        Trace being postprocessed.

    Returns
    -------
    None
        Emits at most one aggregate warning.
    """

    from .._capture_honesty import (
        ADVISORY_MODULE_BOUNDARY_ADOPTION,
        ADVISORY_ORPHAN_UNATTRIBUTED_ARGS,
        ADVISORY_UNRECORDED_OPERATOR_MUTATION,
        append_capture_advisory,
    )

    offenders = _unattributed_op_offenders(self._layer_list or ())
    # Orphan pruning (step 3) runs BEFORE this step and strips the pruned ops, so a
    # source-less tensor whose only consumer was pruned (``G.sum().item()``, a
    # control-flow predicate on a global) would lose its witness with the op;
    # ``_remove_orphan_nodes`` reads the witness before removal.
    orphans = list(self.__dict__.pop("_orphan_unattributed_tensor_args", None) or ())
    self.__dict__.pop("_plain_scalar_escape_labels", None)
    # A direct data read (``G.tolist()``, ``float(G)``) of a source-less tensor has
    # no op at all to carry the witness; the plain escape belt queued it.
    orphans += list(self.__dict__.pop("_plain_direct_escape_gaps", None) or ())
    # R16: module-entry adoptions of untagged tensors (outside disclosed
    # transform/dynamo regions) are the module-consumed twin of the
    # unattributed-args case; without this fold, a stale-ref escape whose
    # output was first consumed by a MODULE was laundered into a clean
    # ``internalsource`` node -- no warning, no rescue,
    # consumption-order-dependent disclosure. Module-EXIT adoptions are the
    # module-returned twin (a stale ref whose output a module returns, e.g.
    # transformers' ``GELUActivation``): the boundary op tagged it before any
    # consumer could flag it.
    adoptions = _pop_boundary_adoptions(self, "_module_boundary_adoptions", "")
    # Closure/forward-global tensors a module consumes or returns directly: no
    # escape (they predate the forward), but no source either.
    outside = _pop_boundary_adoptions(
        self, "_module_boundary_outside_sources", ", closure or forward-global tensor"
    )
    # Direct ``torch.ops`` calls that wrote an argument the recorder could not record
    # in place (``_torch_ops_calls._disclose_unrecorded_mutation``).
    mutations = list(self.__dict__.pop("_unrecorded_operator_mutations", None) or ())
    if not (offenders or orphans or adoptions or outside or mutations):
        return
    # Adopted tensors leave a functionless internal-source node that replays and
    # validates, and pruned ops leave nothing at all; persist both so forward
    # validation fails them (retained per-op positions already persist as
    # ``Op.unattributed_tensor_args``). Rows are written before the warning,
    # which a warning filter may raise.
    append_capture_advisory(self, ADVISORY_MODULE_BOUNDARY_ADOPTION, adoptions + outside)
    append_capture_advisory(self, ADVISORY_ORPHAN_UNATTRIBUTED_ARGS, orphans)
    append_capture_advisory(self, ADVISORY_UNRECORDED_OPERATOR_MUTATION, mutations)
    if offenders or orphans or adoptions:
        # Session-time escape signal: the capture entry reads this flag to decide
        # whether a rescue re-run (TorchFunctionMode net) should be attempted.
        self._had_unattributed_tensor_args = True
    warnings.warn(
        "TorchLens found tensor arguments with no graph/source provenance. "
        "These are usually tensors captured from outside the traced model; "
        "module tensor attributes, inputs, parameters, and buffers are known sources. "
        "Offending ops/arg positions: "
        + "; ".join(offenders + orphans + adoptions + outside + mutations),
        UserWarning,
        stacklevel=2,
    )


def _populate_var_names(self: Trace) -> None:
    """Populate source assignment names for captured operation call sites.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    """

    if not self._options_save_source_context:
        return
    for op in self._layer_list:
        if getattr(op, "type", None) == "output":
            op._source_var_names = []
            continue
        op._source_var_names = resolve_var_names(
            record_member(op, "code_context", []) or [],
            getattr(op, "func_name", None),
        )
