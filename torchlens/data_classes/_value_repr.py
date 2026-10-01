"""Value-record line/card renderers (F10; lovely memo item 6).

One voice for the value-bearing records: ``repr(x)`` is ONE envelope+core
line and ``str(x)`` is a bounded card whose first line IS the repr (D15).
The core bytes come from the C02 sound kernel through
:func:`torchlens.stats.render_core_line`; the envelope affixes provenance
(memo 4.2); honesty tokens are the shared vocabulary of memo 4.4 and are
never laundered. Multi-pass layers show per-pass cores or a count, never
a pooled statistic (D32).

Layering: this module is L1 (data_classes); every stats import is
deferred to call time (the spine layer lint forbids eager upward
imports). Renderers never materialize lazy payloads, never read disk,
never mutate, and never raise (voice rule 10) -- a failure degrades to
the record's legacy placeholder form.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from typing import Any

from ..utils.fail_open import fail_open

#: Sentinel for a payload read that raised (distinct from a freed ``None``).
_READ_FAILED = object()

#: Honesty tokens (memo 4.4): shared vocabulary, identical on every surface.
NOT_SAVED = "(not saved)"
FREED = "(freed)"
METADATA_ONLY = "(metadata only)"
LAZY_DISK = "(disk-backed: metadata only until read)"
PREVIEW_BACKEND = "(preview backend: shape/dtype only)"
EDITED = "(edited)"


def _payload_state(record: Any) -> str:
    """Return ``present`` / ``lazy`` / ``unsaved`` without materializing.

    Delegates to the agent-surface classifier (one probe, shared
    semantics): reading through the private slot accessor so the probe
    never triggers a materializing ``op.out`` read.
    """

    from ..report._agent_json import _payload_state as probe

    return probe(record)


def _metadata_core(record: Any) -> str:
    """Bare shape/dtype token for payload-absent records (compact dtypes)."""

    shape = getattr(record, "shape", None)
    dtype = getattr(record, "dtype", None)
    shape_token = "[" + ",".join(str(d) for d in shape) + "]" if shape is not None else "[?]"
    if dtype is not None:
        from ..stats._tensor_stats import _dtype_token

        dtype_token = fail_open(
            lambda: _dtype_token(dtype),
            lambda _error: str(dtype).replace("torch.", ""),
        )
    else:
        dtype_token = "?"
    return f"{dtype_token}{shape_token}"


def payload_core(record: Any, *, identity: str | None = None, max_width: int | None = None) -> str:
    """Render one record payload's core line with honesty tokens.

    The clean path renders the C02 core grammar over the resident tensor;
    every degraded path renders metadata plus the exact honesty token --
    ``(not saved)`` is distinguishable from saved-and-boring, a lazy
    disk-backed payload is never materialized by a repr, and a non-torch
    preview payload discloses itself instead of duck-typing torch calls.

    Parameters
    ----------
    record:
        Op-shaped record (needs ``out``/``shape``/``dtype`` reads).
    identity:
        Stable identity seeding the gathered sampler (site key / label).
    max_width:
        Optional width bound for the D14 degradation ladder.

    Returns
    -------
    str
        The core line or a metadata+token stand-in. Never raises.
    """

    import torch

    state = _payload_state(record)
    if state == "lazy":
        return f"{_metadata_core(record)} {LAZY_DISK}"
    if state == "unsaved":
        saved_note = (
            " (saved at capture; no bytes here)"
            if getattr(record, "has_saved_activation", False)
            else ""
        )
        return f"{_metadata_core(record)} {NOT_SAVED}{saved_note}"
    payload = fail_open(lambda: record.out, lambda _error: _READ_FAILED)
    if payload is _READ_FAILED:
        return f"{_metadata_core(record)} {NOT_SAVED}"
    if payload is None:
        return f"{_metadata_core(record)} {FREED}"
    if not isinstance(payload, torch.Tensor):
        return f"{_metadata_core(record)} {PREVIEW_BACKEND}"
    from ..stats import render_core_line, tensor_stats

    return render_core_line(tensor_stats(payload, identity=identity), max_width=max_width)


def _edited_token(op: Any) -> str | None:
    """Return the ``(edited)`` mark when intervention evidence exists.

    Post-edit values must never be presented as observed (composition
    row: stats line x intervention). Evidence: the node-level replacement
    flag or a tier-(ii) edge substitution on this record.
    """

    if getattr(op, "intervention_replaced", False):
        return EDITED
    if getattr(op, "edge_substitutions", None):
        return EDITED
    return None


def _trace_token(record: Any) -> str | None:
    """Short owning-trace token (model class name), or None when detached."""

    trace = fail_open(lambda: record.source_trace, lambda _error: None)
    return getattr(trace, "model_class_name", None)


def _op_position(op: Any) -> str | None:
    """Graph position token ``step/total`` for one op, or None."""

    total = fail_open(lambda: op.source_trace.num_ops, lambda _error: None)
    step = getattr(op, "step_index", None)
    if step is None or total is None:
        return None
    return f"{step}/{total}"


def _pass_token(op: Any) -> str:
    """Pass tag ``k/n`` (``(pass 1/1)`` on single-pass ops stays explicit)."""

    return f"{getattr(op, 'pass_index', 1)}/{getattr(op, 'num_passes', 1)}"


def op_repr_line(op: Any) -> str:
    """One envelope+core line for an Op (the lovely demo moment)."""

    from ..stats._envelope import envelope_line

    core = payload_core(op, identity=getattr(op, "site_key", None) or op.layer_label)
    edited = _edited_token(op)
    if edited:
        core = f"{core} {edited}"
    return envelope_line(
        core,
        trace_token=_trace_token(op),
        address=op.layer_label,
        pass_token=_pass_token(op),
        bracket=(
            "op",
            _op_position(op),
            None if getattr(op, "is_input", False) else getattr(op, "func_name", None),
        ),
    )


def _where_line(op: Any) -> str | None:
    """The card's module-location line, when the op sits in a module."""

    module = getattr(op, "module", None)
    if not module:
        return None
    return f"where  module {module}"


def _graph_line(op: Any) -> str | None:
    """Bounded neighbor line: 2 parents, 2 children, exact downstream count."""

    parents = list(getattr(op, "parents", ()) or ())
    children = list(getattr(op, "children", ()) or ())
    downstream = getattr(op, "output_descendants", None)
    parts = []
    if parents:
        shown = ", ".join(parents[:2]) + (f" +{len(parents) - 2}" if len(parents) > 2 else "")
        parts.append(f"<- {shown}")
    if children:
        shown = ", ".join(children[:2]) + (f" +{len(children) - 2}" if len(children) > 2 else "")
        parts.append(f"-> {shown}")
    if downstream is not None:
        parts.append(f"({len(downstream)} downstream)")
    if not parts:
        return None
    return "graph  " + "   ".join(parts)


def _func_line(op: Any) -> str | None:
    """Function + grad_fn line; config stays behind the More: exit."""

    if getattr(op, "is_input", False):
        return None
    func_name = getattr(op, "func_name", None)
    if not func_name:
        return None
    grad_fn = getattr(op, "grad_fn_class_name", None)
    tail = f"   grad_fn {grad_fn}" if grad_fn and grad_fn != "none" else ""
    return f"func   {func_name}(...){tail}"


def _params_line(op: Any) -> str | None:
    """Parameter geometry line, only when params were consumed."""

    shapes = getattr(op, "param_shapes", None) or []
    if not shapes:
        return None
    from ._repr import format_shape_list

    return (
        f"params {format_shape_list(shapes)} = "
        f"{getattr(op, 'num_params', '?')} ({getattr(op, 'param_memory', '?')})"
    )


def _cost_line(op: Any) -> str | None:
    """Timing line through the typed quantity formatter (never a bare unit)."""

    if getattr(op, "is_input", False):
        return None
    duration = getattr(op, "func_duration", None)
    if duration is None:
        return None
    return f"cost   {duration}"


def op_card(op: Any) -> str:
    """Bounded Op card: preview/relations/keys live behind More: exits."""

    from ..stats._envelope import record_card

    body = [
        line
        for line in (
            _where_line(op),
            _graph_line(op),
            _func_line(op),
            _params_line(op),
            _cost_line(op),
        )
        if line
    ]
    return record_card(
        op_repr_line(op),
        body,
        more=(".out", ".parents", ".children", ".func_config", ".lookup_keys"),
    )


def layer_repr_line(layer: Any) -> str:
    """One envelope+core line for a Layer; multi-pass shows a count (D32)."""

    from ..stats._envelope import envelope_line

    num_passes = getattr(layer, "num_passes", 1)
    ops = list(getattr(layer, "ops", ()) or ())
    first = ops[0] if ops else None
    if num_passes > 1:
        core = f"(x{num_passes} passes) (per-pass cores in str())"
    elif first is not None:
        core = payload_core(first, identity=getattr(first, "site_key", None) or layer.layer_label)
        edited = _edited_token(first)
        if edited:
            core = f"{core} {edited}"
    else:
        core = METADATA_ONLY
    position = _op_position(first) if first is not None else None
    return envelope_line(
        core,
        trace_token=_trace_token(layer),
        address=layer.layer_label,
        pass_token=None if num_passes > 1 else "1/1",
        bracket=(
            "layer",
            position,
            None if getattr(layer, "is_input", False) else getattr(layer, "func_name", None),
        ),
    )


def layer_card(layer: Any) -> str:
    """Bounded Layer card; per-pass cores for multi-pass layers, never pooled."""

    from ..stats._envelope import record_card

    ops = list(getattr(layer, "ops", ()) or ())
    first = ops[0] if ops else None
    body: list[str] = []
    num_passes = getattr(layer, "num_passes", 1)
    if num_passes > 1:
        for op in ops[:3]:
            core = payload_core(op, identity=getattr(op, "site_key", None) or op.label)
            body.append(f"pass {_pass_token(op)}: {core}")
        if len(ops) > 3:
            body.append(f"... {len(ops) - 3} more passes (.ops)")
    if first is not None:
        body.extend(
            line
            for line in (
                _where_line(first),
                _graph_line(first),
                _func_line(first),
                _params_line(first),
                _cost_line(first) if num_passes == 1 else None,
            )
            if line
        )
    return record_card(
        layer_repr_line(layer),
        body,
        more=(".ops", ".parents", ".children", ".to_pandas()"),
    )


def param_repr_line(param: Any) -> str:
    """One line for a Param: live/versioned core over the live tensor (D30)."""

    import torch

    from ..stats import render_core_line, tensor_stats
    from ..stats._envelope import envelope_line

    live = fail_open(lambda: param._peek_live_param(cache=False), lambda _error: None)
    if isinstance(live, torch.Tensor):
        stats = tensor_stats(live, identity=getattr(param, "address", None))
        version = stats.tensor_version
        core = render_core_line(stats)
        if version is not None:
            core = f"{core} live v{version}"
    else:
        core = (
            f"{_metadata_core(param)} (live tensor unavailable: source model gone or ref released)"
        )
    status = "trainable" if getattr(param, "is_trainable", False) else "frozen"
    return envelope_line(
        f"{core} {status}",
        address=getattr(param, "address", None),
        bracket=("param",),
    )


def param_card(param: Any) -> str:
    """Bounded Param card with tie disclosure (real gpt2: wte.weight x2)."""

    from ..stats._envelope import record_card

    body: list[str] = []
    module_address = getattr(param, "module_address", None)
    if module_address:
        body.append(f"module {module_address}")
    used_by = list(getattr(param, "used_by_layers", ()) or ())
    if used_by:
        tie = "  (tied: feeds multiple sites)" if len(used_by) > 1 else ""
        shown = ", ".join(used_by[:3]) + (f" +{len(used_by) - 3}" if len(used_by) > 3 else "")
        body.append(f"used by {shown}{tie}")
    linked = list(getattr(param, "co_parent_params", ()) or ())
    if linked:
        body.append(f"linked {', '.join(linked[:3])}")
    body.append(f"has_grad {getattr(param, 'has_grad', None)}")
    if getattr(param, "has_optimizer", None) is not None:
        body.append(f"optimizer {param.has_optimizer}")
    return record_card(
        param_repr_line(param),
        body,
        more=(
            ".grad",
            ".to_pandas()",
        ),
    )


def buffer_repr_line(buffer: Any) -> str:
    """One line for a Buffer: snapshot core + explicit value basis."""

    import torch

    from ..stats._envelope import envelope_line

    initial = getattr(buffer, "initial_value", None)
    if isinstance(initial, torch.Tensor):
        from ..stats import render_core_line, tensor_stats

        core = render_core_line(tensor_stats(initial, identity=getattr(buffer, "address", None)))
        basis = "snapshot at capture"
    else:
        core = _metadata_core(buffer)
        basis = "metadata only"
    versions = getattr(buffer, "versions", None) or ()
    tail = f" ({len(versions)} versions)" if len(versions) > 1 else ""
    return envelope_line(f"{core} [{basis}]{tail}", address=buffer.address, bracket=("buffer",))


def buffer_card(buffer: Any) -> str:
    """Bounded Buffer card: module path + version/overwrite facts."""

    from ..stats._envelope import record_card

    body: list[str] = []
    module_address = getattr(buffer, "module_address", None)
    if module_address:
        body.append(f"module {module_address}")
    overwrites = getattr(buffer, "num_overwrites", None)
    if overwrites:
        body.append(f"overwrites {overwrites}")
    return record_card(buffer_repr_line(buffer), body, more=(".versions", ".to_pandas()"))


def module_call_repr_line(call: Any) -> str:
    """One line for a ModuleCall: address:pass, class, op span."""

    from ..stats._envelope import envelope_line

    module_class = getattr(call, "class_name", None)
    if not module_class:
        module_class = fail_open(lambda: call.module.class_name, lambda _error: "?")
    label = getattr(call, "call_label", "?")
    core = f"{module_class} ops={getattr(call, 'num_ops', '?')}"
    return envelope_line(core, address=label, bracket=("module_call",))


def module_call_card(call: Any) -> str:
    """Bounded ModuleCall card: <=3 output cores then the exact count."""

    from ..stats._envelope import record_card

    body: list[str] = []
    output_labels = list(getattr(call, "output_layers", ()) or ())
    if len(output_labels) > 3:
        body.append(f"{len(output_labels)} outputs (first 3 shown)")
    for label in output_labels[:3]:
        record = _resolve_trace_record(call, label)
        if record is None:
            body.append(f"out {label}")
        else:
            body.append(f"out {label} {payload_core(record, identity=label)}")
    children = list(getattr(call, "call_children", ()) or ())
    if children:
        shown = ", ".join(children[:3]) + (f" +{len(children) - 3}" if len(children) > 3 else "")
        body.append(f"children {shown}")
    return record_card(module_call_repr_line(call), body, more=(".outs", ".to_pandas()"))


def _resolve_trace_record(record: Any, label: str) -> Any | None:
    """Resolve one layer label through the owning trace, or None.

    Module/ModuleCall records reach their trace through ``_source_trace``;
    Op/Layer through ``source_trace``. Every failure degrades to ``None``
    (a card must render on detached records, voice rule 10).
    """

    trace = getattr(record, "_source_trace", None)
    if trace is None:
        trace = fail_open(lambda: record.source_trace, lambda _error: None)
    if trace is None:
        return None
    return fail_open(lambda: trace[label], lambda _error: None)


def module_repr_line(module: Any) -> str:
    """One line for a Module: address/class/calls/ops/params/mode."""

    from ..stats._envelope import envelope_line

    mode = "train" if getattr(module, "training", False) else "eval"
    tokens = [
        f"{getattr(module, 'class_name', '?')}",
        f"calls={getattr(module, 'num_calls', '?')}",
        f"ops={getattr(module, 'num_layers', '?')}",
        f"params={getattr(module, 'num_params', '?')}",
        mode,
    ]
    if getattr(module, "has_multiple_addresses", False):
        tokens.append("(aliased)")
    return envelope_line(
        " ".join(tokens), address=getattr(module, "address", None), bracket=("module",)
    )


def module_card(module: Any) -> str:
    """Bounded Module card: children + tie/alias disclosure.

    The output core prints ONLY when exactly one saved output call is in
    scope (matrix rule: explicit plurality otherwise).
    """

    from ..stats._envelope import record_card

    body: list[str] = []
    if getattr(module, "has_multiple_addresses", False):
        aliases = list(getattr(module, "all_addresses", ()) or ())
        body.append(f"aliases {', '.join(aliases[:3])}")
    children = list(getattr(module, "address_children", ()) or ())
    if children:
        shown = ", ".join(children[:3]) + (f" +{len(children) - 3}" if len(children) > 3 else "")
        body.append(f"children {shown}")
    output_labels = list(getattr(module, "output_layers", ()) or ())
    if len(output_labels) == 1:
        record = _resolve_trace_record(module, output_labels[0])
        if record is not None:
            body.append(f"out {output_labels[0]} {payload_core(record, identity=output_labels[0])}")
    elif output_labels:
        body.append(f"{len(output_labels)} output sites (explicit plurality; see .output_layers)")
    return record_card(module_repr_line(module), body, more=(".calls", ".layers", ".to_pandas()"))
