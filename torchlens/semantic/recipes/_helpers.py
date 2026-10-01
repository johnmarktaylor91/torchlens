"""Shared helpers for built-in semantic recipes."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..facets import _MISSING_CONTRIBUTION_KEY, AbsenceReason, Facet, FacetSpec, MissingFacet
from ..reconstruction import ReconstructionFacet, sdpa_reconstruction_spec


def child_module(module: Any, child_name: str) -> Any | None:
    """Return a direct child Module record by address.

    Parameters
    ----------
    module:
        Parent TorchLens Module record.
    child_name:
        Child attribute name.
    """

    trace = getattr(module, "_source_trace", None)
    if trace is None:
        return None
    child_address = child_name if module.address == "self" else f"{module.address}.{child_name}"
    try:
        return trace.modules[child_address]
    except (KeyError, ValueError):
        return None


def child_output_spec(module: Any, child_name: str, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return an op-anchored spec for a child module's single output."""

    child = child_module(module, child_name)
    if child is None:
        return structural(f"child module {child_name!r} is absent")
    try:
        call = child._single_call_or_error()
        if len(call.output_ops) != 1:
            return structural(f"child module {child_name!r} has ambiguous outputs")
        op = child.trace.ops[call.output_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return structural(f"child module {child_name!r} output op is unavailable")
    if not op_output_readable(op):
        return needs_capture(
            f"child module {child_name!r} output op {getattr(op, 'label', '<unknown>')!r} "
            "was not saved",
            f"save=... including {getattr(op, 'label', child_name)!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


def first_input(module: Any) -> Any | AbsenceReason:
    """Return a module's first captured forward input when available."""

    try:
        call = module.calls[0]
        args = call.forward_args
    except (AttributeError, KeyError, IndexError):
        return structural("module has no captured call")
    if args:
        return args[0]
    input_ops = tuple(getattr(call, "input_ops", ()) or ())
    if input_ops:
        return needs_capture(
            "module input value was not captured",
            "save_arg_values=True or save=... including the module input",
        )
    return structural("module call has no positional inputs")


def first_input_spec(module: Any, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return a spec for a module's first captured input.

    Falls back to resolving the input through the op graph -- the same fallback
    ``module_input_op_spec`` already implements -- when ``call.forward_args`` is
    empty/``None``. Finalization ("GC-11",
    ``torchlens/postprocess/finalization.py``) unconditionally nulls
    ``forward_args``/``forward_kwargs`` for every ``module_identity_mode`` except
    ``pytree_module``/``object_module`` -- i.e. for the standard/default
    ``torch_module`` mode used by ordinary ``nn.Module`` tracing -- regardless of
    whether ``save_arg_values=True`` was passed. Without this fallback the
    "input"/"indices" facet was reported as never captured even when the value
    was fully captured, saved, and trivially readable one line away via
    ``call.input_ops[0]``.
    """

    value = first_input(module)
    if not isinstance(value, AbsenceReason):
        return FacetSpec.from_home(value, home_kind="module_input", recipe_id=recipe_id)
    op_spec = module_input_op_spec(module, recipe_id)
    if not isinstance(op_spec, AbsenceReason):
        return op_spec
    return value


def module_input_op_spec(module: Any, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return an op-anchored spec for a module's first input op.

    Parameters
    ----------
    module:
        TorchLens module record.
    recipe_id:
        Recipe identifier.

    Returns
    -------
    FacetSpec | None
        Op-anchored input spec when the module input is a captured op.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return structural("module trace is unavailable")
    try:
        call = module._single_call_or_error()
        input_ops = list(getattr(call, "input_ops", ()) or ())
        if not input_ops:
            return structural("module has no dataflow input op")
        op = trace.ops[input_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return needs_capture(
            "module input op is unavailable", "save=... including the module input"
        )
    if not op_output_readable(op):
        return needs_capture(
            f"module input op {getattr(op, 'label', '<unknown>')!r} was not saved",
            f"save=... including {getattr(op, 'label', 'the module input')!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


def module_output(module: Any) -> Any | None:
    """Return a module's first-call single output when available."""

    try:
        return module.calls[0].out
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None


def module_output_spec(module: Any, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return an op-anchored spec for a module's single or PRIMARY output.

    Multi-output modules (HF attention returns ``(attn_output, attn_weights)``,
    4.x adds present-KV entries) resolve to the PRIMARY output leaf -- container
    position 0 -- through each output op's ``multi_output_name`` role, which was
    minted from the leaf's container path at capture time. ``ModuleCall.output_ops``
    order is NOT container order (finalization rebuilds it in op order), so
    positional pairing with ``output_paths`` would silently pick the wrong leaf
    (F2, walkthrough A-I: eager ``attn_out`` reported structurally absent while
    the value sat captured one lookup away).
    """

    try:
        call = module._single_call_or_error()
        output_ops = list(call.output_ops)
        if len(output_ops) == 1:
            op = module.trace.ops[output_ops[0]]
        else:
            primary = _primary_output_label(module, call)
            if primary is None:
                return structural(
                    f"module has {len(output_ops)} outputs and no uniquely"
                    " resolvable primary output leaf; read the exact leaf via"
                    " ModuleCall.outs[i] / output_ops[i]"
                )
            op = module.trace.ops[primary]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return structural("module output op is unavailable")
    if not op_output_readable(op):
        return needs_capture(
            f"module output op {getattr(op, 'label', '<unknown>')!r} was not saved",
            f"save=... including {getattr(op, 'label', 'the module output')!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


def _primary_output_label(module: Any, call: Any) -> str | None:
    """Return the label of a multi-output module call's primary output leaf.

    The primary leaf is the first entry of the capture-ordered container
    traversal (``call.output_paths[0]`` -- tuple position 0 for tuple returns,
    the first declared field for mapping returns). The match runs through each
    op's ``multi_output_name`` (minted from the same container path at capture
    time), never through positional alignment with ``output_paths``. Role-hinted
    module families (LSTM/GRU/RNN/MultiheadAttention mint hint names instead of
    path strings) deliberately return ``None`` here and keep the ambiguous-output
    refusal.
    """

    paths = tuple(getattr(call, "output_paths", ()) or ())
    if not paths:
        return None
    from ...data_classes._module_role_hints import multi_output_role_from_path

    primary_role = multi_output_role_from_path(paths[0], 0)
    if primary_role is None:
        return None
    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    matches: list[str] = []
    for label in call.output_ops:
        try:
            op = trace.ops[label]
        except (KeyError, TypeError):
            continue
        if getattr(op, "multi_output_name", None) == primary_role:
            matches.append(label)
    if len(matches) == 1:
        return matches[0]
    return None


def parameter_spec(module: Any, name: str, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return a read-only spec for a named module parameter."""

    params = getattr(module, "params", None)
    if params is None:
        return structural(f"parameter {name!r} is absent")
    try:
        for param in params:
            if getattr(param, "name", None) == name:
                return FacetSpec.from_home(param, home_kind="parameter", recipe_id=recipe_id)
    except TypeError:
        return structural(f"parameter {name!r} is absent")
    cls = getattr(module, "cls", None)
    parameter = getattr(cls, name, None)
    if parameter is None:
        return structural(f"parameter {name!r} is absent")
    return FacetSpec.from_home(parameter, home_kind="parameter", recipe_id=recipe_id)


def add_if_present(result: dict[str, Any], name: str, value: Any) -> None:
    """Set a result item or record its absence reason."""

    if isinstance(value, AbsenceReason):
        missing = result.setdefault(_MISSING_CONTRIBUTION_KEY, {})
        if isinstance(missing, dict):
            missing[name] = value
    elif value is not None:
        result[name] = value


def config_value(obj: Any, *names: str) -> Any | AbsenceReason:
    """Return the first available attribute from an object."""

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    custom_attributes = getattr(obj, "custom_attributes", None)
    if isinstance(custom_attributes, dict):
        for name in names:
            if name in custom_attributes:
                return custom_attributes[name]
    return structural(f"config metadata {names!r} is absent")


def config_object(module: Any) -> Any | None:
    """Return the module's captured HF config snapshot when present.

    Modern transformers attention modules no longer mirror head counts as
    module attributes; the numbers live only on ``self.config``, which the
    capture snapshots into ``custom_attributes["config"]``. This is the
    CONFIG-BASED detection source (walkthrough A-I item 3): head geometry and
    ``_attn_implementation`` come from here, never from class names.
    """

    custom_attributes = getattr(module, "custom_attributes", None)
    if isinstance(custom_attributes, dict):
        config = custom_attributes.get("config")
        if config is not None:
            return config
    return getattr(module, "config", None)


def config_object_value(module: Any, *names: str) -> Any | None:
    """Return the first non-``None`` config field from the captured config."""

    config = config_object(module)
    if config is None:
        return None
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def attention_implementation(module: Any) -> str | None:
    """Return the model-declared attention implementation for a module.

    Reads ``_attn_implementation`` / ``attn_implementation`` from the module
    record and its captured config snapshot. This is corroborating evidence
    only: the captured graph (a real SDPA op, a real softmax score path) is
    the detection authority; the config string covers the no-graph-evidence
    case (e.g. an external fused kernel TorchLens cannot see).
    """

    names = ("_attn_implementation", "attn_implementation")
    custom_attributes = getattr(module, "custom_attributes", None)
    if isinstance(custom_attributes, dict):
        for name in names:
            value = custom_attributes.get(name)
            if isinstance(value, str):
                return value
    value = config_object_value(module, *names)
    if isinstance(value, str):
        return value
    return None


def module_op_records(module: Any) -> list[Any]:
    """Return the op records captured inside a module, in label order."""

    trace = getattr(module, "trace", None)
    if trace is None:
        return []
    try:
        labels = list(module._op_labels())
    except (AttributeError, TypeError, ValueError):
        labels = list(getattr(module, "output_ops", ()) or ())
    records: list[Any] = []
    for label in labels:
        try:
            records.append(trace.ops[label])
        except (KeyError, TypeError):
            continue
    return records


def _reshape_spec_heads(
    value: FacetSpec, n_heads: int | None, d_head: int | None
) -> Any | AbsenceReason:
    """Reshape a spec-backed projection output to the per-head layout."""

    if n_heads is None:
        return structural("head-count metadata is absent")
    if d_head is None:
        try:
            last_dim = value.read().shape[-1]
        except (AttributeError, RuntimeError, ValueError):
            return needs_capture(
                "projection output could not be read to infer head dimension",
                "save=... including the projection output",
            )
        d_head = last_dim // n_heads
    return value.heads(n_heads, d_head)


def reshape_heads(
    value: Any, n_heads: int | None, d_head: int | None = None
) -> Any | AbsenceReason:
    """Reshape a projection output to ``(B, S, n_heads, d_head)``."""

    if isinstance(value, AbsenceReason):
        return value
    if isinstance(value, FacetSpec):
        return _reshape_spec_heads(value, n_heads, d_head)
    if not isinstance(value, torch.Tensor) or n_heads is None:
        return structural("head reshape requires tensor value and head-count metadata")
    if value.ndim < 3:
        return structural("head reshape requires rank >= 3")
    inferred_d_head = d_head if d_head is not None else value.shape[-1] // n_heads
    if inferred_d_head <= 0 or value.shape[-1] != n_heads * inferred_d_head:
        return structural("projection output shape is incompatible with head metadata")
    return value.view(*value.shape[:-1], n_heads, inferred_d_head)


def activation_gelu(value: Any) -> Any | AbsenceReason:
    """Return GELU activation when the value is tensor-like."""

    if isinstance(value, AbsenceReason):
        return value
    if isinstance(value, FacetSpec):
        return FacetSpec.computed(lambda: F.gelu(Facet(value).value), recipe_id=value.recipe_id)
    if isinstance(value, torch.Tensor):
        return F.gelu(value)
    return structural("GELU activation operand is not tensor-like")


def activation_silu(value: Any) -> Any | AbsenceReason:
    """Return SiLU activation when the value is tensor-like."""

    if isinstance(value, AbsenceReason):
        return value
    if isinstance(value, FacetSpec):
        return FacetSpec.computed(lambda: F.silu(Facet(value).value), recipe_id=value.recipe_id)
    if isinstance(value, torch.Tensor):
        return F.silu(value)
    return structural("SiLU activation operand is not tensor-like")


def computed_product(left: Any, right: Any, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return a read-only computed product spec for two spec-backed values."""

    if isinstance(left, AbsenceReason):
        return left
    if isinstance(right, AbsenceReason):
        return right
    if isinstance(left, FacetSpec) and isinstance(right, FacetSpec):
        return FacetSpec.computed(
            lambda: Facet(left).value * Facet(right).value,
            recipe_id=recipe_id,
        )
    return structural("computed product operands are not both facet specs")


def fused_sdpa_facet(module: Any, facet: ReconstructionFacet, recipe_id: str) -> Any:
    """Return a reconstructed SDPA facet or an absence reason.

    Parameters
    ----------
    module:
        Attention module record.
    facet:
        Reconstructed facet name.
    recipe_id:
        Recipe identifier for provenance.

    Returns
    -------
    Any
        Computed read-only ``FacetSpec`` or absence reason.
    """

    value = sdpa_reconstruction_spec(module, facet, recipe_id=recipe_id)
    if isinstance(value, MissingFacet):
        return needs_capture(
            value.reason,
            "reconstruction_ready=True or save_arg_values=True including SDPA arguments",
        )
    return value


def fused_sdpa_pattern(module: Any) -> Any:
    """Return a reconstructed fused attention pattern or an absence reason."""

    label = getattr(module.calls[0], "call_label", getattr(module, "address", "<unknown>"))
    value = fused_sdpa_facet(module, "pattern", "attention_reconstruction")
    if not isinstance(value, AbsenceReason):
        return value
    return needs_capture(
        f"attention pattern not captured: model uses a fused attention kernel "
        f"(SDPA/FlashAttention) at {label}. "
        f"{value.detail} To READ pattern/scores/z, re-run with reconstruction_ready=True "
        "(or save_arg_values=True). To EDIT the pattern, capture a model that runs eager / "
        "unfused attention so it is a real, editable op -- e.g. load HF models with "
        "attn_implementation='eager' (AutoModel.from_pretrained(..., attn_implementation='eager')). "
        "The reconstructed pattern is read-only by design: editing it would silently change the "
        "attention kernel, so capture eager for a consistent baseline.",
        "reconstruction_ready=True or save_arg_values=True including SDPA arguments",
    )


def structural(detail: str) -> AbsenceReason:
    """Return a structural-absence reason."""

    return AbsenceReason(status="structurally_absent", detail=detail)


def needs_capture(detail: str, save_hint: str) -> AbsenceReason:
    """Return a needs-capture absence reason."""

    return AbsenceReason(status="needs_capture", detail=detail, save_hint=save_hint)


def op_output_readable(op: Any) -> bool:
    """Return whether an op-backed facet can read its home output now."""

    return bool(
        getattr(op, "has_saved_activation", False) or getattr(op, "out_ref", None) is not None
    )
