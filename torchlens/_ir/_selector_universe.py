"""Per-lifecycle subject universes for the selector interpreter.

:mod:`torchlens._ir.selector_eval` compiles every selector into a plan whose
nodes read the subject through the adapters here: which label spellings a
lifecycle exposes, which module candidates a subject carries, how output and
input container paths compare, and how a backward ``GradFn`` site's type and
calls read. Nothing here branches on selector shape; the helpers
describe subjects only, so the evaluator stays a dispatch table.

Lifecycle keys are the ones :mod:`torchlens._ir.selector_eval` documents:
``"capture"`` (in-flight record contexts), ``"site"`` (finalized ``Op`` /
``Layer`` / ``GradFn`` records) and ``"live"`` (hook-site proxies). This module
imports only leaf modules from the package (the live-trace guard, the backend
name), plus a deferred read of the ``GradFn`` pass pairing, which sits outside
every import cycle: adapters that need the evaluator's own types (container
paths, record contexts) live beside the evaluator instead, so the selector
split adds no module to the import cycle.
"""

from __future__ import annotations

import re
from typing import Any

from .._constants import pass_token_labels
from .._data_classes._trace_missing_attrs import is_live_trace
from ..backends._backend_names import TORCH_BACKEND_NAME

#: String-label attribute universes per lifecycle, in match order.
_STRING_LABEL_ATTRS: dict[str, tuple[str, ...]] = {
    "capture": ("label", "raw_label", "label_raw", "layer_label", "short_layer_label"),
    "site": ("layer_label", "label", "short_layer_label", "short_label", "_layer_label_raw"),
    "live": ("_layer_label_raw", "_label_raw", "layer_label"),
}

#: Substring/regex universes. Post-hoc search stays on the final label only:
#: matching raw/short spellings would make ``contains("raw")`` match every op.
_SUBSTRING_LABEL_ATTRS: dict[str, tuple[str, ...]] = {
    **_STRING_LABEL_ATTRS,
    "site": ("layer_label",),
}

_FINAL_LABEL_PATTERN = re.compile(r"(?:_\d+_\d+(?::(?:\d+|last|all))?$|:(?:\d+|last|all)$)")


def sanitize_transform_kind(kind: object) -> str:
    """Return the transform-kind spelling used for labels.

    Parameters
    ----------
    kind:
        Transform kind or selector value.

    Returns
    -------
    str
        Lowercase spelling with underscores and dots removed.
    """

    return str(kind).lower().replace("_", "").replace(".", "")


def module_address_matches(module_pass: Any, address: str) -> bool:
    """Return whether a module candidate belongs to an address.

    Parameters
    ----------
    module_pass:
        Pass-qualified label, ``(address, call_index)`` tuple, or tuple repr.
    address:
        Module address with or without pass qualification. A bare address
        matches every call of the module; a pass label (``"block:2"``) matches
        only that call, whichever spelling the candidate uses.

    Returns
    -------
    bool
        Whether the module candidate belongs to the requested address.
    """

    if isinstance(module_pass, tuple) and module_pass and isinstance(module_pass[0], str):
        return _module_pass_tuple_matches(module_pass, address)
    module_label = str(module_pass)
    if module_label.startswith("("):
        return _module_pass_repr_matches(module_label, address)
    module_address = module_label.rsplit(":", 1)[0]
    return module_label == address or module_address == address


def _module_pass_tuple_matches(module_pass: tuple[Any, ...], address: str) -> bool:
    """Return whether an ``(address, call_index)`` tuple candidate matches ``address``.

    Live boundary sites carry these tuples; a pass label must compare against
    the same ``"address:call"`` spelling the string candidates (and bind's
    boundary hook) use.
    """

    if module_pass[0] == address:
        return True
    return len(module_pass) > 1 and f"{module_pass[0]}:{module_pass[1]}" == address


def _module_pass_repr_matches(module_label: str, address: str) -> bool:
    """Return whether a tuple-repr candidate (``"('block', 2)"``) matches ``address``."""

    base, sep, call = address.rpartition(":")
    if sep and call.isdigit():
        return f"'{base}', {call})" in module_label or f'"{base}", {call})' in module_label
    return f"'{address}'" in module_label or f'"{address}"' in module_label


def looks_like_finalized_label(label_value: str) -> bool:
    """Return whether a label literal looks postprocessed.

    Parameters
    ----------
    label_value:
        Label literal from ``tl.label`` / ``tl.label_contains`` / ``tl.label_matches``.

    Returns
    -------
    bool
        True for pass suffixes like ``:2`` or ``relu_4_27``-style names.
    """

    return bool(_FINAL_LABEL_PATTERN.search(label_value)) and not label_value.endswith("_raw")


def live_label_error_message(label_value: str, selector_kind: str = "label") -> str:
    """Build a copy-pasteable finalized-label diagnostic.

    Parameters
    ----------
    label_value:
        Finalized-looking label.
    selector_kind:
        Label-oriented selector kind for the message.

    Returns
    -------
    str
        User-facing error message.
    """

    return (
        f"tl.{selector_kind}({label_value!r}) looks like a finalized postprocess label, but "
        "live selectors run during capture before those labels exist. For post-capture "
        f'selection on a saved trace use tl.where(lambda p: p.layer_label == "{label_value}") '
        "with the push engine; a rerun or a live capture never matches it. For live hooks "
        'and reruns, use a capture-time selector such as tl.func("relu") or '
        'tl.module("encoder.layer.4").'
    )


def _string_labels(
    subject: Any,
    lifecycle: str,
    attrs: dict[str, tuple[str, ...]] = _STRING_LABEL_ATTRS,
) -> tuple[str, ...]:
    """Return the label-string universe for one subject.

    Parameters
    ----------
    subject:
        Lifecycle subject (context, site, or live proxy).
    lifecycle:
        Active lifecycle key.
    attrs:
        Attribute-universe table; substring/regex matching passes the
        narrower :data:`_SUBSTRING_LABEL_ATTRS`.

    Returns
    -------
    tuple[str, ...]
        Non-``None`` label spellings, deduplicated in match order.
    """

    labels: list[str] = []
    seen: set[str] = set()
    for attr in attrs[lifecycle]:
        value = getattr(subject, attr, None)
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            labels.append(text)
    if lifecycle == "site" and attrs is _STRING_LABEL_ATTRS:
        # Pass tokens are post-hoc spellings: the pass count is known only
        # once the trace has settled, so the live and capture lanes never
        # answer to them.
        labels.extend(token for token in pass_token_labels(subject) if token not in seen)
    return tuple(labels)


def _module_containment_candidates(subject: Any, lifecycle: str) -> tuple[Any, ...]:
    """Return module-containment candidates for ``tl.in_module``.

    Parameters
    ----------
    subject:
        Lifecycle subject.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    tuple[Any, ...]
        Containment candidates: addresses, pass-qualified labels, or tuples.
    """

    if lifecycle == "capture":
        return _capture_module_candidates(subject)
    return tuple(getattr(subject, "output_of_module_calls", ()) or ()) + tuple(
        getattr(subject, "modules", ()) or ()
    )


def _capture_module_candidates(ctx: Any) -> tuple[Any, ...]:
    """Return module-address candidates visible on a capture context.

    Parameters
    ----------
    ctx:
        Capture-time context or layer-like object.

    Returns
    -------
    tuple[Any, ...]
        Module addresses and pass-qualified module labels.
    """

    candidates: list[str] = []
    address = getattr(ctx, "address", None)
    if address is not None:
        candidates.append(str(address))
    source_trace: Any = getattr(ctx, "source_trace", None)
    if (
        source_trace._options_module_identity_mode if is_live_trace(source_trace) else None
    ) == "function_root":
        candidates.append("self")
        candidates.append("self:1")
    for frame in getattr(ctx, "module_stack", ()):
        candidates.extend(_module_frame_candidates(frame))
    modules = getattr(ctx, "modules", ())
    module_ops = getattr(ctx, "output_of_module_calls", ())
    candidates.extend(
        _module_candidate_string(candidate) for candidate in tuple(modules) + tuple(module_ops)
    )
    return tuple(candidates)


def _module_frame_candidates(frame: Any) -> tuple[str, ...]:
    """Return the address and pass-qualified address one module-stack frame carries.

    Parameters
    ----------
    frame:
        Module-stack frame: an object with ``address`` / ``pass_number``
        attributes or a mapping with those keys.

    Returns
    -------
    tuple[str, ...]
        Empty without an address; the bare address; or both spellings.
    """

    frame_address = _frame_field(frame, "address")
    if frame_address is None:
        return ()
    frame_pass = _frame_field(frame, "pass_number")
    if frame_pass is None:
        return (str(frame_address),)
    return (str(frame_address), f"{frame_address}:{frame_pass}")


def _frame_field(frame: Any, name: str) -> Any:
    """Read a module-stack frame field from an attribute, else a mapping key."""

    value = getattr(frame, name, None)
    if value is None and isinstance(frame, dict):
        value = frame.get(name)
    return value


def _module_candidate_string(candidate: Any) -> str:
    """Return a selector-compatible module candidate string.

    Parameters
    ----------
    candidate:
        Module candidate from an op, module call, or capture context.

    Returns
    -------
    str
        Address or pass-qualified address for :func:`module_address_matches`.
    """

    if isinstance(candidate, tuple) and candidate and isinstance(candidate[0], str):
        if len(candidate) > 1:
            return f"{candidate[0]}:{candidate[1]}"
        return candidate[0]
    return str(candidate)


def _is_module_scope_alias(kind: str, subject: Any, lifecycle: str) -> bool:
    """Return whether a module-scoped selector must skip ``subject`` as an alias.

    ``tl.module`` / ``tl.in_module`` address the ops a module call produces or
    contains. Two kinds of subject carry that module stamp without being such
    an op, and matching them would apply a value edit twice:

    - post hoc, the synthetic ``output_N`` node (``is_output``). Postprocess
      mints one per returned value; it executes nothing and re-carries its
      producer's module calls in ``output_of_module_calls`` for display only,
      and replay recomputes it from the producer;
    - live, for ``tl.in_module`` only, a module-boundary leaf whose tensor an
      op INSIDE that call produced (``_tl_boundary_inner_alias``): that op
      already matched at the op door. ``tl.module`` keeps firing at the
      boundary, where op doors never match it.
    """

    if lifecycle == "site":
        return bool(getattr(subject, "is_output", False))
    if lifecycle == "live" and kind == "in_module":
        return bool(getattr(subject, "_tl_module_boundary", False)) and bool(
            getattr(subject, "_tl_boundary_inner_alias", False)
        )
    return False


def _module_output_candidates(subject: Any, lifecycle: str) -> tuple[Any, ...]:
    """Return module-output-boundary candidates for ``tl.module``."""

    module_outputs = tuple(getattr(subject, "output_of_module_calls", ()) or ())
    if lifecycle == "capture" and not module_outputs:
        # Capture subjects include backend ``OpEvent`` records, which carry the
        # owning session as ``source_trace`` (they have no ``trace`` member).
        source_trace: Any = getattr(subject, "source_trace", None)
        if (
            source_trace._provenance_backend if is_live_trace(source_trace) else TORCH_BACKEND_NAME
        ) != TORCH_BACKEND_NAME:
            module_candidate = getattr(subject, "module", None)
            if module_candidate is not None:
                module_outputs = (module_candidate,)
    return module_outputs


def _output_matches(subject: Any, value: Any, lifecycle: str) -> bool:
    """Return whether a subject matches an output index or semantic role."""

    if lifecycle == "capture":
        return getattr(subject, "output_index", None) == value
    if isinstance(value, int):
        return getattr(subject, "multi_output_index", None) == value
    return getattr(subject, "multi_output_name", None) == str(value)


def _site_label_spellings(subject: Any) -> set[str]:
    """Return the final and raw label spellings a finalized site answers to."""

    spellings = (
        getattr(subject, "layer_label", None),
        getattr(subject, "layer_label_raw", None),
        getattr(subject, "_layer_label_raw", None),
    )
    return {label for label in spellings if label is not None}


def _predicate_callable(value: Any) -> Any:
    """Unpack a predicate selector payload without refusing.

    Parameters
    ----------
    value:
        Stored predicate payload: a callable or ``(callable, name_hint)``.

    Returns
    -------
    Any
        The predicate callable, or ``None`` when the payload is malformed (the
        evaluator turns ``None`` into the typed ``tl.where`` refusal at
        evaluation time, so a short-circuited sibling still hides nothing).
    """

    if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
        return value[0]
    if callable(value):
        return value
    return None


def _layer_names(layer: Any) -> set[str]:
    """Return the ``label`` and ``layer_label`` spellings of one layer as strings."""

    return {str(getattr(layer, name, "")) for name in ("label", "layer_label")}


def _shares_grad_fn_tensor(layer: Any, accepted: set[str], paired_parents: set[str]) -> bool:
    """Return whether a same-GradFn ``layer`` provably holds the paired op's tensor.

    Parameters
    ----------
    layer:
        Candidate layer whose ``grad_fn_object_id`` equals the site's.
    accepted:
        Labels already proven to be the tensor (the paired op and prior aliases).
    paired_parents:
        Parent labels of the pass-level paired op (empty for a multi-pass Layer).

    Returns
    -------
    bool
        True for a boundary op, a child of an accepted op, or a parent of the
        paired op.
    """

    if getattr(layer, "is_input", False) or getattr(layer, "is_output", False):
        return True
    parents = {str(parent) for parent in getattr(layer, "parents", ()) or ()}
    return bool(parents & accepted or _layer_names(layer) & paired_parents)


def _grad_fn_type_matches(site: Any, requested: str) -> bool:
    """Return whether a grad_fn_handle type matches user spelling flexibly.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    requested:
        Requested type string.

    Returns
    -------
    bool
        Whether class name or normalized type matches.
    """

    lowered = requested.lower()
    normalized = lowered.removesuffix("backward0").removesuffix("backward")
    candidates = {
        site.class_name.lower(),
        site.type.lower(),
        site.label.lower(),
    }
    return lowered in candidates or normalized in candidates


def _grad_fn_call_values(site: Any) -> tuple[Any, ...]:
    """Return GradFnCall values from dict or accessor-backed call storage.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Any, ...]
        Recorded GradFnCall values.
    """

    calls = site.calls
    if isinstance(calls, dict):
        return tuple(calls.values())
    return tuple(calls._list)


def _grad_fn_forward_op(site: Any) -> Any:
    """Return the forward op one GradFn site pairs with, at its own pass.

    ``GradFn.op`` resolves the paired layer label, which on a multi-pass layer
    is the multi-pass ``Layer`` itself; its per-pass fields (``label``) refuse
    by design. A GradFn belongs to exactly one pass, so the pass whose
    ``grad_fn_object_id`` is this GradFn's autograd identity is the subject
    every forward selector kind evaluates against.
    """

    from torchlens._data_classes.grad_fn import _pass_op_for_grad_fn

    op = site.op
    pass_op = _pass_op_for_grad_fn(op, site.grad_fn_object_id)
    # No pass carries this autograd identity: keep the multi-pass Layer so its
    # typed per-pass refusal fires rather than silently matching nothing.
    return op if pass_op is None else pass_op


def _grad_fn_boundary_aliases(site: Any) -> tuple[Any, ...]:
    """Return every other op sharing a GradFn identity with ``site``.

    One autograd node can back several captured ops: synthetic input/output
    ops, and identity calls that return their input tensor itself (``dropout``
    in eval mode, a no-op ``contiguous`` or ``to``), whose output IS the
    paired op's output. A forward selector naming any of them names that
    tensor's gradient, so each one is a matching subject (a module whose
    output op is an eval-mode dropout matched no GradFn before, and its
    gradient edit was a silent no-op).
    """

    from torchlens._data_classes.grad_fn import _pass_op_for_grad_fn

    trace = site.trace
    if trace is None:
        return ()
    paired_label = site.op_label
    paired = _grad_fn_forward_op(site)
    # Identity is proved, never assumed from a shared ``id()``: a recorded id
    # could in principle be reused by an unrelated node after its first owner
    # was freed. An identity call's input is alive while it runs, so a child
    # of an already-accepted op that shares the node IS the same tensor.
    # A pass-level subject (an Op, or a single-pass Layer) has a label and
    # parents of its own; the multi-pass Layer kept when no pass carries this
    # identity refuses both, so only its bare label is accepted.
    pass_level = _pass_op_for_grad_fn(paired, site.grad_fn_object_id) is not None
    accepted = {str(paired_label)}
    if pass_level:
        accepted.add(str(getattr(paired, "label", paired_label)))
    paired_parents = (
        {str(parent) for parent in getattr(paired, "parents", ()) or ()} if pass_level else set()
    )
    aliases: list[Any] = []
    for layer in trace._layer_list:
        if getattr(layer, "layer_label", None) == paired_label:
            continue
        if getattr(layer, "grad_fn_object_id", None) != site.grad_fn_object_id:
            continue
        if _shares_grad_fn_tensor(layer, accepted, paired_parents):
            aliases.append(layer)
            accepted.update(_layer_names(layer))
    return tuple(aliases)
