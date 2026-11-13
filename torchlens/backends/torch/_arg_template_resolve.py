"""Resolve one call's replay argument template from the facts observed at the call.

The argument template of a call is built in two steps. The observation, made at the
call by ``_set_arg_template_fields`` through ``observe_template``, reads every value the
template carries from live state: each tensor slot's label and capture layout, the copy
of an unlabeled tensor, each literal, the classified component of any other value, the
function's portable identity when its plan does not hold it, and the notes. The
resolver below is a pure function of those recorded facts and the call's layout plan:
it reads no tensor, no label, no option and no trace state, so it returns the call-time
template even after the call's tensors have been freed or mutated.

Recorded facts, ``(plan, arg facts, kwarg facts, function key, notes)``:

- ``plan`` is the layout plan ``(func, registry key fields or None, arg branches,
  kwarg names, kwarg branches)``, or ``None`` when every fact is a finished component
  (a call with no plan, classified the general way; its kwarg facts are then
  ``(name, component)`` pairs).
- A literal slot's fact is the value itself (an immutable builtin by its exact type).
- A plain tensor slot's fact is ``(label, capture layout)`` for a labeled tensor and
  ``(None, literal tensor component)`` for an unlabeled one, whose copy was made at
  the call.
- Any other slot's fact is its component, classified at the call.
- ``function key`` is the observed registry key when the plan holds no key fields.

The field facts handed to :func:`resolve_arg_template_fields` are
``(template facts or None, has kwargs, keep func_id, function key)``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._args_template_cache import TemplateParts


__all__ = (
    "GENERAL",
    "LITERAL",
    "TENSOR",
    "resolve_arg_template",
    "resolve_arg_template_fields",
)

#: Slot branches of a layout plan.
LITERAL = 0
TENSOR = 1
GENERAL = 2


def _resolved_components(
    parts: "TemplateParts", branches: tuple[int, ...], facts: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Return the template components of one argument list's recorded slot facts."""

    literal_value = parts.literal_value
    parent_ref = parts.parent_ref
    out: list[Any] = []
    append = out.append
    for branch, fact in zip(branches, facts, strict=True):
        if branch == LITERAL:
            append(literal_value(fact))
        elif branch == GENERAL:
            append(fact)
        else:
            label, detail = fact
            append(detail if label is None else parent_ref(label, detail))
    return tuple(out)


def resolve_arg_template(parts: "TemplateParts", facts: tuple[Any, ...]) -> Any:
    """Return the ``CapturedArgTemplate`` of one call's recorded template facts."""

    fields: dict[str, Any] = {}
    resolve_arg_template_fields(fields, (facts, False, True, None), parts)
    return fields["args_template"]


def resolve_arg_template_fields(
    fields_dict: dict[str, Any], observed: tuple[Any, ...], parts: "TemplateParts"
) -> None:
    """Write ``args_template``, ``kwargs_template`` and ``func_id`` from recorded facts.

    ``observed`` is ``(template facts or None, has kwargs, keep func_id, function key)``:
    with template facts, ``func_id`` is the template's key when kept; without them (a
    capture that records no template) it is the observed key.
    """

    template_facts, has_kwargs, keep_func_id, key = observed
    if template_facts is None:
        fields_dict["args_template"] = None
        fields_dict["kwargs_template"] = None
        fields_dict["func_id"] = key
        return
    plan, arg_facts, kwarg_facts, key, notes = template_facts
    if plan is None:
        template = parts.template(arg_facts, kwarg_facts, key, notes)
    else:
        # Positional: ``CapturedArgTemplate(args, kwargs, func_id, notes)``.
        _, fields, arg_branches, names, kwarg_branches = plan
        template = parts.template(
            _resolved_components(parts, arg_branches, arg_facts),
            (
                tuple(
                    zip(
                        names,
                        _resolved_components(parts, kwarg_branches, kwarg_facts),
                        strict=True,
                    )
                )
                if names
                else ()
            ),
            key if fields is None else parts.key_type(*fields),
            notes,
        )
    fields_dict["args_template"] = template
    fields_dict["kwargs_template"] = template if has_kwargs else None
    fields_dict["func_id"] = template.func_id if keep_func_id else None
