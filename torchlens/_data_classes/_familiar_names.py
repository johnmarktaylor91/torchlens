"""Hints for attribute names that TorchLens objects do not have but users guess.

Coding agents and returning users reach for names from other libraries, from
TorchLens 1.x, or from plain PyTorch habit (``op.activation``,
``trace.layer_list``, ``op.parent_layers``). Those names are not aliases and
never resolve; a failed lookup only names the field that answers the question,
so the next attempt is right. Python's own "Did you mean" suggestion picks the
nearest spelling, which for these names is often a different field
(``parent_layers`` suggests ``parent_params``).
"""

from __future__ import annotations

_OUT = "use .out (the saved output tensor)"
_PARENTS = "use .parents (labels of the ops feeding this one; .get_parents() returns objects)"
_CHILDREN = "use .children (labels of the ops this one feeds; .get_children() returns objects)"
_ADDRESS = "use .address (innermost module name; None for top-level forward code)"
_LAYERS = "iterate trace.layers (every op in execution order) or index trace[label]"

#: Hints shared by ops, layers and modules (per-op and per-module readers).
_ENTRY_HINTS: dict[str, str] = {
    "activation": _OUT,
    "activations": _OUT,
    "tensor_contents": _OUT,
    "tensor": _OUT,
    "output": _OUT,
    "value": _OUT,
    "parent_layers": _PARENTS,
    "parent_ops": _PARENTS,
    "child_layers": _CHILDREN,
    "child_ops": _CHILDREN,
    "containing_module": _ADDRESS,
    "containing_modules": "use .modules (every enclosing module call, outermost first)",
    "module_address": _ADDRESS,
    "module_name": _ADDRESS,
    "function": "use .func_name (the function's name, e.g. 'linear' or 'softmax')",
    "func": "use .func_name (the function's name, e.g. 'linear' or 'softmax')",
    "op_name": "use .func_name for the function or .label for the op's own name",
    "name": "use .label for the op's name or .func_name for the function",
    "shape": "use .out.shape (the saved output tensor)",
}

#: Hints for the whole-capture object.
_TRACE_HINTS: dict[str, str] = {
    "layer_list": _LAYERS,
    "layers_list": _LAYERS,
    "ops_list": _LAYERS,
    "layer_dict": _LAYERS,
    "layer_dict_main_keys": _LAYERS,
    "layer_labels": "use [op.label for op in trace.layers]",
    "modules_list": "iterate trace.modules or index trace.modules['blocks.0']",
    "module_dict": "index trace.modules['blocks.0']",
    "activations": "use tl.extract(model, x, names) for {module: tensor}, or op.out per op",
    "outputs": "use trace.output (exactly what the model returned)",
    "model_output": "use trace.output (exactly what the model returned)",
    "logits": "use trace.output (for HuggingFace models, trace.output.logits)",
    "nodes": _LAYERS,
}

_HINTS_BY_OWNER: dict[str, dict[str, str]] = {
    "Trace": _TRACE_HINTS,
    "Layer": _ENTRY_HINTS,
    "Op": _ENTRY_HINTS,
}


#: The op-level names with a hint (``Op.__getattribute__`` checks membership on a miss).
FAMILIAR_OP_NAMES: frozenset[str] = frozenset(_ENTRY_HINTS)


def familiar_name_hint(owner: str, name: str) -> str | None:
    """Return the hint for a guessed attribute name, or None when there is none.

    Parameters
    ----------
    owner:
        Public class name of the object the lookup failed on (``"Trace"``,
        ``"Layer"``, ``"Op"``).
    name:
        The attribute name that does not exist.

    Returns
    -------
    str | None
        A one-line pointer to the field that answers the question.
    """

    return _HINTS_BY_OWNER.get(owner, {}).get(name)


def missing_attribute_error(owner: str, name: str) -> AttributeError:
    """Build the AttributeError for a failed lookup, with a hint when one applies.

    Parameters
    ----------
    owner:
        Public class name used in the message.
    name:
        The attribute name that does not exist.

    Returns
    -------
    AttributeError
        The exception to raise. When a hint applies, ``obj`` is set to None so
        Python does not append a misleading nearest-spelling suggestion.
    """

    hint = familiar_name_hint(owner, name)
    if hint is None:
        return AttributeError(f"'{owner}' object has no attribute '{name}'")
    return AttributeError(
        f"'{owner}' object has no attribute '{name}'; {hint}", name=name, obj=None
    )
