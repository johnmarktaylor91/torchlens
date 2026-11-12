"""Replay adapter contract: the function protocol and the data-only op recipe.

Every non-PyTorch backend already replays each recorded op from its parents
for validation. A replay adapter exposes that per-op replay through one
module per backend (``_replay_adapters/<name>.py``) so that editing a
finished trace, validation after load, the reverse gradient engine and
gradient-site edits share it.

An adapter module defines the module-level functions of
:class:`ReplayAdapter`. Framework imports stay inside those functions, so
importing an adapter module never imports its framework.

:class:`OpRecipe` is the data-only description of one recorded op. It holds
no callables, pickles or live framework handles: every leaf is ``None``, a
bool, an int, a float, a complex, a str, or a tuple or str-keyed mapping of
those. :func:`encode_recipe` and :func:`decode_recipe` move a recipe to and
from a versioned, bounded, JSON-safe tree; :func:`decode_recipe` refuses a
corrupt tree or an op identity outside the backend's closed allowlist with
:class:`ReplayRefusal` before anything executes.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

__all__ = (
    "RECIPE_FORMAT_VERSION",
    "RECIPE_LIMITS",
    "ArgPath",
    "InputOccurrence",
    "OpRecipe",
    "RecipeLimits",
    "ReplayAdapter",
    "ReplayRefusal",
    "decode_recipe",
    "encode_recipe",
    "validate_recipe",
)

RECIPE_FORMAT_VERSION = 1
"""Version of the encoded recipe tree. A decoder refuses any other version."""

ArgPath = tuple[str | int, ...]
"""Path of one argument leaf: positional index or keyword name, then the
flatten path inside that argument (ints index sequences, strs key mappings)."""

_RECIPE_FIELDS = (
    "backend",
    "op_identity",
    "inputs",
    "literals",
    "attrs",
    "output_slots",
    "param_refs",
    "constant_refs",
    "rng",
    "framework_version",
)


class ReplayRefusal(Exception):
    """Internal signal: the replay layer refused a recipe, a plan or a lookup.

    It is raised before any framework call and never reaches a user directly:
    the callers (loading a saved trace, editing a finished trace, the reverse
    engine) catch it and raise their own contracted, coded error naming
    ``op_label``. It is deliberately not an ``*Error`` class and has no error
    code of its own.

    Attributes
    ----------
    reason:
        Short tag. Recipes: ``format_version``, ``shape``, ``type``,
        ``bounds``, ``backend``, ``unknown_op``, ``slots``. Sessions:
        ``cannot_replay``, ``missing_parent``, ``duplicate_slot``, ``arity``,
        ``closed``, ``unknown_label``, ``substitution``. Lookup:
        ``no_adapter``.
    op_label:
        The op's label (first output slot) when known, else ``None``.
    """

    def __init__(self, reason: str, detail: str, *, op_label: str | None = None) -> None:
        self.reason = reason
        self.op_label = op_label
        where = f" (op {op_label})" if op_label else ""
        super().__init__(f"replay refused [{reason}]{where}: {detail}")


@dataclass(frozen=True)
class RecipeLimits:
    """Bounds every decoded recipe must respect."""

    max_inputs: int = 4096
    max_outputs: int = 4096
    max_refs: int = 4096
    max_depth: int = 64
    max_nodes: int = 200_000
    max_str_len: int = 65_536
    max_int_bits: int = 256


RECIPE_LIMITS = RecipeLimits()


@dataclass(frozen=True)
class InputOccurrence:
    """One parent occurrence in an op's call, in call order.

    A parent used twice appears twice, each with its own ``arg_path``.
    """

    parent_label: str
    arg_path: ArgPath


@dataclass(frozen=True)
class OpRecipe:
    """Data-only description of one recorded op.

    Attributes
    ----------
    backend:
        Backend name (``jax``, ``tf``, ``mlx``, ``paddle``, ``tinygrad``).
    op_identity:
        Key into the backend's closed op allowlist; never a callable path the
        loader imports.
    inputs:
        Ordered parent occurrences; repeated parents kept.
    literals:
        ``(arg_path, value)`` for non-array arguments, value data only.
    attrs:
        Op attributes encoded structurally (nested programs recursively as
        mappings, never ``str(params)``).
    output_slots:
        Output labels in emission order; a multi-output op has several and
        replays atomically.
    param_refs:
        ``(arg_path, parameter address)`` for parameter leaves.
    constant_refs:
        ``(arg_path, constant label)`` for closure constants with payloads.
    rng:
        Random-state dependency as a data mapping, or ``None``.
    framework_version:
        Framework version string at capture.
    """

    backend: str
    op_identity: str
    inputs: tuple[InputOccurrence, ...]
    literals: tuple[tuple[ArgPath, Any], ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    output_slots: tuple[str, ...] = ()
    param_refs: tuple[tuple[ArgPath, str], ...] = ()
    constant_refs: tuple[tuple[ArgPath, str], ...] = ()
    rng: Mapping[str, Any] | None = None
    framework_version: str = ""

    @property
    def parent_labels(self) -> tuple[str, ...]:
        """Parent labels per occurrence, in call order, repeats kept."""

        return tuple(occ.parent_label for occ in self.inputs)


@runtime_checkable
class ReplayAdapter(Protocol):
    """Module-level functions every ``_replay_adapters/<name>.py`` defines.

    ``op`` is the recorded :class:`~torchlens.data_classes.Op` (or the
    backend's runtime capture for it); ``parent_values`` holds one framework
    array per :attr:`OpRecipe.inputs` occurrence, repeats kept.

    ``ALLOWLIST`` is a module-level ``frozenset[str]``: the closed set of
    :attr:`OpRecipe.op_identity` values the adapter replays.
    :func:`decode_recipe` takes it as ``allowlist``.
    """

    ALLOWLIST: frozenset[str]

    def can_replay(self, trace: Any, op: Any) -> str | None:
        """``None``, or why this op cannot be replayed (opaque region,
        graph-only entry, unlisted raw op, random op without RNG support)."""

    def forward(self, trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
        """Re-run this one op through the validator's existing replay; one
        value per output slot; differentiable by the framework."""

    def vjp(
        self,
        trace: Any,
        op: Any,
        parent_values: Sequence[Any],
        cotangents: Sequence[Any],
    ) -> tuple[Any | None, ...]:
        """The framework's own derivative of this op at these inputs: one
        cotangent per parent occurrence (``None`` where not differentiable)."""

    def recipe(self, trace: Any, op: Any) -> OpRecipe | None:
        """Data-only description of the op, or ``None`` when it has none."""

    def bind_recipes(self, trace: Any, recipes: Mapping[str, OpRecipe]) -> None:
        """Rebuild the runtime capture list from recipes plus loaded payloads.
        Keys are op labels (the first output slot)."""

    def zeros_like(self, value: Any) -> Any:
        """Framework zeros with the value's shape and dtype."""

    def add(self, left: Any, right: Any) -> Any:
        """Framework elementwise sum (gradient accumulation)."""

    def to_host(self, value: Any) -> Any:
        """Host (NumPy) copy of a framework value."""

    def close(self, expected: Any, actual: Any) -> bool:
        """The backend validator's own comparison, imported; never a new
        tolerance."""


def encode_recipe(recipe: OpRecipe) -> dict[str, Any]:
    """Return the versioned JSON-safe tree for ``recipe``.

    Raises
    ------
    ReplayRefusal
        When a field holds a non-data value (callable, array, object).
    """

    validate_recipe(recipe)
    label = recipe.output_slots[0] if recipe.output_slots else None
    return {
        "format_version": RECIPE_FORMAT_VERSION,
        "backend": recipe.backend,
        "op_identity": recipe.op_identity,
        "inputs": [[occ.parent_label, _encode_path(occ.arg_path)] for occ in recipe.inputs],
        "literals": [[_encode_path(p), _encode_value(v, label)] for p, v in recipe.literals],
        "attrs": _encode_value(dict(recipe.attrs), label),
        "output_slots": list(recipe.output_slots),
        "param_refs": [[_encode_path(p), ref] for p, ref in recipe.param_refs],
        "constant_refs": [[_encode_path(p), ref] for p, ref in recipe.constant_refs],
        "rng": None if recipe.rng is None else _encode_value(dict(recipe.rng), label),
        "framework_version": recipe.framework_version,
    }


def decode_recipe(
    data: Any,
    *,
    backend: str,
    allowlist: Collection[str],
    limits: RecipeLimits = RECIPE_LIMITS,
) -> OpRecipe:
    """Rebuild an :class:`OpRecipe` from an encoded tree, refusing bad input.

    Parameters
    ----------
    data:
        The tree :func:`encode_recipe` produced (possibly after a JSON round
        trip, so tuples may arrive as lists).
    backend:
        The backend the caller is loading for; a recipe for another is refused.
    allowlist:
        The backend's closed set of op identities.
    limits:
        Size bounds.

    Raises
    ------
    ReplayRefusal
        On any structural, type, bound, backend or allowlist failure.
    """

    _check_bounds(data, limits)
    if not isinstance(data, Mapping):
        raise ReplayRefusal("shape", "recipe is not a mapping")
    version = data.get("format_version")
    if type(version) is not int or version != RECIPE_FORMAT_VERSION:
        raise ReplayRefusal("format_version", f"unsupported format version {version!r}")
    expected = {"format_version", *_RECIPE_FIELDS}
    if set(data) != expected:
        diff = sorted(map(str, set(data) ^ expected))
        raise ReplayRefusal("shape", f"recipe keys differ from the contract: {diff}")
    recipe = _build_recipe(data, limits)
    label = recipe.output_slots[0] if recipe.output_slots else None
    if recipe.backend != backend:
        raise ReplayRefusal(
            "backend", f"recipe is for {recipe.backend!r}, loading {backend!r}", op_label=label
        )
    if recipe.op_identity not in allowlist:
        raise ReplayRefusal(
            "unknown_op",
            f"op identity {recipe.op_identity!r} is not in the {backend} allowlist",
            op_label=label,
        )
    validate_recipe(recipe)
    return recipe


def _build_recipe(data: Mapping[str, Any], limits: RecipeLimits) -> OpRecipe:
    """Interpret each field of a shape-checked tree, refusing bad types."""

    slots = _str_list(data["output_slots"], "output_slots", limits.max_outputs)
    label = slots[0] if slots else None
    return OpRecipe(
        backend=_req_str(data["backend"], "backend", label),
        op_identity=_req_str(data["op_identity"], "op_identity", label),
        inputs=tuple(
            InputOccurrence(_req_str(p, "inputs", label), _decode_path(a, label))
            for p, a in _pairs(data["inputs"], "inputs", limits.max_inputs, label)
        ),
        literals=tuple(
            (_decode_path(p, label), _decode_value(v, label))
            for p, v in _pairs(data["literals"], "literals", limits.max_refs, label)
        ),
        attrs=_decode_mapping(data["attrs"], "attrs", label),
        output_slots=tuple(slots),
        param_refs=_decode_refs(data["param_refs"], "param_refs", limits, label),
        constant_refs=_decode_refs(data["constant_refs"], "constant_refs", limits, label),
        rng=None if data["rng"] is None else _decode_mapping(data["rng"], "rng", label),
        framework_version=_req_str(data["framework_version"], "framework_version", label),
    )


def validate_recipe(recipe: OpRecipe) -> None:
    """Check that ``recipe`` is data only and internally consistent.

    Raises
    ------
    ReplayRefusal
        On a non-data leaf, an empty or repeated output slot, or an empty
        backend or op identity.
    """

    label = recipe.output_slots[0] if recipe.output_slots else None
    if not recipe.backend or not recipe.op_identity:
        raise ReplayRefusal("shape", "backend and op_identity must be non-empty", op_label=label)
    if not recipe.output_slots:
        raise ReplayRefusal("slots", "recipe has no output slots")
    if len(set(recipe.output_slots)) != len(recipe.output_slots):
        raise ReplayRefusal("slots", "output slots repeat", op_label=label)
    if any(not isinstance(occ, InputOccurrence) for occ in recipe.inputs):
        raise ReplayRefusal("type", "inputs must be InputOccurrence items", op_label=label)
    for _path, value in recipe.literals:
        _encode_value(value, label)
    _encode_value(dict(recipe.attrs), label)
    if recipe.rng is not None:
        _encode_value(dict(recipe.rng), label)


# --- value encoding -------------------------------------------------------

_TAGS = frozenset({"$tuple", "$dict", "$float", "$complex"})


def _encode_value(value: Any, label: str | None) -> Any:
    """Encode one data value as a JSON-safe tagged tree (essential type switch)."""

    if value is None or type(value) in (bool, str):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"$float": repr(value)}
    if type(value) is complex:
        return {"$complex": [_encode_value(value.real, label), _encode_value(value.imag, label)]}
    if type(value) is tuple:
        return {"$tuple": [_encode_value(v, label) for v in value]}
    if type(value) is list:
        return [_encode_value(v, label) for v in value]
    if type(value) in (dict, MappingProxyType):
        return {"$dict": {_key(k, label): _encode_value(v, label) for k, v in value.items()}}
    raise ReplayRefusal(
        "type", f"non-data value of type {type(value).__name__} in recipe", op_label=label
    )


def _key(key: Any, label: str | None) -> str:
    if type(key) is not str:
        raise ReplayRefusal("type", f"mapping key {key!r} is not a str", op_label=label)
    return key


def _decode_value(node: Any, label: str | None) -> Any:
    """Decode one tagged tree node (essential type switch)."""

    if node is None or type(node) in (bool, str, int):
        return node
    if type(node) is float:
        if not math.isfinite(node):
            raise ReplayRefusal("type", "untagged non-finite float", op_label=label)
        return node
    if isinstance(node, list):
        return [_decode_value(v, label) for v in node]
    if not isinstance(node, Mapping) or len(node) != 1:
        raise ReplayRefusal("type", f"bad recipe value {type(node).__name__}", op_label=label)
    ((tag, body),) = node.items()
    if tag not in _TAGS:
        raise ReplayRefusal("type", f"unknown value tag {tag!r}", op_label=label)
    return _decode_tagged(tag, body, label)


def _decode_tagged(tag: str, body: Any, label: str | None) -> Any:
    if tag == "$tuple" and isinstance(body, list):
        return tuple(_decode_value(v, label) for v in body)
    if tag == "$dict" and isinstance(body, Mapping):
        return {_key(k, label): _decode_value(v, label) for k, v in body.items()}
    if tag == "$float" and body in ("nan", "inf", "-inf"):
        return float(body)
    if tag == "$complex" and isinstance(body, list) and len(body) == 2:
        real, imag = (_decode_value(v, label) for v in body)
        if type(real) is float and type(imag) is float:
            return complex(real, imag)
    raise ReplayRefusal("type", f"malformed {tag} value", op_label=label)


def _decode_mapping(node: Any, name: str, label: str | None) -> dict[str, Any]:
    value = _decode_value(node, label)
    if not isinstance(value, dict):
        raise ReplayRefusal("type", f"{name} is not a mapping", op_label=label)
    return value


# --- structural helpers ---------------------------------------------------


def _encode_path(path: ArgPath) -> list[str | int]:
    return [_path_part(p, None) for p in path]


def _decode_path(node: Any, label: str | None) -> ArgPath:
    if not isinstance(node, (list, tuple)) or not node:
        raise ReplayRefusal("type", "argument path must be a non-empty list", op_label=label)
    return tuple(_path_part(p, label) for p in node)


def _path_part(part: Any, label: str | None) -> str | int:
    if type(part) is int and part >= 0 or type(part) is str:
        return cast("str | int", part)
    raise ReplayRefusal("type", f"bad argument path part {part!r}", op_label=label)


def _req_str(value: Any, name: str, label: str | None) -> str:
    if type(value) is not str:
        raise ReplayRefusal("type", f"{name} must be a str", op_label=label)
    return value


def _str_list(node: Any, name: str, limit: int) -> list[str]:
    if not isinstance(node, (list, tuple)) or len(node) > limit:
        raise ReplayRefusal("shape", f"{name} must be a list of at most {limit}")
    return [_req_str(v, name, None) for v in node]


def _pairs(node: Any, name: str, limit: int, label: str | None) -> list[tuple[Any, Any]]:
    if not isinstance(node, (list, tuple)) or len(node) > limit:
        raise ReplayRefusal("shape", f"{name} must be a list of at most {limit}", op_label=label)
    pairs = []
    for item in node:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ReplayRefusal("shape", f"{name} entries must be pairs", op_label=label)
        pairs.append((item[0], item[1]))
    return pairs


def _decode_refs(
    node: Any, name: str, limits: RecipeLimits, label: str | None
) -> tuple[tuple[ArgPath, str], ...]:
    return tuple(
        (_decode_path(p, label), _req_str(r, name, label))
        for p, r in _pairs(node, name, limits.max_refs, label)
    )


def _check_bounds(data: Any, limits: RecipeLimits) -> None:
    """Walk the raw tree once, iteratively, refusing depth, size or leaf
    types outside the data grammar before any field is interpreted."""

    stack: list[tuple[Any, int]] = [(data, 0)]
    nodes = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            raise ReplayRefusal("bounds", "recipe tree exceeds the size or depth bound")
        if isinstance(node, str):
            if len(node) > limits.max_str_len:
                raise ReplayRefusal("bounds", "recipe string exceeds the length bound")
        elif type(node) is int:
            if node.bit_length() > limits.max_int_bits:
                raise ReplayRefusal("bounds", "recipe integer exceeds the size bound")
        elif isinstance(node, Mapping):
            for key, value in node.items():
                if type(key) is not str:
                    raise ReplayRefusal("type", "recipe mapping key is not a str")
                stack.append((key, depth + 1))
                stack.append((value, depth + 1))
        elif isinstance(node, (list, tuple)):
            stack.extend((v, depth + 1) for v in node)
        elif node is not None and type(node) not in (bool, float):
            raise ReplayRefusal("type", f"non-data leaf {type(node).__name__} in recipe")
