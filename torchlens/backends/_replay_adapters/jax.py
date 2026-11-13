"""JAX replay adapter: per-op replay, derivative and data-only recipe.

Thin wrappers over the replay the JAX validator already uses
(:func:`torchlens.backends.jax.jaxpr.replay_equation`, which binds each
captured primitive with its captured params). Nothing here changes a check:
:func:`close` is the validator's own comparison, and :func:`bind_recipes`
only rebuilds the runtime capture list the unchanged validator reads.

Ops are matched to captures exactly as the validator pairs them (equation
captures zipped with equation ops in execution order). Parent occurrences are
the op's data parents in argument-position order, so a parent read twice is
two occurrences. A multi-result JAX equation is one op whose payload is the
tuple of its outputs, so it has one output slot whose value is that tuple.

JAX randomness is functional: a PRNG primitive's key is an explicit graph
input, so key-based random ops replay exactly and need no RNG state. Only
``rng_uniform`` (XLA's stateful generator) is refused.
"""

from __future__ import annotations

import dataclasses
import enum
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._contract import InputOccurrence, OpRecipe, ReplayRefusal

BACKEND = "jax"

_SYNTHETIC_KINDS = ("scan_read", "scan_stack", "cond_decision", "while_decision")
_PRIMITIVE_NAMES_TEXT = (
    "abs acos acosh add and argmax argmin asin asinh atan atan2 atanh bessel_i0e "
    "bessel_i1e bitcast_convert_type broadcast_in_dim cbrt ceil clamp clz complex "
    "concatenate conj conv_general_dilated convert_element_type copy copy_p cos cosh "
    "cumlogsumexp cummax cummin cumprod cumsum digamma div dot_general dynamic_slice "
    "dynamic_update_slice eq erf erf_inv erfc exp exp2 expm1 floor gather ge gt igamma "
    "igammac imag integer_pow iota is_finite le lgamma log log1p logistic lt max min mul "
    "ne neg nextafter not optimization_barrier or pad polygamma population_count pow "
    "random_bits random_fold_in random_seed random_split random_unwrap random_wrap real "
    "reduce_and reduce_max reduce_min reduce_or reduce_precision reduce_prod reduce_sum "
    "reduce_window_max reduce_window_min reduce_window_sum reduce_xor "
    "regularized_incomplete_beta rem reshape rev round rsqrt scatter scatter-add "
    "scatter-max scatter-min scatter-mul select_and_gather_add select_and_scatter_add "
    "select_n shift_left shift_right_arithmetic shift_right_logical sign sin sinh slice "
    "sort split sqrt square squeeze sub tan tanh threefry2x32 top_k transpose xor zeta"
)
_PRIMITIVE_NAMES = tuple(_PRIMITIVE_NAMES_TEXT.split())

ALLOWLIST: frozenset[str] = frozenset(
    {f"jax.{kind}" for kind in _SYNTHETIC_KINDS}
    | {f"jax.primitive.{name}" for name in _PRIMITIVE_NAMES}
)
"""Closed set of op identities a recipe may name. A primitive outside it
(nested control flow bound whole, custom-derivative calls, jit calls, effects)
has no recipe; the runtime capture still replays it live."""

_STATEFUL_RANDOM_PRIMITIVES = frozenset({"rng_uniform"})
_ENUM_TYPE_NAMES = ("Precision", "GatherScatterMode", "RoundingMethod", "FftType")
_STRUCT_TYPE_NAMES = ("ConvDimensionNumbers", "GatherDimensionNumbers", "ScatterDimensionNumbers")
_TAG = "$jax"
_MAP_TAG = "$map"


class _Unencodable(ValueError):
    """A capture value has no data-only encoding (callable, sharding, effect)."""


# --- op and capture resolution ----------------------------------------------


@dataclass(frozen=True)
class _TraceIndex:
    source: Any
    ops_by_label: Mapping[str, Any]
    capture_by_op: Mapping[int, Any]
    op_by_capture_index: Mapping[int, Any]


@dataclass(frozen=True)
class _Plan:
    """How to replay one op: ``kind`` is equation, source or passthrough."""

    kind: str
    op: Any
    capture: Any = None
    positions: tuple[int, ...] = ()
    parents: tuple[Any, ...] = ()
    element_index: tuple[int | None, ...] = ()


_INDEX_CACHE: weakref.WeakKeyDictionary[Any, _TraceIndex] = weakref.WeakKeyDictionary()


def _trace_index(trace: Any) -> _TraceIndex:
    """Pair captures with equation ops the way the validator does (cached)."""

    from ..jax.backend import _jax_equation_ops, _jax_ops_by_label

    raw = getattr(trace, "jax_equation_captures", None)
    try:
        cached = _INDEX_CACHE.get(trace)
    except TypeError:
        cached = None
    if cached is not None and cached.source is raw:
        return cached
    captures = tuple(raw or ())
    ops = _jax_equation_ops(trace)
    pairs = tuple(zip(captures, ops, strict=False)) if len(captures) == len(ops) else ()
    index = _TraceIndex(
        source=raw,
        ops_by_label=_jax_ops_by_label(trace),
        capture_by_op={id(op): capture for capture, op in pairs},
        op_by_capture_index={capture.index: op for capture, op in pairs},
    )
    try:
        _INDEX_CACHE[trace] = index
    except TypeError:
        pass
    return index


def _kind_annotation(op: Any) -> str | None:
    annotations = getattr(op, "annotations", None)
    kind = annotations.get("jax_capture_kind") if isinstance(annotations, Mapping) else None
    return kind if isinstance(kind, str) else None


def _resolve(trace: Any, op: Any) -> tuple[Any, Any]:
    """Return ``(capture or None, materialized op or None)`` for an op or capture."""

    from ..jax.jaxpr import JaxEquationCapture, JaxRegionCapture

    index = _trace_index(trace)
    if isinstance(op, JaxEquationCapture | JaxRegionCapture):
        return op, index.op_by_capture_index.get(op.index)
    return index.capture_by_op.get(id(op)), op


def _parent_label(op: Any) -> str:
    return str(getattr(op, "label", None) or getattr(op, "_label_raw", ""))


def _equation_plan(trace: Any, capture: Any, op: Any) -> _Plan:
    """Occurrences in argument order with the validator's parent consistency check."""

    from ..jax._trace_graph_helpers import _data_parent_arg_positions, _data_parent_labels

    index = _trace_index(trace)
    graph_positions = _data_parent_arg_positions(op)
    if set(graph_positions.values()) != _data_parent_labels(op):
        raise ValueError("JAX trace data parent labels and parent_arg_positions disagree.")
    positions = tuple(sorted(graph_positions))
    if any(p < 0 or p >= len(capture.input_values) for p in positions):
        raise ValueError("JAX trace parent arg position is out of range.")
    parents = tuple(index.ops_by_label[graph_positions[p]] for p in positions)
    element_index = tuple(
        _element_index(index, parent, capture.input_values[p])
        for p, parent in zip(positions, parents, strict=True)
    )
    return _Plan("equation", op, capture, positions, parents, element_index)


def _element_index(index: _TraceIndex, parent: Any, recorded: Any) -> int | None:
    """Which output of a multi-result parent the child read (None: single)."""

    parent_capture = index.capture_by_op.get(id(parent))
    outputs = getattr(parent_capture, "output_values", ())
    if len(outputs) < 2:
        return None
    for position, value in enumerate(outputs):
        if value is recorded:
            return position
    raise ValueError("JAX multi-output parent has no output identical to the child's input.")


def _plan_or_reason(trace: Any, op: Any) -> _Plan | str:
    """The replay plan for ``op``, or the reason it cannot be replayed."""

    if getattr(trace, "backend", BACKEND) not in (BACKEND, None):
        return f"not a JAX trace (backend {getattr(trace, 'backend', None)!r})"
    capture, materialized = _resolve(trace, op)
    if materialized is None:
        return "no recorded op matches this JAX capture"
    kind = _kind_annotation(materialized)
    if kind in ("region", "region_output"):
        reason = getattr(capture, "unverified_reason", "jax_region_not_per_op_replayable")
        return f"opaque region ({reason}): JAX replays it only as a whole"
    if kind is None:
        return _leaf_plan_or_reason(trace, materialized)
    if capture is None:
        return "no runtime capture for this op (stripped at save; bind recipes first)"
    return _equation_plan_or_reason(trace, capture, materialized)


def _equation_plan_or_reason(trace: Any, capture: Any, op: Any) -> _Plan | str:
    from ..jax.jaxpr import EFFECT_PRIMITIVES, JAX_REPLAY_HANDLERS

    if capture.kind not in JAX_REPLAY_HANDLERS:
        return f"unlisted JAX replay kind {capture.kind!r}"
    if capture.kind != _kind_annotation(op):
        return "capture kind differs from the op's recorded kind"
    if capture.primitive in _STATEFUL_RANDOM_PRIMITIVES:
        return f"random op without RNG support ({capture.primitive} draws from hidden state)"
    if capture.primitive in EFFECT_PRIMITIVES:
        return f"effectful callback {capture.primitive}"
    try:
        return _equation_plan(trace, capture, op)
    except (KeyError, ValueError) as exc:
        return f"parent wiring cannot be resolved: {exc}"


def _leaf_plan_or_reason(trace: Any, op: Any) -> _Plan | str:
    """Sources replay their saved payload; an output node passes its parent through."""

    from ..._utils._record_group_table import record_member

    parents = tuple(record_member(op, "parents", None) or ())
    if not parents:
        if not getattr(op, "has_saved_activation", False):
            return "source op has no saved payload"
        return _Plan("source", op)
    parent = _trace_index(trace).ops_by_label.get(parents[0])
    if record_member(op, "is_output", False) and len(parents) == 1 and parent is not None:
        return _Plan("passthrough", op, None, (0,), (parent,), (None,))
    return "not a recorded JAX equation"


def _plan(trace: Any, op: Any) -> _Plan:
    plan = _plan_or_reason(trace, op)
    if isinstance(plan, str):
        raise ValueError(f"JAX replay refused for {_parent_label(_resolve(trace, op)[1])}: {plan}")
    return plan


# --- contract functions -----------------------------------------------------


def can_replay(trace: Any, op: Any) -> str | None:
    """``None``, or why this op cannot be replayed."""

    plan = _plan_or_reason(trace, op)
    return plan if isinstance(plan, str) else None


def step(trace: Any, op: Any) -> Any:
    """The :class:`ReplayStep` for ``op``: parent occurrences and output slots.

    Available for every replayable op, including those without a recipe.
    """

    from ._session import ReplayStep

    plan = _plan(trace, op)
    pairs = zip(plan.positions, plan.parents, strict=True)
    inputs = tuple(InputOccurrence(_parent_label(par), (pos,)) for pos, par in pairs)
    return ReplayStep(op=plan.op, inputs=inputs, output_slots=(_parent_label(plan.op),))


def _replay_inputs(plan: _Plan, parent_values: Sequence[Any]) -> list[Any]:
    """Captured inputs with each parent occurrence replaced by its given value."""

    if len(parent_values) != len(plan.positions):
        raise ValueError(f"JAX replay expects {len(plan.positions)} parent values.")
    inputs = list(plan.capture.input_values) if plan.capture is not None else [None]
    for position, element, value in zip(
        plan.positions, plan.element_index, parent_values, strict=True
    ):
        inputs[position] = _select(value, element)
    return inputs


def _select(value: Any, element: int | None) -> Any:
    if element is not None and isinstance(value, tuple | list):
        return value[element]
    return value


def _to_slots(outputs: Sequence[Any]) -> tuple[Any, ...]:
    return (outputs[0],) if len(outputs) == 1 else (tuple(outputs),)


def forward(trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
    """Re-run one op through the validator's replay; one value per output slot."""

    from ..jax.jaxpr import replay_equation

    plan = _plan(trace, op)
    if plan.kind == "source":
        return (plan.op.out,)
    if plan.kind == "passthrough":
        return tuple(_replay_inputs(plan, parent_values))
    return _to_slots(replay_equation(plan.capture, _replay_inputs(plan, parent_values)))


def _is_inexact(value: Any) -> bool:
    import jax.numpy as jnp

    dtype = getattr(value, "dtype", None)
    return dtype is not None and bool(jnp.issubdtype(dtype, jnp.inexact))


def _output_cotangent(cotangent: Any, primal: Any) -> Any:
    """The cotangent ``jax.vjp`` wants for one primal output (zeros for None)."""

    import jax
    import numpy as np

    if not _is_inexact(primal):
        return np.zeros(np.shape(primal), dtype=jax.dtypes.float0)
    if cotangent is None:
        return zeros_like(primal)
    return cotangent


def vjp(
    trace: Any, op: Any, parent_values: Sequence[Any], cotangents: Sequence[Any]
) -> tuple[Any | None, ...]:
    """JAX's own derivative of this one op (``jax.vjp`` over its primitive bind).

    One cotangent per parent occurrence, ``None`` where the parent is not
    differentiable (integer, boolean or key dtype). A parent given as a
    multi-result tuple gets a tuple with ``None`` in the slots the op did not read.
    """

    import jax

    from ..jax.jaxpr import replay_equation

    plan = _plan(trace, op)
    if plan.kind == "source":
        return ()
    if plan.kind == "passthrough":
        return (cotangents[0],)
    if len(cotangents) != 1:
        raise ValueError("JAX vjp expects one cotangent per output slot (1).")
    inputs = _replay_inputs(plan, parent_values)
    diff = tuple(i for i, p in enumerate(plan.positions) if _is_inexact(inputs[p]))
    if not diff:
        return (None,) * len(plan.positions)

    def replay(*diff_values: Any) -> tuple[Any, ...]:
        replaced = list(inputs)
        for slot, value in zip(diff, diff_values, strict=True):
            replaced[plan.positions[slot]] = value
        return tuple(replay_equation(plan.capture, replaced))

    primals, pullback = jax.vjp(replay, *(inputs[plan.positions[i]] for i in diff))
    slot_cts = cotangents[0] if len(primals) > 1 else (cotangents[0],)
    if slot_cts is None:
        slot_cts = (None,) * len(primals)
    grads = pullback(tuple(_output_cotangent(c, p) for c, p in zip(slot_cts, primals, strict=True)))
    by_slot = dict(zip(diff, grads, strict=True))
    return tuple(
        _wrap_element(by_slot.get(i), parent_values[i], plan.element_index[i])
        for i in range(len(plan.positions))
    )


def _wrap_element(grad: Any, parent_value: Any, element: int | None) -> Any:
    if element is None or not isinstance(parent_value, tuple | list):
        return grad
    return tuple(grad if k == element else None for k in range(len(parent_value)))


def zeros_like(value: Any) -> Any:
    """JAX zeros with the value's shape and dtype (tuples elementwise)."""

    import jax.numpy as jnp

    if isinstance(value, tuple | list):
        return tuple(zeros_like(v) for v in value)
    return jnp.zeros_like(value)


def add(left: Any, right: Any) -> Any:
    """Elementwise sum for gradient accumulation; ``None`` is a zero."""

    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, tuple | list):
        return tuple(add(a, b) for a, b in zip(left, right, strict=True))
    return left + right


def to_host(value: Any) -> Any:
    """Host NumPy copy (tuples elementwise)."""

    import numpy as np

    if isinstance(value, tuple | list):
        return tuple(to_host(v) for v in value)
    return np.asarray(value)


def close(expected: Any, actual: Any) -> bool:
    """The validator's comparison: ``jax.tree.all(jax.tree.map(_values_close, ...))``."""

    import jax

    from ..jax._trace_graph_helpers import _values_close

    try:
        return bool(jax.tree.all(jax.tree.map(_values_close, actual, expected)))
    except ValueError:  # differing tree structure
        return False


# --- recipes ------------------------------------------------------------------


def recipe_or_reason(trace: Any, op: Any) -> OpRecipe | str:
    """The recipe for ``op``, or why it has none."""

    try:
        return _build_recipe(trace, op)
    except (_Unencodable, ValueError) as exc:
        return str(exc)


def recipe(trace: Any, op: Any) -> OpRecipe | None:
    """Data-only description of one equation op, or ``None`` when it has none."""

    found = recipe_or_reason(trace, op)
    return None if isinstance(found, str) else found


def _op_identity(capture: Any) -> str:
    kind = capture.kind
    identity = f"jax.primitive.{capture.primitive}" if kind == "primitive" else f"jax.{kind}"
    if identity not in ALLOWLIST:
        raise _Unencodable(f"op identity {identity!r} is outside the closed allowlist")
    return identity


def _build_recipe(trace: Any, op: Any) -> OpRecipe:
    import jax

    plan = _plan_or_reason(trace, op)
    if isinstance(plan, str):
        raise ValueError(plan)
    if plan.kind != "equation":
        raise ValueError(f"{plan.kind} op: its payload is loaded, not replayed")
    capture = plan.capture
    identity = _op_identity(capture)
    parent_positions = set(plan.positions)
    literals = tuple(
        ((position,), _encode(value))
        for position, value in enumerate(capture.input_values)
        if position not in parent_positions
    )
    return OpRecipe(
        backend=BACKEND,
        op_identity=identity,
        inputs=step(trace, plan.op).inputs,
        literals=literals,
        attrs=_capture_attrs(capture, plan),
        output_slots=(_parent_label(plan.op),),
        param_refs=_refs(plan, _param_address),
        constant_refs=_refs(plan, _constant_label),
        rng=None,
        framework_version=str(jax.__version__),
    )


_TUPLE_FIELDS = ("source_path", "invars", "outvars", "input_avals", "output_avals", "module_stack")


def _capture_attrs(capture: Any, plan: _Plan) -> dict[str, Any]:
    return {
        "kind": capture.kind,
        "primitive": capture.primitive,
        "index": int(capture.index),
        "params": {_MAP_TAG: {str(k): _encode(v) for k, v in capture.params.items()}},
        "num_inputs": len(capture.input_values),
        "num_outputs": len(capture.output_values),
        "element_index": tuple(-1 if e is None else e for e in plan.element_index),
        "inlined": bool(capture.inlined),
        "control_capture_indices": tuple(capture.control_capture_indices),
        "module_call_stack": tuple((str(a), int(n)) for a, n in capture.module_call_stack),
        **{name: tuple(getattr(capture, name)) for name in _TUPLE_FIELDS},
    }


def _param_address(parent: Any) -> str | None:
    annotations = getattr(parent, "annotations", None) or {}
    path = annotations.get("jax_container_path") if isinstance(annotations, Mapping) else None
    return path if isinstance(path, str) and path.startswith("params") else None


def _constant_label(parent: Any) -> str | None:
    return _parent_label(parent) if getattr(parent, "layer_type", None) == "constant" else None


def _refs(plan: _Plan, read: Any) -> tuple[tuple[tuple[str | int, ...], str], ...]:
    pairs = zip(plan.positions, plan.parents, strict=True)
    return tuple(((position,), ref) for position, parent in pairs if (ref := read(parent)))


# --- value codec (essential type switch over JAX param kinds) -------------------


def _encode(value: Any) -> Any:
    """Encode one capture value as recipe data, or raise :class:`_Unencodable`.

    One branch per JAX param kind (an essential type switch).
    """

    import jax
    import jax.extend.core
    import numpy as np

    if value is None or type(value) in (bool, int, float, complex, str):
        return value
    if type(value) is tuple or type(value) is list:
        encoded = [_encode(v) for v in value]
        return tuple(encoded) if type(value) is tuple else encoded
    if isinstance(value, np.dtype):
        return {_TAG: ("dtype", _dtype_name(value))}
    if isinstance(value, np.generic | np.ndarray | jax.Array):
        return _encode_array(value)
    if isinstance(value, enum.Enum):
        return {_TAG: ("enum", _lax_type_name(value, _ENUM_TYPE_NAMES), value.name)}
    if isinstance(value, jax.extend.core.ClosedJaxpr):
        return {_TAG: ("closed_jaxpr", _encode_closed_jaxpr(value))}
    if isinstance(value, Mapping) and all(type(k) is str for k in value):
        return {_MAP_TAG: {k: _encode(v) for k, v in value.items()}}
    name = _lax_type_name(value, _STRUCT_TYPE_NAMES)
    return {_TAG: ("struct", name, {f: _encode(v) for f, v in _struct_fields(value)})}


def _dtype_name(dtype: Any) -> str:
    import jax

    if jax.dtypes.issubdtype(dtype, jax.dtypes.extended):
        raise _Unencodable(f"extended dtype {dtype} (PRNG key data) has no data encoding")
    return str(dtype)


def _encode_array(value: Any) -> Any:
    import numpy as np

    kind = "np_scalar" if isinstance(value, np.generic) else "array"
    host = np.asarray(value)
    name = _dtype_name(host.dtype)
    return {_TAG: (kind, name, tuple(host.shape), tuple(host.ravel().tolist()))}


def _lax_type_name(value: Any, names: Sequence[str]) -> str:
    import jax

    for name in names:
        if type(value) is getattr(jax.lax, name, None):
            return name
    raise _Unencodable(f"{type(value).__module__}.{type(value).__qualname__} has no data encoding")


def _struct_fields(value: Any) -> tuple[tuple[str, Any], ...]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple((f.name, getattr(value, f.name)) for f in dataclasses.fields(value))
    return tuple(zip(type(value)._fields, value, strict=True))


def _aval(var: Any) -> tuple[Any, ...]:
    aval = var.aval
    return (tuple(aval.shape), _dtype_name(aval.dtype), bool(getattr(aval, "weak_type", False)))


def _encode_closed_jaxpr(closed: Any) -> dict[str, Any]:
    """Nested program as data: avals, consts, equations (params recursively)."""

    jaxpr = closed.jaxpr
    if jaxpr.effects:
        raise _Unencodable("nested program has effects")
    var_ids: dict[Any, int] = {}

    def define(var: Any) -> tuple[Any, ...]:
        var_ids[var] = len(var_ids)
        return _aval(var)

    def ref(atom: Any) -> Any:
        if atom in var_ids:
            return ("var", var_ids[atom])
        if hasattr(atom, "val"):
            return ("lit", _encode(atom.val))
        raise _Unencodable("nested program reads an undefined variable")

    constvars = tuple(define(v) for v in jaxpr.constvars)
    invars = tuple(define(v) for v in jaxpr.invars)
    eqns = []
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if f"jax.primitive.{name}" not in ALLOWLIST:
            raise _Unencodable(f"nested primitive {name!r} is outside the closed allowlist")
        reads = tuple(ref(a) for a in eqn.invars)
        params = {_MAP_TAG: {str(k): _encode(v) for k, v in eqn.params.items()}}
        eqns.append((name, reads, params, tuple(define(v) for v in eqn.outvars)))
    return {
        "constvars": constvars,
        "consts": tuple(_encode(c) for c in closed.consts),
        "invars": invars,
        "eqns": tuple(eqns),
        "outvars": tuple(ref(a) for a in jaxpr.outvars),
    }


def _decode(node: Any) -> Any:
    """Inverse of :func:`_encode` (lists from a JSON round trip become tuples)."""

    import jax.numpy as jnp
    import numpy as np

    if node is None or type(node) in (bool, int, float, complex, str):
        return node
    if isinstance(node, tuple | list):
        return tuple(_decode(v) for v in node)
    if isinstance(node, Mapping) and set(node) == {_MAP_TAG}:
        return {str(k): _decode(v) for k, v in node[_MAP_TAG].items()}
    if not (isinstance(node, Mapping) and set(node) == {_TAG}):
        raise ReplayRefusal("type", f"bad JAX recipe value {type(node).__name__}")
    kind, *body = node[_TAG]
    if kind == "dtype":
        return jnp.dtype(body[0])
    if kind in ("np_scalar", "array"):
        array = np.asarray(body[2], dtype=jnp.dtype(body[0])).reshape(tuple(body[1]))
        return array.reshape(())[()] if kind == "np_scalar" else array
    return _decode_typed(kind, body)


def _decode_typed(kind: str, body: Sequence[Any]) -> Any:
    import jax

    if kind == "enum" and body[0] in _ENUM_TYPE_NAMES:
        return getattr(jax.lax, body[0])[body[1]]
    if kind == "struct" and body[0] in _STRUCT_TYPE_NAMES:
        return getattr(jax.lax, body[0])(**{str(k): _decode(v) for k, v in body[1].items()})
    if kind == "closed_jaxpr":
        return _decode_closed_jaxpr(body[0])
    raise ReplayRefusal("type", f"unknown JAX recipe value kind {kind!r}")


def _decode_closed_jaxpr(data: Mapping[str, Any]) -> Any:
    """Rebuild a nested program by binding its equations under ``jax.make_jaxpr``."""

    import jax

    consts = tuple(_decode(c) for c in data["consts"])

    def program(*args: Any) -> tuple[Any, ...]:
        env: dict[int, Any] = dict(enumerate((*consts, *args)))

        def read(ref: Sequence[Any]) -> Any:
            return env[int(ref[1])] if ref[0] == "var" else _decode(ref[1])

        for name, reads, params, outs in data["eqns"]:
            primitive = _resolve_primitive(str(name))
            result = primitive.bind(*(read(r) for r in reads), **_decode(params))
            results = result if primitive.multiple_results else (result,)
            for value in results:
                env[len(env)] = value
            if len(results) != len(outs):
                raise ReplayRefusal("slots", f"nested {name} output count changed")
        return tuple(read(r) for r in data["outvars"])

    specs = [
        jax.ShapeDtypeStruct(tuple(shape), jax.numpy.dtype(dtype), weak_type=bool(weak))
        for shape, dtype, weak in data["invars"]
    ]
    return jax.make_jaxpr(program)(*specs)


_PRIMITIVES: dict[str, Any] = {}


def _resolve_primitive(name: str) -> Any:
    """Allowlisted primitive name to the framework's primitive object."""

    if f"jax.primitive.{name}" not in ALLOWLIST:
        raise ReplayRefusal("unknown_op", f"JAX primitive {name!r} is not allowlisted")
    if not _PRIMITIVES:
        import jax
        import jax.extend.core
        import jax.extend.core.primitives as public

        for module in (public, jax.lax):
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, jax.extend.core.Primitive):
                    _PRIMITIVES.setdefault(obj.name, obj)
    try:
        return _PRIMITIVES[name]
    except KeyError:
        raise ReplayRefusal(
            "unknown_op", f"JAX primitive {name!r} is not exposed by this JAX version"
        ) from None


# --- binding recipes back onto a trace ------------------------------------------


def bind_recipes(trace: Any, recipes: Mapping[str, OpRecipe]) -> None:
    """Rebuild ``trace.jax_equation_captures`` from recipes plus loaded payloads.

    Refuses (``ReplayRefusal``) a missing or foreign recipe, an identity
    outside :data:`ALLOWLIST`, and a trace whose region ops have no region
    captures (regions are not rebuilt, and an empty region list would let the
    region check pass vacuously).
    """

    from ..jax.backend import _jax_equation_ops, _jax_ops_by_label, _jax_region_ops

    if _jax_region_ops(trace) and not getattr(trace, "jax_region_captures", ()):
        raise ReplayRefusal("unbindable", "JAX region ops cannot be rebuilt from recipes")
    ops_by_label = _jax_ops_by_label(trace)
    hidden = {
        label: value if isinstance(value, tuple) else (value,)
        for label, value in getattr(trace, "_selective_save_hidden_payloads", {}).items()
    }
    captures = tuple(
        _capture_from_recipe(_recipe_for(recipes, op), op, ops_by_label, hidden)
        for op in _jax_equation_ops(trace)
    )
    regions = tuple(getattr(trace, "jax_region_captures", ()) or ())
    trace.jax_equation_captures = captures
    trace.jax_ordered_captures = tuple(sorted((*captures, *regions), key=lambda c: c.index))
    _INDEX_CACHE.pop(trace, None)


def _recipe_for(recipes: Mapping[str, OpRecipe], op: Any) -> OpRecipe:
    label = _parent_label(op)
    found = recipes.get(label)
    if found is None:
        raise ReplayRefusal("slots", "no recipe for this JAX equation op", op_label=label)
    if found.backend != BACKEND or found.op_identity not in ALLOWLIST:
        raise ReplayRefusal("unknown_op", f"{found.op_identity!r} not allowed", op_label=label)
    if found.output_slots != (label,) or found.attrs.get("kind") != _kind_annotation(op):
        raise ReplayRefusal("slots", "recipe does not describe this op", op_label=label)
    return found


def _capture_from_recipe(
    rec: OpRecipe, op: Any, ops_by_label: Mapping[str, Any], hidden: Mapping[str, Any]
) -> Any:
    from ..jax._trace_graph_helpers import _saved_single_output
    from ..jax.jaxpr import JaxEquationCapture

    attrs = rec.attrs
    label = rec.output_slots[0]
    inputs: list[Any] = [_UNSET] * int(attrs["num_inputs"])
    for path, value in rec.literals:
        inputs[_position(path, len(inputs), label)] = _decode(value)
    for occurrence, element in zip(rec.inputs, attrs["element_index"], strict=True):
        parent = ops_by_label.get(occurrence.parent_label)
        if parent is None:
            raise ReplayRefusal(
                "slots", f"unknown parent {occurrence.parent_label}", op_label=label
            )
        payload = _saved_single_output(parent, hidden)
        position = _position(occurrence.arg_path, len(inputs), label)
        inputs[position] = payload[element] if element >= 0 else payload
    if any(value is _UNSET for value in inputs):
        raise ReplayRefusal("slots", "recipe leaves a primitive input unset", op_label=label)
    output = _saved_single_output(op, hidden)
    outputs = (output,) if int(attrs["num_outputs"]) == 1 else tuple(output)
    kind = str(attrs["kind"])
    return JaxEquationCapture(
        index=int(attrs["index"]),
        kind=kind,  # type: ignore[arg-type]
        primitive=str(attrs["primitive"]),
        primitive_obj=_resolve_primitive(str(attrs["primitive"])) if kind == "primitive" else None,
        input_values=tuple(inputs),
        output_values=outputs,
        params=_decode(attrs["params"]),
        inlined=bool(attrs["inlined"]),
        control_capture_indices=tuple(int(i) for i in attrs["control_capture_indices"]),
        module_call_stack=tuple((str(a), int(n)) for a, n in attrs["module_call_stack"]),
        **{name: tuple(attrs[name]) for name in _TUPLE_FIELDS},
    )


_UNSET: Any = object()


def _position(path: Sequence[Any], count: int, label: str) -> int:
    if len(path) != 1 or type(path[0]) is not int or not 0 <= path[0] < count:
        raise ReplayRefusal("shape", f"bad JAX argument path {tuple(path)!r}", op_label=label)
    return int(path[0])
