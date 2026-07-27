"""Meaningful runtime-mode detection and train/eval divergence classification."""

from __future__ import annotations

import hashlib
import inspect
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from menagerie.crawler.constants import RunMode


@dataclass(frozen=True)
class DivergenceResult:
    """Train/eval divergence classification and concise evidence.

    Parameters
    ----------
    classification:
        ``none``, ``statistical``, or ``structural``.
    evidence:
        Mechanical explanation based on output or graph signatures.
    """

    classification: str
    evidence: str


def _shape(value: object) -> Optional[list[int]]:
    """Return a concrete tensor/array shape when exposed.

    Parameters
    ----------
    value:
        Candidate output leaf.

    Returns
    -------
    list[int] | None
        Concrete shape or ``None`` for a non-array leaf.
    """

    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dimension) for dimension in shape]
    except (TypeError, ValueError):
        return None


def _dtype(value: object) -> Optional[str]:
    """Return a normalized tensor/array dtype when exposed.

    Parameters
    ----------
    value:
        Candidate output leaf.

    Returns
    -------
    str | None
        Dtype text or ``None``.
    """

    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def output_signature(value: object, path: str = "output") -> dict[str, Any]:
    """Describe an output pytree without retaining tensor payloads.

    Parameters
    ----------
    value:
        Arbitrary nested output.
    path:
        Root path used in leaf locations.

    Returns
    -------
    dict[str, Any]
        Deterministic tree and leaf signature.
    """

    leaves: list[dict[str, Any]] = []

    def visit(item: object, item_path: str) -> Any:
        """Visit one output node and append leaf metadata.

        Parameters
        ----------
        item:
            Current output node.
        item_path:
            Stable pytree path.

        Returns
        -------
        Any
            JSON-compatible tree descriptor.
        """

        if isinstance(item, Mapping):
            keys = sorted(item, key=str)
            return {str(key): visit(item[key], f"{item_path}.{key}") for key in keys}
        if isinstance(item, tuple):
            return {
                "tuple": [visit(child, f"{item_path}[{index}]") for index, child in enumerate(item)]
            }
        if isinstance(item, list):
            return {
                "list": [visit(child, f"{item_path}[{index}]") for index, child in enumerate(item)]
            }
        leaf = {
            "path": item_path,
            "kind": "tensor" if _shape(item) is not None else "python",
            "shape": _shape(item),
            "dtype": _dtype(item),
            "device": str(getattr(item, "device", "")) or None,
            "python_type": f"{type(item).__module__}.{type(item).__qualname__}",
        }
        leaves.append(leaf)
        return {"leaf": len(leaves) - 1}

    tree = visit(value, path)
    return {"tree": tree, "leaves": leaves}


def _to_numpy(value: object) -> Optional[np.ndarray[Any, Any]]:
    """Convert a tensor-like leaf to a detached CPU NumPy array.

    Parameters
    ----------
    value:
        Tensor-like leaf.

    Returns
    -------
    numpy.ndarray | None
        Array view/copy, or ``None`` for a non-array value.
    """

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        try:
            return np.asarray(numpy_method())
        except (TypeError, ValueError, RuntimeError):
            return None
    if isinstance(current, (np.ndarray, np.generic)):
        return np.asarray(current)
    return None


def _flatten(value: object) -> list[object]:
    """Flatten an output pytree in signature order.

    Parameters
    ----------
    value:
        Arbitrary nested output.

    Returns
    -------
    list[object]
        Ordered leaf values.
    """

    if isinstance(value, Mapping):
        return [leaf for key in sorted(value, key=str) for leaf in _flatten(value[key])]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


class _UnstableOutputValue(TypeError):
    """Raised internally when an output has no process-independent encoding."""


def _frame(tag: bytes, payload: bytes) -> bytes:
    """Frame one stable output value without ambiguous concatenation.

    Parameters
    ----------
    tag:
        Closed semantic type tag.
    payload:
        Stable value bytes.

    Returns
    -------
    bytes
        Length-delimited type and payload bytes.
    """

    return struct.pack(">I", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload


def _type_name(value: object) -> str:
    """Return the stable qualified Python type name for a structured value.

    Parameters
    ----------
    value:
        Value whose type participates in an encoding.

    Returns
    -------
    str
        Module-qualified type name.
    """

    return f"{type(value).__module__}.{type(value).__qualname__}"


def _stable_array_bytes(array: np.ndarray[Any, Any]) -> bytes:
    """Encode a non-object NumPy array in canonical little-endian C order.

    Parameters
    ----------
    array:
        Array converted from a supported tensor or supplied directly.

    Returns
    -------
    bytes
        Versioned dtype, shape, and exact value bytes.

    Raises
    ------
    _UnstableOutputValue
        If the array contains Python objects.
    """

    if array.dtype.hasobject:
        raise _UnstableOutputValue("object arrays have no stable output encoding")
    contiguous = np.ascontiguousarray(array)
    byteorder = contiguous.dtype.byteorder
    if byteorder == ">" or (byteorder == "=" and not np.little_endian):
        contiguous = contiguous.byteswap().view(contiguous.dtype.newbyteorder("<"))
    elif byteorder == "=":
        contiguous = contiguous.view(contiguous.dtype.newbyteorder("<"))
    dtype = contiguous.dtype.str.encode("ascii")
    shape = b"".join(struct.pack(">q", int(dimension)) for dimension in contiguous.shape)
    payload = (
        _frame(b"dtype", dtype)
        + _frame(b"rank", struct.pack(">I", contiguous.ndim))
        + _frame(b"shape", shape)
        + _frame(b"data", contiguous.tobytes(order="C"))
    )
    return _frame(b"array-v2", payload)


def _stable_output_bytes(value: object, active: set[int]) -> bytes:
    """Recursively encode one explicitly supported output value.

    Parameters
    ----------
    value:
        Output value or nested child.
    active:
        Object identities on the active recursion path.

    Returns
    -------
    bytes
        Process-independent type-tagged bytes.

    Raises
    ------
    _UnstableOutputValue
        If a leaf is unsupported, cyclic, or otherwise unstable.
    """

    array = _to_numpy(value)
    if array is not None:
        return _stable_array_bytes(array)
    if value is None:
        return _frame(b"none", b"")
    if isinstance(value, bool):
        return _frame(b"bool", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame(b"int", str(value).encode("ascii"))
    if isinstance(value, float):
        return _frame(b"float64", struct.pack(">d", value))
    if isinstance(value, complex):
        return _frame(b"complex128", struct.pack(">dd", value.real, value.imag))
    if isinstance(value, str):
        return _frame(b"utf8", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _frame(b"bytes", value)

    identity = id(value)
    if identity in active:
        raise _UnstableOutputValue("cyclic outputs have no stable output encoding")
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            encoded_fields = b"".join(
                _frame(
                    field.name.encode("utf-8"),
                    _stable_output_bytes(getattr(value, field.name), active),
                )
                for field in fields(value)
            )
            return _frame(
                b"dataclass-v1",
                _frame(b"type", _type_name(value).encode("utf-8"))
                + _frame(b"fields", encoded_fields),
            )
        named_fields = getattr(type(value), "_fields", None)
        if (
            isinstance(value, tuple)
            and isinstance(named_fields, tuple)
            and all(isinstance(name, str) for name in named_fields)
        ):
            encoded_fields = b"".join(
                _frame(name.encode("utf-8"), _stable_output_bytes(getattr(value, name), active))
                for name in named_fields
            )
            return _frame(
                b"namedtuple-v1",
                _frame(b"type", _type_name(value).encode("utf-8"))
                + _frame(b"fields", encoded_fields),
            )
        if isinstance(value, Mapping):
            encoded_items: list[tuple[bytes, bytes]] = []
            for key, child in value.items():
                encoded_key = _stable_output_bytes(key, active)
                encoded_items.append((encoded_key, _stable_output_bytes(child, active)))
            encoded_items.sort(key=lambda item: item[0])
            if any(
                encoded_items[index - 1][0] == encoded_items[index][0]
                for index in range(1, len(encoded_items))
            ):
                raise _UnstableOutputValue("mapping keys collide under stable output encoding")
            payload = b"".join(
                _frame(b"key", key) + _frame(b"value", child) for key, child in encoded_items
            )
            return _frame(b"mapping-v1", payload)
        if isinstance(value, tuple):
            return _frame(
                b"tuple-v1",
                b"".join(_frame(b"item", _stable_output_bytes(child, active)) for child in value),
            )
        if isinstance(value, list):
            return _frame(
                b"list-v1",
                b"".join(_frame(b"item", _stable_output_bytes(child, active)) for child in value),
            )
    finally:
        active.remove(identity)
    raise _UnstableOutputValue(f"unsupported output leaf type: {_type_name(value)}")


def output_value_sha256(value: object) -> Optional[str]:
    """Hash exact output values using a versioned stable encoder.

    Parameters
    ----------
    value:
        Arbitrary nested output whose structure is separately captured by
        :func:`output_signature`.

    Returns
    -------
    str | None
        Domain-separated SHA-256, or ``None`` when any leaf lacks an explicitly
        supported process-independent representation.
    """

    try:
        encoded = _stable_output_bytes(value, set())
    except _UnstableOutputValue:
        return None
    digest = hashlib.sha256(b"menagerie-output-values-v2\0")
    digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def classify_observed_mode_receipts(
    train_receipt: Mapping[str, object], eval_receipt: Mapping[str, object]
) -> Optional[DivergenceResult]:
    """Classify independently captured train/eval receipt observations.

    Parameters
    ----------
    train_receipt, eval_receipt:
        Per-mode worker receipts carrying output structure and optional value digests.

    Returns
    -------
    DivergenceResult | None
        Mechanical classification, or ``None`` when equal structures lack the value
        digests required to distinguish ``none`` from ``statistical``.
    """

    train_signature = train_receipt.get("output_signature")
    eval_signature = eval_receipt.get("output_signature")
    if train_signature != eval_signature:
        return DivergenceResult(
            "structural", "train and eval output trees, types, dtypes, or shapes differ"
        )
    train_values = train_receipt.get("output_value_sha256")
    eval_values = eval_receipt.get("output_value_sha256")
    if not isinstance(train_values, str) or not isinstance(eval_values, str):
        return None
    if train_values == eval_values:
        return DivergenceResult("none", "train and eval output value digests match")
    return DivergenceResult(
        "statistical", "train and eval structures match but output value digests differ"
    )


def _leaf_equal(left: object, right: object) -> bool:
    """Compare two structurally equivalent output leaves.

    Parameters
    ----------
    left, right:
        Output leaf values.

    Returns
    -------
    bool
        True only when values are exactly equal, treating paired NaNs as equal.
    """

    left_array = _to_numpy(left)
    right_array = _to_numpy(right)
    if left_array is not None and right_array is not None:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    try:
        equality = left == right
        if isinstance(equality, (bool, np.bool_)):
            return bool(equality)
    except (TypeError, ValueError, RuntimeError):
        pass
    left_digest = output_value_sha256(left)
    right_digest = output_value_sha256(right)
    return left_digest is not None and left_digest == right_digest


def classify_train_eval_divergence(
    train_output: object,
    eval_output: object,
    *,
    train_graph_signature: Optional[object] = None,
    eval_graph_signature: Optional[object] = None,
) -> DivergenceResult:
    """Classify train/eval behavior from captured outputs and optional graphs.

    Parameters
    ----------
    train_output, eval_output:
        Captured per-mode outputs.
    train_graph_signature, eval_graph_signature:
        Optional operation-graph signatures from a framework-native observer.

    Returns
    -------
    DivergenceResult
        ``structural`` for tree/shape/graph changes, ``statistical`` for value-only
        changes, and ``none`` for equal outputs.
    """

    train_signature = output_signature(train_output)
    eval_signature = output_signature(eval_output)
    if train_signature != eval_signature:
        return DivergenceResult(
            "structural", "train and eval output trees, types, dtypes, or shapes differ"
        )
    if (
        train_graph_signature is not None
        and eval_graph_signature is not None
        and train_graph_signature != eval_graph_signature
    ):
        return DivergenceResult("structural", "train and eval operation graphs differ")
    train_leaves = _flatten(train_output)
    eval_leaves = _flatten(eval_output)
    if all(_leaf_equal(left, right) for left, right in zip(train_leaves, eval_leaves)):
        return DivergenceResult("none", "train and eval outputs are equal")
    return DivergenceResult(
        "statistical", "train and eval output signatures match but values differ"
    )


def _has_mode_sensitive_child(model: object) -> bool:
    """Detect known normalization/dropout mode-sensitive children.

    Parameters
    ----------
    model:
        Native model object.

    Returns
    -------
    bool
        True when a child type names BatchNorm or Dropout.
    """

    modules_method = getattr(model, "modules", None)
    if not callable(modules_method):
        return False
    try:
        modules: Sequence[object] = tuple(modules_method())
    except (TypeError, RuntimeError):
        return False
    return any(
        "batchnorm" in type(module).__name__.lower() or "dropout" in type(module).__name__.lower()
        for module in modules
    )


def _forward_reads_training(model: object) -> bool:
    """Detect an explicit ``training`` branch in Python forward source.

    Parameters
    ----------
    model:
        Native model object.

    Returns
    -------
    bool
        True when inspectable forward source references training mode.
    """

    forward = getattr(model, "forward", None)
    if not callable(forward):
        return False
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return False
    return ".training" in source or "training=" in source or "training =" in source


def detect_meaningful_modes(
    model: object, *, captured_outputs: Optional[Mapping[str, object]] = None
) -> tuple[RunMode, ...]:
    """Detect the minimal meaningful train/eval mode set.

    Models with BatchNorm, Dropout, an inspectable training branch, or already
    observed train/eval divergence require both modes. A model confidently lacking
    those features uses one eval-mode representative.

    Parameters
    ----------
    model:
        Native model or transparent adapter.
    captured_outputs:
        Optional train/eval outputs from a probe.

    Returns
    -------
    tuple[RunMode, ...]
        Meaningful modes in train/eval order.
    """

    # This heuristic cannot see every native/custom branch; author declarations and checker vetting cover it.
    if captured_outputs is not None and {"train", "eval"}.issubset(captured_outputs):
        divergence = classify_train_eval_divergence(
            captured_outputs["train"], captured_outputs["eval"]
        )
        if divergence.classification != "none":
            return (RunMode.TRAIN, RunMode.EVAL)
    if _has_mode_sensitive_child(model) or _forward_reads_training(model):
        return (RunMode.TRAIN, RunMode.EVAL)
    return (RunMode.EVAL,)
