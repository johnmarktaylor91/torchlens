"""Meaningful runtime-mode detection and train/eval divergence classification."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
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
    if isinstance(current, np.ndarray):
        return current
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
    return repr(left) == repr(right)


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

    if captured_outputs is not None and {"train", "eval"}.issubset(captured_outputs):
        divergence = classify_train_eval_divergence(
            captured_outputs["train"], captured_outputs["eval"]
        )
        if divergence.classification != "none":
            return (RunMode.TRAIN, RunMode.EVAL)
    if _has_mode_sensitive_child(model) or _forward_reads_training(model):
        return (RunMode.TRAIN, RunMode.EVAL)
    return (RunMode.EVAL,)
