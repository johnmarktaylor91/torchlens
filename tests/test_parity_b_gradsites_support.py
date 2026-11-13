"""Shared harness for the per-backend ``tl.grad`` site proofs (no tests here).

Every backend traces the same program, ``loss = sum(s * s + s * x)`` with
``s = x * x + x``, and edits the gradient of ``s``. The neutral module's
result is compared with the framework's own gradient where the same edit is
applied by hand between the two halves of the program (each backend test
computes that oracle with its framework's derivative API).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from torchlens.backends._replay_adapters import get_adapter
from torchlens.intervention._grad_site_neutral import (
    GradSiteEdit,
    GradSitePass,
    GradSiteResult,
    edited_gradients_for_backend,
)

X_VALUES = np.array([0.5, -1.25, 2.0, 0.3], dtype=np.float32)
"""Inputs where the clip edit below is active on some elements and not others."""

SCALE = 0.25
CLIP = 0.5


def x_np() -> np.ndarray:
    """A fresh copy of the input."""

    return X_VALUES.copy()


def s_np() -> np.ndarray:
    """The site value ``s = x * x + x`` in NumPy."""

    x = X_VALUES.astype(np.float64)
    return x * x + x


@dataclass(frozen=True)
class NeutralRun:
    """The neutral module's result on one trace, with the labels the tests read."""

    result: GradSiteResult
    site: str
    input_label: str
    adapter: Any

    def host(self, label: str) -> np.ndarray:
        """The gradient of ``label`` as a NumPy array."""

        return np.asarray(self.adapter.to_host(self.result.grads[label]), dtype=np.float64)


def _labels(op: Any) -> tuple[str, ...]:
    """Every label spelling a recipe may use to name ``op``'s output."""

    names = (getattr(op, "_label_raw", None), getattr(op, "label", None))
    return tuple(str(name) for name in names if name)


def _replay_values(
    trace: Any, adapter: Any, x_value: Any
) -> tuple[dict[str, Any], list[Any], tuple[str, ...]]:
    """Values of every label from the input through ``adapter.forward``; the graph ops.

    The input is the op with no recipe (JAX, TF, Paddle, tinygrad) or an
    ``"input"`` recipe (MLX); it is left out of the graph ops, so it is a leaf.
    """

    values: dict[str, Any] = {}
    graph_ops: list[Any] = []
    sources: list[tuple[str, ...]] = []
    for op in trace._layer_list:
        recipe = adapter.recipe(trace, op)
        if recipe is None or recipe.op_identity == "input":
            sources.append(_labels(op))
            values.update(dict.fromkeys(_labels(op), x_value))
            continue
        graph_ops.append(op)
        if recipe.output_slots[0] in values:
            continue
        outputs = adapter.forward(trace, op, [values[p] for p in recipe.parent_labels])
        values.update(zip(recipe.output_slots, outputs, strict=True))
    assert len(sources) == 1, f"expected one source (the input), got {sources}"
    return values, graph_ops, sources[0]


def _site_label(adapter: Any, values: dict[str, Any], graph_ops: Sequence[Any], trace: Any) -> str:
    """The one graph label whose value is ``s`` (the site)."""

    expected = s_np()
    matches = []
    for op in graph_ops:
        for slot in adapter.recipe(trace, op).output_slots:
            host = np.asarray(adapter.to_host(values[slot]), dtype=np.float64)
            if host.shape == expected.shape and np.allclose(host, expected, rtol=1e-5):
                matches.append(slot)
    assert len(set(matches)) == 1, f"site value found at {matches}"
    return matches[0]


def run_neutral(
    trace: Any,
    backend: str,
    x_value: Any,
    edits: Sequence[tuple[str, Callable[[Any], Any]]],
    ones_like: Callable[[Any], Any],
) -> NeutralRun:
    """Run the neutral module on ``trace`` with ``edits`` (name, fn) on the gradient of ``s``."""

    adapter = get_adapter(backend)
    values, graph_ops, source = _replay_values(trace, adapter, x_value)
    site = _site_label(adapter, values, graph_ops, trace)
    loss = adapter.recipe(trace, graph_ops[-1]).output_slots[0]
    armed = [GradSiteEdit(site, lambda grad, call, fn=fn: fn(grad), name) for name, fn in edits]
    result = edited_gradients_for_backend(
        trace,
        backend,
        graph_ops,
        values.__getitem__,
        GradSitePass({loss: ones_like(values[loss])}, armed),
    )
    input_label = next(label for label in source if label in result.grads)
    return NeutralRun(result=result, site=site, input_label=input_label, adapter=adapter)


def expected_site_grad(edited: bool) -> np.ndarray:
    """``d loss / d s = 2 s + x``, scaled then clipped when ``edited``."""

    grad = 2 * s_np() + X_VALUES.astype(np.float64)
    return np.clip(SCALE * grad, -CLIP, CLIP) if edited else grad


def assert_matches_oracle(run: NeutralRun, oracle_x: Any, *, edited: bool) -> None:
    """Input gradient equals the framework oracle; the site holds the (edited) gradient."""

    got = run.host(run.input_label)
    want = np.asarray(oracle_x, dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(run.host(run.site), expected_site_grad(edited), rtol=1e-5, atol=1e-6)
    assert [fire.label for fire in run.result.fires] == ([run.site] * 2 if edited else [])
