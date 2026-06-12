"""Tensor kwargs must be extracted as op parents (bug ARG-KWARGS-MISSING).

``extract_tensors_and_params()`` consults static ``ArgSpec`` table entries
whose ``tensor_kwargs`` tuples were empty (or wrongly named) for several
common functions, so tensors passed as KEYWORD arguments were silently
dropped from parent/parameter extraction. Symptoms: ``torch.bmm(a, mat2=b)``
recorded only ``a`` as a parent; ``torch.normal(mean=m, std=s)`` recorded no
parents at all and was misclassified as an internal source.

Scope: full static-table audit for tensor-bearing kwargs, including linear,
cat/stack, where, normal, conv/norm/loss, scatter/gather/index, attention,
factory-from-source, matmul/mm/bmm/mv, and addmm-style ternary functions.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.capture.arg_positions import (
    FUNC_ARG_SPECS,
    _normalize_func_name,
    extract_tensors_and_params,
)

# ---------------------------------------------------------------------------
# Unit level: the static table extracts kwarg tensors for each function
# ---------------------------------------------------------------------------

_A = torch.randn(2, 3)
_B = torch.randn(2, 3)

_KWARG_CASES = [
    # (func_name, args, kwargs) — every tensor in kwargs must be found.
    ("linear", (torch.randn(2, 3),), {"weight": torch.randn(4, 3), "bias": torch.randn(4)}),
    ("conv1d", (torch.randn(1, 2, 8),), {"weight": torch.randn(3, 2, 3), "bias": torch.randn(3)}),
    ("conv2d", (torch.randn(1, 2, 8, 8),), {"weight": torch.randn(3, 2, 3, 3)}),
    ("conv3d", (torch.randn(1, 2, 4, 4, 4),), {"weight": torch.randn(3, 2, 2, 2, 2)}),
    ("cat", (), {"tensors": [_A, _B]}),
    ("stack", (), {"tensors": [_A, _B]}),
    ("where", (_A > 0,), {"input": _A, "other": _B}),
    ("normal", (), {"mean": _A, "std": _B.abs() + 0.1}),
    ("matmul", (_A,), {"other": _A.t().t()}),
    ("mm", (torch.randn(2, 3),), {"mat2": torch.randn(3, 2)}),
    ("bmm", (torch.randn(2, 3, 4),), {"mat2": torch.randn(2, 4, 5)}),
    ("mv", (torch.randn(2, 3),), {"vec": torch.randn(3)}),
    (
        "addmm",
        (torch.randn(2, 2),),
        {"mat1": torch.randn(2, 3), "mat2": torch.randn(3, 2)},
    ),
    (
        "addbmm",
        (torch.randn(3, 5),),
        {"batch1": torch.randn(2, 3, 4), "batch2": torch.randn(2, 4, 5)},
    ),
    (
        "baddbmm",
        (torch.randn(2, 3, 5),),
        {"batch1": torch.randn(2, 3, 4), "batch2": torch.randn(2, 4, 5)},
    ),
    ("addmv", (torch.randn(2),), {"mat": torch.randn(2, 3), "vec": torch.randn(3)}),
    ("addcmul", (_A,), {"tensor1": _B, "tensor2": _B}),
    ("addcdiv", (_A,), {"tensor1": _B, "tensor2": _B.abs() + 1.0}),
    ("lerp", (_A,), {"end": _B, "weight": torch.tensor(0.5)}),
    ("sum", (), {"input": _A}),
    ("mseloss", (), {"input": _A, "target": _B}),
    ("crossentropy", (), {"input": torch.randn(3, 5), "target": torch.tensor([1, 2, 3])}),
    (
        "batchnorm",
        (),
        {
            "input": torch.randn(2, 3, 4),
            "running_mean": torch.randn(3),
            "running_var": torch.rand(3) + 1.0,
            "weight": torch.randn(3),
            "bias": torch.randn(3),
        },
    ),
    ("layernorm", (), {"input": _A, "weight": torch.randn(3), "bias": torch.randn(3)}),
    ("scatter", (_A, 1), {"index": torch.zeros(2, 3, dtype=torch.long), "src": _B}),
    ("scatteradd", (_A, 1), {"index": torch.zeros(2, 3, dtype=torch.long), "src": _B}),
    ("gather", (_A, 1), {"index": torch.zeros(2, 3, dtype=torch.long)}),
    (
        "indexput",
        (_A,),
        {"indices": (torch.tensor([0, 1]), torch.tensor([1, 2])), "values": torch.randn(2)},
    ),
    ("maskedscatter", (_A,), {"mask": _A > 0, "source": _B}),
    ("searchsorted", (), {"sorted_sequence": torch.arange(5), "values": torch.tensor([2, 4])}),
    ("bincount", (), {"input": torch.tensor([0, 1, 1]), "weights": torch.randn(3)}),
    ("histogram", (), {"input": _A, "weight": torch.rand_like(_A)}),
    ("stft", (_A.flatten(), 4), {"window": torch.hann_window(4)}),
    ("einsum", ("ij,jk->ik",), {"operands": (torch.randn(2, 3), torch.randn(3, 4))}),
    ("tensordot", (), {"a": torch.randn(2, 3), "b": torch.randn(3, 2)}),
    (
        "scaleddotproductattention",
        (),
        {
            "query": torch.randn(1, 1, 2, 4),
            "key": torch.randn(1, 1, 2, 4),
            "value": torch.randn(1, 1, 2, 4),
            "attn_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        },
    ),
    (
        "multiheadattentionforward",
        (),
        {
            "query": torch.randn(2, 1, 4),
            "key": torch.randn(2, 1, 4),
            "value": torch.randn(2, 1, 4),
            "in_proj_weight": torch.randn(12, 4),
            "in_proj_bias": torch.randn(12),
            "out_proj_weight": torch.randn(4, 4),
            "out_proj_bias": torch.randn(4),
            "key_padding_mask": torch.zeros(1, 2, dtype=torch.bool),
            "attn_mask": torch.zeros(2, 2),
        },
    ),
    ("zeroslike", (), {"input": _A}),
    ("fulllike", (_A,), {"fill_value": torch.tensor(2.0)}),
    ("newzeros", (_A,), {}),
    ("newfull", (_A, (2, 3)), {"fill_value": torch.tensor(2.0)}),
    ("newtensor", (_A,), {"data": _B}),
]


def _count_tensors(value: object) -> int:
    """Count tensors nested one level inside a test argument value."""

    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(_count_tensors(item) for item in value)
    return 0


@pytest.mark.parametrize(
    "func_name,args,kwargs", _KWARG_CASES, ids=[case[0] for case in _KWARG_CASES]
)
def test_static_table_extracts_kwarg_tensors(
    func_name: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    """Every tensor passed positionally OR by keyword must be extracted."""
    spec = FUNC_ARG_SPECS.get(_normalize_func_name(func_name))
    assert spec is not None, f"no static ArgSpec for {func_name!r}"

    tensors, params = extract_tensors_and_params(spec, args, kwargs)
    expected = sum(_count_tensors(a) for a in args) + sum(
        _count_tensors(v) for v in kwargs.values()
    )
    found = len(tensors) + len(params)
    assert found == expected, (
        f"{func_name}: extracted {found} tensors, expected {expected} "
        f"(spec tensor_kwargs={spec.tensor_kwargs})"
    )


def test_kwarg_parameters_routed_to_params() -> None:
    """nn.Parameter passed by keyword must land in the parameter list."""
    spec = FUNC_ARG_SPECS[_normalize_func_name("linear")]
    weight = nn.Parameter(torch.randn(4, 3))
    tensors, params = extract_tensors_and_params(spec, (torch.randn(2, 3),), {"weight": weight})
    assert len(tensors) == 1
    assert params == [weight]


# ---------------------------------------------------------------------------
# Integration level: kwarg tensors appear as parents in the trace graph
# ---------------------------------------------------------------------------


def _layer_by_func(log: tl.Trace, func_substring: str) -> Any:
    """Return the first traced layer whose label contains ``func_substring``."""

    labels = [name for name in log.layer_labels if func_substring in name]
    assert labels, f"no layer matching {func_substring!r} in {list(log.layer_labels)}"
    return log[labels[0]]


@pytest.mark.smoke
def test_traced_where_kwargs_recorded_as_parents() -> None:
    """torch.where(cond, input=a, other=b) records a and b as parents."""

    class M(nn.Module):
        def forward(self, x):
            a = x + 1
            b = x * 2
            return torch.where(x > 0, input=a, other=b)

    log = tl.trace(M(), torch.randn(2, 3))
    where_layer = _layer_by_func(log, "where")
    parents = set(where_layer.parents)
    assert any("add" in p for p in parents), parents
    assert any("mul" in p for p in parents), parents
    assert any("gt" in p for p in parents), parents


@pytest.mark.smoke
def test_traced_linear_weight_kwarg_recorded_as_parent() -> None:
    """F.linear(x, weight=w) with a derived (non-Parameter) weight records w."""

    class M(nn.Module):
        def forward(self, x):
            w = (x * 2).t()
            return F.linear(x + 1, weight=w)

    log = tl.trace(M(), torch.randn(3, 3))
    linear_layer = _layer_by_func(log, "linear")
    parents = set(linear_layer.parents)
    assert any("t_" in p for p in parents), parents
    assert any("add" in p for p in parents), parents


@pytest.mark.smoke
def test_traced_bmm_mat2_kwarg_recorded_as_parent() -> None:
    """torch.bmm(a, mat2=b) records b as a parent."""

    class M(nn.Module):
        def forward(self, x):
            a = x + 1
            b = (x * 2).transpose(1, 2)
            return torch.bmm(a, mat2=b)

    log = tl.trace(M(), torch.randn(2, 3, 4))
    bmm_layer = _layer_by_func(log, "bmm")
    parents = set(bmm_layer.parents)
    assert any("add" in p for p in parents), parents
    assert any("transpose" in p for p in parents), parents


@pytest.mark.smoke
def test_traced_addmm_kwargs_recorded_as_parents() -> None:
    """torch.addmm(bias, m1, mat2=m2) records the kwarg matrix as a parent."""

    class M(nn.Module):
        def forward(self, x):
            bias = x.sum(dim=0)
            m1 = x + 1
            m2 = x * 2
            return torch.addmm(bias, m1, mat2=m2)

    log = tl.trace(M(), torch.randn(3, 3))
    addmm_layer = _layer_by_func(log, "addmm")
    parents = set(addmm_layer.parents)
    assert any("sum" in p for p in parents), parents
    assert any("add" in p for p in parents), parents
    assert any("mul" in p for p in parents), parents


@pytest.mark.smoke
def test_traced_normal_mean_std_kwargs_recorded_as_parents() -> None:
    """torch.normal(mean=m, std=s) records both kwarg tensors as parents.

    Before the fix the op had NO parents and was misclassified as an
    internal source.
    """

    class M(nn.Module):
        def forward(self, x):
            m = x + 1
            s = (x * 2).abs() + 0.1
            return torch.normal(mean=m, std=s)

    log = tl.trace(M(), torch.randn(2, 3))
    normal_layer = _layer_by_func(log, "normal")
    parents = set(normal_layer.parents)
    assert any("add_1" in p for p in parents), parents
    assert any("add_2" in p or "abs" in p for p in parents), parents
    assert not normal_layer.is_internal_source


@pytest.mark.smoke
def test_traced_zeros_like_source_recorded_as_parent() -> None:
    """torch.zeros_like(y) records y as a shape/source parent."""

    class M(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a zeros_like tensor derived from an intermediate."""

            y = x + 1
            return torch.zeros_like(y)

    log = tl.trace(M(), torch.randn(2, 3))
    zeros_like_layer = _layer_by_func(log, "zeroslike")
    parents = set(zeros_like_layer.parents)
    assert any("add" in p for p in parents), parents
    assert not zeros_like_layer.is_internal_source


@pytest.mark.smoke
def test_traced_new_zeros_source_recorded_as_parent() -> None:
    """x.new_zeros(...) records x as the source tensor parent."""

    class M(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a new_zeros tensor derived from an intermediate source."""

            y = x + 1
            return y.new_zeros((2, 3))

    log = tl.trace(M(), torch.randn(2, 3))
    new_zeros_layer = _layer_by_func(log, "newzeros")
    parents = set(new_zeros_layer.parents)
    assert any("add" in p for p in parents), parents
    assert not new_zeros_layer.is_internal_source
