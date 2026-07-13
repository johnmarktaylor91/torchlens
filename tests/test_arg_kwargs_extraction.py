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
from torchlens import _state
from torchlens.backends.torch.ops import _extract_arg_tensors_and_params
from torchlens.capture.arg_positions import (
    FUNC_ARG_SPECS,
    _normalize_func_name,
    extract_tensors_and_params,
)
from torchlens.intervention.types import ParentRef
from torchlens.options import CaptureOptions

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
    ("gradient", (), {"input": _A, "spacing": (torch.arange(2.0),)}),
    ("quantile", (_A,), {"q": torch.tensor(0.5)}),
    ("nanquantile", (_A,), {"q": torch.tensor(0.5)}),
    (
        "marginrankingloss",
        (),
        {"input1": _A, "input2": _B, "target": torch.ones_like(_A)},
    ),
    (
        "cosineembeddingloss",
        (),
        {"input1": _A, "input2": _B, "target": torch.tensor([1.0, -1.0])},
    ),
    (
        "tripletmarginloss",
        (),
        {"anchor": _A, "positive": _B, "negative": _A + 2},
    ),
    ("clamp", (), {"input": _A, "min": _B, "max": _B}),
    ("clampmin", (), {"input": _A, "min": _B}),
    ("clampmax", (), {"input": _A, "max": _B}),
    ("clip", (), {"input": _A, "min": _B, "max": _B}),
    (
        "histogram",
        (),
        {"input": _A, "bins": torch.linspace(-2, 2, 5), "weight": torch.rand_like(_A)},
    ),
    (
        "histogramdd",
        (),
        {
            "input": torch.randn(4, 2),
            "bins": (torch.linspace(-2, 2, 5), torch.linspace(-2, 2, 5)),
            "weight": torch.rand(4),
        },
    ),
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


def test_unknown_arg_fallback_partitions_tensors_and_parameters() -> None:
    """Unknown function fallback finds tensors and parameters in one partition."""

    normalized_name = "torchlenslocalunknownfallback"
    _state._dynamic_arg_specs.pop(normalized_name, None)
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    weight = nn.Parameter(torch.randn(2, 3))

    tensors, params = _extract_arg_tensors_and_params(
        normalized_name,
        ({"left": [x, weight], "duplicate": x},),
        {"right": y, "weight": weight},
    )

    assert tensors == [y, x]
    assert params == [weight]
    assert normalized_name in _state._dynamic_arg_specs
    _state._dynamic_arg_specs.pop(normalized_name, None)


# ---------------------------------------------------------------------------
# Integration level: kwarg tensors appear as parents in the trace graph
# ---------------------------------------------------------------------------


def _layer_by_func(log: tl.Trace, func_substring: str) -> Any:
    """Return the first traced layer whose label contains ``func_substring``."""

    labels = [name for name in log.layer_labels if func_substring in name]
    assert labels, f"no layer matching {func_substring!r} in {list(log.layer_labels)}"
    return log[labels[0]]


def _template_parent_labels(value: object) -> set[str]:
    """Collect parent labels from a captured argument template.

    Parameters
    ----------
    value:
        Template value to traverse.

    Returns
    -------
    set[str]
        Parent labels referenced by the template.
    """

    if isinstance(value, ParentRef):
        return {value.parent_label}
    if isinstance(value, tuple | list):
        return set().union(*(_template_parent_labels(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return (
            set().union(*(_template_parent_labels(item) for item in value.values()))
            if value
            else set()
        )
    return set()


def _assert_parents_match_args_template(layer: Any) -> None:
    """Assert that a captured op's graph edges match its runnable template.

    Parameters
    ----------
    layer:
        Captured operation whose parent edges and template are compared.
    """

    template = layer.args_template
    assert template is not None
    template_parents = _template_parent_labels((template.args, template.kwargs))
    assert set(layer.parents) == template_parents


def _assert_child_edge(parent: Any, child: Any) -> None:
    """Assert that a parent operation exposes an edge to its consumer.

    Parameters
    ----------
    parent:
        Operation expected to provide a tensor input.
    child:
        Operation expected to consume the tensor.
    """

    child_label = str(child.label).split(":", maxsplit=1)[0]
    assert child_label in parent.children


def _trace_with_args_templates(
    model: nn.Module, inputs: torch.Tensor | list[torch.Tensor]
) -> tl.Trace:
    """Capture a model while retaining runnable argument templates.

    Parameters
    ----------
    model:
        Model to capture.
    inputs:
        Tensor input or positional tensor input sequence.

    Returns
    -------
    tl.Trace
        Trace with per-operation argument templates available for comparison.
    """

    return tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


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


@pytest.mark.smoke
def test_traced_gradient_spacing_tensor_recorded_as_parent() -> None:
    """torch.gradient records a tensor supplied through ``spacing`` as a parent."""

    class GradientWithTensorSpacing(nn.Module):
        """Compute a gradient with coordinates derived from a second input."""

        def forward(self, values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
            """Return the gradient of values at scaled coordinates.

            Parameters
            ----------
            values:
                Values to differentiate.
            coordinates:
                Coordinates used to derive tensor spacing.

            Returns
            -------
            torch.Tensor
                Numerical gradient of ``values``.
            """

            scaled_coordinates = coordinates * 2
            return torch.gradient(values, spacing=(scaled_coordinates,), dim=(0,))[0]

    trace = _trace_with_args_templates(
        GradientWithTensorSpacing(),
        [torch.tensor([0.0, 1.0, 4.0, 9.0]), torch.arange(4.0)],
    )
    gradient = next(op for op in trace.ops if op.func_name == "gradient")
    scaled_coordinates = next(op for op in trace.ops if op.func_name in {"mul", "__mul__"})
    assert any("mul" in parent for parent in gradient.parents), gradient.parents
    _assert_child_edge(scaled_coordinates, gradient)
    _assert_parents_match_args_template(gradient)


@pytest.mark.parametrize("call_style", ["positional", "keyword"])
@pytest.mark.parametrize("func_name", ["quantile", "nanquantile"])
@pytest.mark.smoke
def test_traced_quantile_tensor_q_recorded_as_parent(call_style: str, func_name: str) -> None:
    """Tensor ``q`` is a parent for positional and keyword quantile calls.

    Parameters
    ----------
    call_style:
        Whether ``q`` is supplied positionally or by keyword.
    func_name:
        Name of the quantile operation under test.
    """

    class QuantileWithTensorQ(nn.Module):
        """Select a quantile using a tensor input."""

        def forward(self, values: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            """Return the requested quantile.

            Parameters
            ----------
            values:
                Values to reduce.
            q:
                Tensor quantile to select.

            Returns
            -------
            torch.Tensor
                Selected quantile.
            """

            quantile = getattr(torch, func_name)
            if call_style == "positional":
                return quantile(values, q)
            return quantile(values, q=q)

    trace = _trace_with_args_templates(
        QuantileWithTensorQ(), [torch.tensor([0.0, 10.0, 20.0]), torch.tensor(0.25)]
    )
    quantile = next(op for op in trace.ops if op.func_name == func_name)
    inputs = [op for op in trace.ops if op.is_input]

    assert set(quantile.parents) == {op.layer_label for op in inputs}
    for input_op in inputs:
        _assert_child_edge(input_op, quantile)
    _assert_parents_match_args_template(quantile)


@pytest.mark.parametrize(
    ("func", "kwarg_names", "inputs"),
    [
        (
            F.margin_ranking_loss,
            ("input1", "input2", "target"),
            [
                torch.tensor([1.0, 2.0]),
                torch.tensor([2.0, 0.0]),
                torch.tensor([1.0, -1.0]),
            ],
        ),
        (
            F.cosine_embedding_loss,
            ("input1", "input2", "target"),
            [
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[2.0, 0.0]]),
                torch.tensor([1.0]),
            ],
        ),
        (
            F.triplet_margin_loss,
            ("anchor", "positive", "negative"),
            [
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.5, 2.5]),
                torch.tensor([4.0, 5.0]),
            ],
        ),
    ],
    ids=lambda case: case.__name__ if callable(case) else None,
)
@pytest.mark.parametrize("call_style", ["positional", "keyword"])
@pytest.mark.smoke
def test_traced_three_input_losses_record_all_tensor_parents(
    func: Any,
    kwarg_names: tuple[str, str, str],
    inputs: list[torch.Tensor],
    call_style: str,
) -> None:
    """Every tensor operand of a three-input loss is retained as a graph parent.

    Parameters
    ----------
    func:
        Loss operation under test.
    kwarg_names:
        Schema argument names for the three tensor operands.
    inputs:
        Valid loss inputs.
    call_style:
        Whether operands are supplied positionally or by keyword.
    """

    class ThreeInputLoss(nn.Module):
        """Apply one three-input functional loss."""

        def forward(self, *operands: torch.Tensor) -> torch.Tensor:
            """Return the configured loss.

            Parameters
            ----------
            operands:
                Three tensor operands for the configured loss.

            Returns
            -------
            torch.Tensor
                Scalar loss value.
            """

            if call_style == "positional":
                return func(*operands)
            return func(**dict(zip(kwarg_names, operands, strict=True)))

    trace = _trace_with_args_templates(ThreeInputLoss(), inputs)
    loss = next(op for op in trace.ops if op.func_name == func.__name__)
    input_ops = [op for op in trace.ops if op.is_input]

    assert set(loss.parents) == {op.layer_label for op in input_ops}
    for input_op in input_ops:
        _assert_child_edge(input_op, loss)
    _assert_parents_match_args_template(loss)


@pytest.mark.parametrize("bound_style", ["positional", "keyword", "mixed"])
@pytest.mark.smoke
def test_traced_clamp_tensor_bounds_recorded_as_parents(bound_style: str) -> None:
    """torch.clamp records tensor bounds in every supported calling style.

    Parameters
    ----------
    bound_style:
        Whether the tensor bounds are passed positionally, by keyword, or with
        the minimum positional and maximum by keyword.
    """

    class ClampWithTensorBounds(nn.Module):
        """Clamp a derived value between two derived tensor bounds."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Clamp a value and bounds that each depend on ``x``.

            Parameters
            ----------
            x:
                Source tensor for all three clamp inputs.

            Returns
            -------
            torch.Tensor
                Clamped tensor.
            """

            value = x + 2
            lower = x - 1
            upper = x + 1
            if bound_style == "positional":
                return torch.clamp(value, lower, upper)
            if bound_style == "keyword":
                return torch.clamp(value, min=lower, max=upper)
            return torch.clamp(value, lower, max=upper)

    trace = _trace_with_args_templates(ClampWithTensorBounds(), torch.zeros(3))
    clamp = next(op for op in trace.ops if op.func_name == "clamp")
    bound_producers = [
        op for op in trace.ops if op.func_name in {"add", "__add__", "sub", "__sub__"}
    ]
    assert len(clamp.parents) == 3
    assert any("sub" in parent for parent in clamp.parents), clamp.parents
    assert sum("add" in parent for parent in clamp.parents) == 2
    assert len(bound_producers) == 3
    for producer in bound_producers:
        _assert_child_edge(producer, clamp)
    _assert_parents_match_args_template(clamp)


@pytest.mark.smoke
def test_traced_histogram_tensor_bins_recorded_as_parent() -> None:
    """torch.histogram records tensor ``bins`` supplied by keyword as a parent."""

    class HistogramWithTensorBins(nn.Module):
        """Build histogram values and bin edges from separate inputs."""

        def forward(self, values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
            """Histogram derived values using derived tensor bin edges.

            Parameters
            ----------
            values:
                Values to histogram.
            bin_edges:
                Source bin edges.

            Returns
            -------
            torch.Tensor
                Histogram counts.
            """

            shifted_values = values + 1
            scaled_bins = bin_edges * 2
            return torch.histogram(shifted_values, bins=scaled_bins)[0]

    trace = _trace_with_args_templates(
        HistogramWithTensorBins(),
        [torch.tensor([-1.5, -0.5, 0.5, 1.5]), torch.tensor([-1.0, 0.0, 1.0])],
    )
    histogram = next(op for op in trace.ops if op.func_name == "histogram")
    scaled_bins = next(op for op in trace.ops if op.func_name in {"mul", "__mul__"})
    assert any("mul" in parent for parent in histogram.parents), histogram.parents
    _assert_child_edge(scaled_bins, histogram)
    _assert_parents_match_args_template(histogram)


@pytest.mark.smoke
def test_traced_histogramdd_tensor_bins_recorded_as_parents() -> None:
    """torch.histogramdd records every tensor in keyword ``bins`` as a parent."""

    class HistogramddWithTensorBins(nn.Module):
        """Build two-dimensional histogram bins from a tensor input."""

        def forward(self, values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
            """Histogram points using a pair of derived tensor bin-edge vectors.

            Parameters
            ----------
            values:
                Two-dimensional points to histogram.
            bin_edges:
                Source vector for both dimensions' edges.

            Returns
            -------
            torch.Tensor
                Two-dimensional histogram counts.
            """

            shifted_values = values + 1
            lower_bins = bin_edges * 2
            upper_bins = bin_edges * 3
            return torch.histogramdd(shifted_values, bins=(lower_bins, upper_bins))[0]

    trace = _trace_with_args_templates(
        HistogramddWithTensorBins(),
        [
            torch.tensor([[-1.5, -1.5], [-0.5, -0.5], [0.5, 0.5], [1.5, 1.5]]),
            torch.tensor([-1.0, 0.0, 1.0]),
        ],
    )
    histogramdd = next(op for op in trace.ops if op.func_name == "histogramdd")
    bin_ops = [op for op in trace.ops if op.func_name in {"mul", "__mul__"}]
    assert sum("mul" in parent for parent in histogramdd.parents) == 2
    assert len(bin_ops) == 2
    for bin_op in bin_ops:
        _assert_child_edge(bin_op, histogramdd)
    _assert_parents_match_args_template(histogramdd)


@pytest.mark.parametrize("func_name", ["gradient", "clamp", "histogram"])
def test_plain_optional_tensor_apis_keep_only_required_tensor_parents(func_name: str) -> None:
    """Scalar/default optional arguments do not add parents to the three fixed APIs.

    Parameters
    ----------
    func_name:
        Name of the public torch API exercised with no tensor-valued optional argument.
    """

    class PlainOptionalTensorApi(nn.Module):
        """Call one optional-tensor API without a tensor-valued optional argument."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the configured API using scalar or default optional arguments.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                API result or first result tensor for multi-output APIs.
            """

            if func_name == "gradient":
                return torch.gradient(x, dim=(0,))[0]
            if func_name == "clamp":
                return torch.clamp(x, min=-1.0, max=1.0)
            return torch.histogram(x, bins=4)[0]

    trace = _trace_with_args_templates(PlainOptionalTensorApi(), torch.linspace(-2, 2, 5))
    op = next(layer for layer in trace.ops if layer.func_name == func_name)
    assert set(op.parents) == set(trace.input_layers)
    _assert_parents_match_args_template(op)
