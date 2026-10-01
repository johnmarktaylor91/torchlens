"""Phase 8 bundle comparison primitive tests."""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import torch
from torch import nn

import torchlens as tl


class _TinyRelu(nn.Module):
    """Small model used for bundle primitive tests."""

    def __init__(self, offset: float = 0.0) -> None:
        """Initialize the model.

        Parameters
        ----------
        offset:
            Constant output offset.
        """

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x)) + self.offset


def _capture_pair(seed: int, offset: float) -> tl.Bundle:
    """Return a two-member bundle with deterministic captures.

    Parameters
    ----------
    seed:
        Random seed for model initialization and input.
    offset:
        Output offset for the second model.

    Returns
    -------
    tl.Bundle
        Bundle with baseline and compared traces.
    """

    torch.manual_seed(seed)
    x = torch.randn(2, 3)
    baseline_model = _TinyRelu()
    changed_model = _TinyRelu(offset=offset)
    baseline = tl.trace(
        baseline_model, x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    changed = tl.trace(changed_model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    return tl.bundle({"baseline": baseline, "changed": changed}, baseline="baseline")


def test_bundle_call_accessors_resolve_listed_labels() -> None:
    """Bundle module/grad-fn call accessors can index labels they advertise."""

    torch.manual_seed(0)
    model = _TinyRelu()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)
    bundle = tl.bundle({"first": trace, "second": trace}, baseline="first")

    module_label = next(iter(bundle.module_calls))
    grad_fn_label = next(iter(bundle.grad_fn_calls))

    assert type(bundle.module_calls[module_label]).__name__ == "SuperModuleCall"
    assert type(bundle.grad_fn_calls[grad_fn_label]).__name__ == "SuperGradFnCall"


def _three_model_pairs() -> Iterator[tl.Bundle]:
    """Yield the three model pairs required by Phase 8.

    Yields
    ------
    tl.Bundle
        Bundle under test.
    """

    yield _capture_pair(seed=1, offset=0.0)
    yield _capture_pair(seed=2, offset=0.25)
    yield _capture_pair(seed=3, offset=-0.5)


def test_delta_map_norm_delta_output_delta_on_three_model_pairs() -> None:
    """Bundle delta helpers return stable per-node and per-output payloads."""

    for bundle in _three_model_pairs():
        delta = bundle.delta_map("relative_l2")
        norm_delta = bundle.norm_delta()
        output_delta = bundle.output_delta("baseline")
        comparison = bundle.compare("relative_l2")

        assert delta
        assert norm_delta == delta
        assert set(output_delta) == {"baseline", "changed"}
        assert comparison["nodes"] == delta
        assert comparison["outputs"] == output_delta
        assert comparison["metric"] == "relative_l2"
        assert any("baseline" in node_values for node_values in delta.values())


def test_bundle_method_count_stays_within_phase_budget() -> None:
    """Bundle public method/property count includes Phase 11 persistence."""

    members = [
        name
        for name, value in inspect.getmembers(tl.Bundle)
        if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, property))
    ]

    assert len(members) <= 34
    assert "joint_metric" in members
    assert "set_capacity" in members
    assert "save" in members
    assert "remove_except" in members
    assert "supergraph" in members


def test_bundle_add_remove_accept_single_and_list_forms() -> None:
    """Bundle add/remove/remove_except normalize single and list member references."""

    source_bundle = _capture_pair(seed=41, offset=0.1)
    log_a = source_bundle["baseline"]
    log_b = source_bundle["changed"]
    log_c = tl.trace(
        _TinyRelu(offset=0.2),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    bundle = tl.bundle({"a": log_a})

    assert bundle.add(log_b, names="b") is bundle
    assert bundle.add([log_c], names=["c"]) is bundle
    assert bundle.names == ["a", "b", "c"]

    assert bundle.remove(log_b) is log_b
    assert bundle.names == ["a", "c"]

    removed = bundle.remove(["c"])
    assert removed == [log_c]
    assert bundle.names == ["a"]

    bundle.add([log_b, log_c], names=["b", "c"])
    bundle.remove_except([log_a, "c"])
    assert bundle.names == ["a", "c"]


def test_store_comparison_stamps_named_field_and_matches_delta_map() -> None:
    """store_comparison persists delta_map's values on the supergraph nodes."""

    bundle = _capture_pair(seed=4, offset=0.25)

    name = bundle.store_comparison("relative_l2")

    assert name == "relative_l2:out@baseline"
    stored = bundle.stored_comparison(name)
    assert stored == bundle.delta_map("relative_l2")
    assert bundle.stored_comparison_names() == (name,)
    # Queryable per node, without recomputation.
    node_label = next(iter(stored))
    node = bundle.supergraph.nodes[node_label]
    assert node.comparisons[name] == stored[node_label]


def test_store_comparison_overwrites_same_name_and_keeps_others() -> None:
    """Restoring under one name replaces it; other names are untouched."""

    bundle = _capture_pair(seed=5, offset=0.5)

    out_name = bundle.store_comparison("relative_l2")
    custom = bundle.store_comparison("relative_l2", name="custom")
    assert set(bundle.stored_comparison_names()) == {out_name, custom}

    again = bundle.store_comparison("relative_l2")
    assert again == out_name
    assert set(bundle.stored_comparison_names()) == {out_name, custom}
    assert bundle.stored_comparison(out_name) == bundle.stored_comparison(custom)


def test_store_comparison_callable_metric_requires_explicit_name() -> None:
    """A callable metric has no derivable stable name and refuses typed."""

    import pytest

    from torchlens._errors import InvalidArgumentError

    bundle = _capture_pair(seed=6, offset=0.25)

    def metric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the maximum absolute difference.

        Parameters
        ----------
        a:
            Baseline tensor.
        b:
            Compared tensor.

        Returns
        -------
        torch.Tensor
            Scalar distance.
        """

        return (a - b).abs().max()

    with pytest.raises(InvalidArgumentError) as excinfo:
        bundle.store_comparison(metric)
    assert excinfo.value.fields["code"] == "comparison_name_required"

    named = bundle.store_comparison(metric, name="max_abs")
    assert named == "max_abs"
    assert bundle.stored_comparison("max_abs")


def test_stored_comparison_unknown_name_refuses_typed() -> None:
    """Reading a never-stored name refuses typed, never an empty dict."""

    import pytest

    from torchlens._errors import InvalidArgumentError

    bundle = _capture_pair(seed=7, offset=0.25)

    with pytest.raises(InvalidArgumentError) as excinfo:
        bundle.stored_comparison("never_stored")
    assert excinfo.value.fields["code"] == "comparison_unknown"
