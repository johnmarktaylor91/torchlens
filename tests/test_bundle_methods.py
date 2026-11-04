"""Phase 8 bundle comparison primitive tests."""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
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
    baseline = tl.trace(baseline_model, x, capture=tl.options.CaptureOptions(replayable=True))
    changed = tl.trace(changed_model, x, capture=tl.options.CaptureOptions(replayable=True))
    return tl.Bundle({"baseline": baseline, "changed": changed}, baseline="baseline")


def test_bundle_call_accessors_resolve_listed_labels() -> None:
    """Bundle module/grad-fn call accessors can index labels they advertise."""

    torch.manual_seed(0)
    model = _TinyRelu()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss)
    bundle = tl.Bundle({"first": trace, "second": trace}, baseline="first")

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


def test_private_delta_reads_on_three_model_pairs() -> None:
    """The private per-node reads behind compare and the diff picture stay stable."""

    for bundle in _three_model_pairs():
        delta = bundle._delta_map("relative_l2")
        norm_delta = bundle._norm_delta()
        output_delta = bundle._output_delta("baseline")
        rows = bundle.compare("baseline", "changed", metric="relative_l2")

        assert delta
        assert norm_delta == delta
        assert set(output_delta) == {"baseline", "changed"}
        assert rows and all(
            isinstance(label, str) and isinstance(value, float) for label, value in rows
        )
        assert dict(rows).keys() <= set(delta)
        assert any("baseline" in node_values for node_values in delta.values())


def test_bundle_method_count_stays_within_phase_budget() -> None:
    """Bundle public method/property count includes Phase 11 persistence."""

    members = [
        name
        for name, value in inspect.getmembers(tl.Bundle)
        if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, property))
    ]

    # 44 = the historical 34 cap + the F03 ledger-memo roster, raised
    # deliberately per the lane brief: relate/derive_episode_status promoted
    # to real methods (item 0), the lineage read surface (bundle_id,
    # operations, member_construction; items 0b/3), the provenance join
    # (why, provenance; item 4), vary (item 5), and the effect-table reads
    # (effects; item 8). The former helper-table methods (stored comparisons,
    # upstream/downstream, token_pairs, token_divergence, add_alternative)
    # and draw are real methods now (max-push walk surface.7, late.26) and
    # count here; measure_members is removed into Bundle.map (walk names.11).
    assert len(members) <= 45
    assert "relate" in members
    for removed in (
        "joint_metric",
        "set_capacity",
        "help",
        "cluster",
        "node",
        "compare_at",
        "measure_members",
    ):
        assert removed not in members, removed
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
        capture=tl.options.CaptureOptions(replayable=True),
    )
    bundle = tl.Bundle({"a": log_a})

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

    from torchlens._errors import InvalidArgumentError

    bundle = _capture_pair(seed=7, offset=0.25)

    with pytest.raises(InvalidArgumentError) as excinfo:
        bundle.stored_comparison("never_stored")
    assert excinfo.value.fields["code"] == "comparison_unknown"


def test_bundle_summary_is_a_text_card() -> None:
    """Bundle.summary returns the card: members, outcomes, baseline, relations (RC-278)."""

    bundle = _capture_pair(seed=0, offset=0.5)
    card = bundle.summary()
    assert isinstance(card, str)
    assert "members (2): baseline (baseline), changed" in card
    assert "baseline: 'baseline'" in card
    assert "outcomes:" in card and "relations:" in card
    assert "structurally consistent:" in card


def test_bundle_to_pandas_stacks_members() -> None:
    """Bundle.to_pandas stacks each member's table under a leading member column (RC-278)."""

    pd = pytest.importorskip("pandas")
    bundle = _capture_pair(seed=0, offset=0.5)
    frame = bundle.to_pandas()
    assert list(frame.columns)[0] == "member"
    assert list(frame.index) == list(range(len(frame)))
    per_member = {name: len(bundle[name].to_pandas()) for name in bundle.names}
    assert frame["member"].value_counts().to_dict() == per_member
    assert list(dict.fromkeys(frame["member"])) == bundle.names
    empty = tl.Bundle().to_pandas()
    assert isinstance(empty, pd.DataFrame)
    assert list(empty.columns) == ["member"] and len(empty) == 0


def test_bundle_to_agent_json_dumps_every_member() -> None:
    """Bundle.to_agent_json is the agent_bundle.v1 dump with per-member dumps (RC-278)."""

    import json

    bundle = _capture_pair(seed=0, offset=0.5)
    dump = bundle.to_agent_json(max_ops=3)
    assert dump["schema"] == "torchlens.agent_bundle.v1"
    assert dump["member_count"] == 2
    assert dump["names"] == ["baseline", "changed"]
    assert dump["baseline_name"] == "baseline"
    assert sum(dump["outcome"].values()) == 2
    assert set(dump["members"]) == {"baseline", "changed"}
    assert dump["members"]["changed"] == bundle["changed"].to_agent_json(max_ops=3)
    json.dumps(dump)
