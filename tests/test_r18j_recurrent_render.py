"""Regression tests for r18j -- recurrent-render multi-pass crash + identity class.

Covers the shared helper honesty guard (with a mutation proof) and every fixed
public visualization path exercised on a recurrent (multi-pass) model:

* helper -- MultiPassAmbiguityError / is_multipass_layer / get_multipass_attr
* H1 -- _module_key_for_forward_op no longer leaks the multi-pass ValueError
* H7 -- combined correspondence edges attach to declared forward nodes (no phantoms)
* H3 -- draw(node_label_fields=[..,"time",..]) rolled recurrent no longer crashes
* F5 -- node_label_fields "pass" shows the real per-pass number in unrolled mode
* F2 -- unrolled node titles are the resolvable layer_label:pass identity
* H4 -- Layer.show() raises a typed select-a-pass error on a recurrent layer
* H6 -- node overlays degrade to "n/a" on a rolled recurrent node (no crash)
* F15 -- the nan overlay reports "n/a" when no tensor was inspected
* H5 -- preview_fastlog is per-pass honest (no overwrite, correct pass_index)
* H9 -- intervention overlay colors only the intervened recurrent pass
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.utils._multipass_access import (
    MISSING,
    MultiPassAmbiguityError,
    get_multipass_attr,
    is_multipass_layer,
)

os.environ.setdefault("MPLBACKEND", "Agg")


class RecurrentLinear(nn.Module):
    """A 3-pass recurrent model: one Linear reused three times with tanh."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.fc(x))
        return x


class RecurrentReLU(nn.Module):
    """A 3-pass recurrent ReLU model (single shared module)."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.relu(x)
        return x


class FeedForward(nn.Module):
    """A plain feedforward model (no recurrence)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def _recurrent_trace():
    return tl.trace(RecurrentLinear(), torch.randn(2, 8))


# --------------------------------------------------------------------------- #
# Shared helper + mutation proof
# --------------------------------------------------------------------------- #


def test_is_multipass_layer_only_true_for_aggregate_recurrent_layer():
    log = _recurrent_trace()
    layer = log["linear_1_1"]
    op = layer.ops[2]
    assert is_multipass_layer(layer) is True
    # An Op proxies num_passes from its parent but is NOT itself multi-pass.
    assert is_multipass_layer(op) is False
    assert is_multipass_layer(object()) is False
    single = tl.trace(FeedForward(), torch.randn(2, 8))
    assert is_multipass_layer(single["linear_1_1"]) is False


def test_get_multipass_attr_semantics():
    log = _recurrent_trace()
    layer = log["linear_1_1"]
    op = layer.ops[2]

    # Op resolves its own per-pass value.
    assert get_multipass_attr(op, "func_duration", 0.0) is not None
    # Missing attribute with a default behaves like getattr.
    assert get_multipass_attr(layer, "no_such_attr", "DEF") == "DEF"
    # Aggregate-stable field resolves normally (no ValueError, no None).
    assert get_multipass_attr(layer, "flops_forward", 0, multipass=None) is not None
    # Honest degrade sentinel on a multi-pass ambiguity.
    assert get_multipass_attr(layer, "out", None, multipass=None) is None


def test_get_multipass_attr_honesty_guard():
    """The helper must NEVER swallow the multi-pass ambiguity into the default.

    This is the load-bearing honesty guard the whole sprint depends on. Mutation
    proof: replacing the ``except ValueError`` branch in
    ``torchlens/utils/_multipass_access.py:get_multipass_attr`` with
    ``return default`` (the naive ``getattr(layer, attr, default)`` behaviour the
    DECISION forbids) makes this assertion return ``0.0`` instead of raising --
    killing the test. Verified executed in R18J_REPORT.md.
    """

    log = _recurrent_trace()
    layer = log["linear_1_1"]
    # Default multipass=RAISE: per-pass read on a recurrent aggregate MUST raise a
    # typed error, never fabricate the 0.0 default.
    with pytest.raises(MultiPassAmbiguityError):
        get_multipass_attr(layer, "func_duration", 0.0)
    # MultiPassAmbiguityError is a ValueError subclass (back-compatible catches).
    assert issubclass(MultiPassAmbiguityError, ValueError)


def test_get_multipass_attr_propagates_unrelated_value_error():
    class Weird:
        @property
        def boom(self):
            raise ValueError("unrelated failure")

    with pytest.raises(ValueError, match="unrelated failure"):
        get_multipass_attr(Weird(), "boom", "default")
    assert MISSING is not None  # sentinel importable


# --------------------------------------------------------------------------- #
# H1 / H7 -- combined render (_render_leaf)
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_h1_module_key_for_forward_op_no_multipass_leak():
    from torchlens.visualization._render_leaf import (
        _forward_op_is_module_output,
        _module_key_for_forward_op,
    )

    log = _recurrent_trace()
    layer = log["tanh_1_2"]  # aggregate recurrent Layer (grad_fn.op is this type)
    # Must not raise the multi-pass ValueError tripwire any more.
    assert isinstance(_forward_op_is_module_output(layer), bool)
    key = _module_key_for_forward_op(layer)
    assert key is None or isinstance(key, str)


def test_h7_forward_correspondence_node_name_scoped_to_multipass():
    from torchlens.visualization._render_leaf import _forward_correspondence_node_name

    log = _recurrent_trace()
    single = tl.trace(FeedForward(), torch.randn(2, 8))
    # Scoped to the multi-pass case ONLY (gate rework): a recurrent aggregate Layer
    # has an unrecoverable forward pass -> None so the caller SKIPS the edge rather
    # than emit an ambiguous aggregate endpoint.
    assert _forward_correspondence_node_name(log["tanh_1_2"]) is None
    assert _forward_correspondence_node_name(None) is None
    # A NON-recurrent op preserves the historical aggregate layer_label emission
    # verbatim (locked render-identity oracle stays byte-identical; the feedforward
    # phantom-correspondence cosmetic fix is DEFERRED pending an approved golden).
    name = _forward_correspondence_node_name(single["linear_1_1"])
    assert name == single["linear_1_1"].layer_label


# --------------------------------------------------------------------------- #
# H3 / F5 / F2 -- node labels (_render_nodes)
# --------------------------------------------------------------------------- #


def test_h3_rolled_node_label_time_no_crash(tmp_path):
    log = _recurrent_trace()
    graph = log.draw(
        vis_mode="rolled",
        vis_save_only=True,
        vis_outpath=str(tmp_path / "rolled"),
        node_label_fields=["label", "memory", "time", "flops", "pass"],
        return_graph=True,
    )
    assert "linear_1_1" in graph.source  # rendered without leaking the tripwire


def test_f5_unrolled_pass_field_shows_real_pass_numbers(tmp_path):
    import re

    log = _recurrent_trace()
    graph = log.draw(
        vis_mode="unrolled",
        vis_save_only=True,
        vis_outpath=str(tmp_path / "unrolled"),
        node_label_fields=["label", "pass"],
        return_graph=True,
    )
    # The 'pass' rows must include 1, 2 and 3 -- not just 1 for every pass.
    present = set(re.findall(r">([123])<", graph.source))
    assert {"1", "2", "3"} <= present


def test_f2_unrolled_titles_are_resolvable(tmp_path):
    import re

    log = _recurrent_trace()
    graph = log.draw(
        vis_mode="unrolled",
        vis_save_only=True,
        vis_outpath=str(tmp_path / "titles"),
        return_graph=True,
    )
    titles = set(re.findall(r"<B>([a-z]+_[0-9_]+:[0-9]+)</B>", graph.source))
    assert titles, "expected pass-qualified unrolled titles"
    for title in titles:
        # Every displayed title must resolve back to a real trace layer.
        assert log[title] is not None
    # The fabricated position-renumbered forms must be gone.
    assert "linear_1_3:2" not in graph.source
    assert "linear_1_5:3" not in graph.source


# --------------------------------------------------------------------------- #
# H4 -- Layer.show()
# --------------------------------------------------------------------------- #


def test_h4_layer_show_recurrent_raises_select_pass():
    log = tl.trace(RecurrentLinear(), torch.randn(2, 8), save=lambda op: True)
    with pytest.raises(ValueError) as excinfo:
        log["linear_1_1"].show()
    message = str(excinfo.value)
    assert "recurrent" in message and "pass" in message
    # A specific pass does not raise the recurrence error (reaches the display gate).
    result = log["linear_1_1:2"].show()
    assert result is not None


# --------------------------------------------------------------------------- #
# H6 / F15 -- overlays
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overlay", ["magnitude", "time", "bytes", "flops", "nan", "grad_norm", "intervention"]
)
def test_h6_rolled_overlays_do_not_crash(tmp_path, overlay):
    log = _recurrent_trace()
    graph = log.draw(
        vis_mode="rolled",
        node_overlay=overlay,
        vis_save_only=True,
        vis_outpath=str(tmp_path / f"ov_{overlay}"),
        return_graph=True,
    )
    assert graph.source  # rendered without leaking the multi-pass tripwire


def test_h6_rolled_per_pass_overlays_are_na():
    from torchlens.visualization.overlays import builtin_overlay_value, format_overlay_value

    log = _recurrent_trace()
    layer = log["linear_1_1"]  # aggregate recurrent Layer
    for overlay in ("time", "magnitude", "grad_norm"):
        value = builtin_overlay_value(layer, overlay)
        assert value is None
        assert format_overlay_value(overlay.replace("_", "-"), value).endswith("n/a")


def test_f15_nan_overlay_reports_na_when_nothing_inspected():
    from torchlens.visualization.overlays import builtin_overlay_value

    # No available tensor -> "not checked" -> None -> "nan: n/a" (NOT False -> "no").
    assert builtin_overlay_value(SimpleNamespace(out=None), "nan") is None
    assert builtin_overlay_value(SimpleNamespace(out="not a tensor"), "nan") is None
    # A real finite tensor -> checked, clean -> False ("no").
    assert builtin_overlay_value(SimpleNamespace(out=torch.zeros(4)), "nan") is False
    # A real non-finite tensor -> True ("yes").
    assert (
        builtin_overlay_value(SimpleNamespace(out=torch.tensor([float("nan"), 1.0])), "nan") is True
    )


# --------------------------------------------------------------------------- #
# H5 -- preview_fastlog
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_h5_preview_fastlog_recurrent_per_pass(tmp_path):
    from torchlens.visualization.fastlog_preview import (
        _build_preview_nodes,
        _make_node_spec_fn,
        preview_fastlog,
    )
    from torchlens.visualization.node_spec import NodeSpec

    log = tl.trace(RecurrentReLU(), torch.tensor([-1.0, 2.0]))
    # No crash on a recurrent model.
    preview_fastlog(
        log,
        predicate=lambda ctx: getattr(ctx, "pass_index", None) == 1,
        vis_save_only=True,
        vis_outpath=str(tmp_path / "preview"),
    )
    # Decisions are keyed per pass (no overwrite) with correct per-pass context.
    preview = _build_preview_nodes(log, lambda ctx: ctx.event_index == 3)
    relu_keys = sorted(k for k in preview if "relu" in k and ":" in k)
    assert relu_keys == ["relu_1_1:1", "relu_1_1:2", "relu_1_1:3"]
    assert [preview[k].ctx.pass_index for k in relu_keys] == [0, 1, 2]
    # The event_index==3 predicate keeps ONLY the second recurrent call.
    ops = list(log.layer_logs["relu_1_1"].ops.values())
    callback = _make_node_spec_fn(
        preview,
        color_kept="green",
        color_rejected="gray",
        color_unreachable="yellow",
        color_predicate_error="red",
        show_predicate_inputs=False,
        show_module_events=False,
    )
    colors = [callback(op, NodeSpec(lines=[op.label])).fillcolor for op in ops]
    assert colors == ["gray", "green", "gray"]


# --------------------------------------------------------------------------- #
# H9 -- intervention overlay
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_h9_intervention_colors_only_intervened_pass():
    from torchlens.intervention import resolver
    from torchlens.visualization.node_spec import (
        NodeSpec,
        intervention_site_and_cone_labels,
        make_intervention_node_spec_fn,
    )

    log = tl.trace(RecurrentReLU(), torch.tensor([-1.0, 2.0]))
    ops = list(log.layer_logs["relu_1_1"].ops.values())
    fake = SimpleNamespace(
        _intervention_spec=SimpleNamespace(
            targets=("pass2",), target_value_specs=(), hook_specs=()
        ),
        layer_list=ops,
    )
    original = resolver.resolve_sites
    resolver.resolve_sites = lambda trace, target, max_fanout: [ops[1]]  # resolve pass 2 only
    try:
        callback = make_intervention_node_spec_fn(
            fake, show_cone=False, graph_overrides=None, user_node_spec_fn=None
        )
        colors = [callback(op, NodeSpec(lines=[op.label])).color for op in ops]
        # Only pass 2 is colored; the other passes are untouched.
        assert colors[1] is not None
        assert colors[0] is None and colors[2] is None
        # A rolled aggregate Layer node still colors (any pass is a site) and never
        # leaks the multi-pass tripwire.
        aggregate = log.layer_logs["relu_1_1"]
        assert callback(aggregate, NodeSpec(lines=["relu_1_1"])).color is not None
        # Public helper contract is UNCHANGED: aggregate layer_labels only.
        site_labels, _ = intervention_site_and_cone_labels(fake, show_cone=False)
        assert site_labels == {"relu_1_1"}
    finally:
        resolver.resolve_sites = original
