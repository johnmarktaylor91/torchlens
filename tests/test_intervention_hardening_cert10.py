"""Regression tests for the cert10 (round-10-prep) intervention hardening fixes.

BLOCKER -- ``_record_predicate_intervention_spec``
(``torchlens/backends/torch/ops.py``) built its dedup cache key from
``repr(decision.hook)``. When the fired helper is a ``HelperSpec`` carrying a live
``torch.Tensor`` argument -- i.e. TorchLens's flagship activation-steering/ablation
helpers ``tl.steer``, ``mean_ablate``, ``resample_ablate``, ``project_onto``/
``project_off``, ``swap_with`` -- that ``repr()`` call reprs the dataclass's tensor
field, which routes through TorchLens's own intercepted ``Tensor.__repr__``. The
interception itself (``print_override`` in ``torchlens/utils/tensor_utils.py``)
had a second, deeper bug: it paused logging only around the ``.cpu()`` conversion
and called ``.detach()`` *after* the ``pause_logging()`` block exited. Since
``detach`` is an ordinary decorated torch method (not in ``funcs_not_to_log`` or
``print_funcs``), calling it with logging still active logged a real "detach" op,
consumed a raw-op-counter slot, and left a graph orphan -- staling the target
label ``_record_predicate_intervention_spec`` had just recorded relative to the
op's real final raw label. ``save_intervention()`` then crashed with an unhandled
``SiteResolutionError`` on completely ordinary usage:
``tl.trace(model, x, intervene=tl.when(tl.func(...), tl.steer(direction)))`` then
``.save_intervention(path)``.

Two fixes land together:

1. ``ops.py::_record_predicate_intervention_spec`` now computes
   ``repr(decision.hook)`` inside ``pause_logging()`` so the dedup-key bookkeeping
   itself can never perturb the capture in progress, regardless of what the repr
   internally does.
2. ``tensor_utils.py::print_override`` now keeps ``.detach()`` inside its own
   ``pause_logging()`` block, closing the root cause so that *any* tensor
   ``repr()``/``str()`` fired during active logging -- not just the
   ``intervene=`` predicate path -- cannot leak an orphan "detach" op or burn a
   raw-op-counter slot. This also fixes plain (non-intervention) traces of
   multi-output torch ops such as ``torch.max``: ``torch.return_types.max``'s
   C-level ``__repr__`` reprs its tensor fields, which used to leak four
   ``detach_*_raw`` orphans and skip raw indices per call.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class _LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class _MaxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        values, _indices = torch.max(x, dim=1)
        return values


def test_steer_intervene_predicate_save_intervention_round_trips(tmp_path: Path) -> None:
    """BLOCKER: intervene=tl.when(..., tl.steer(dir)) + save_intervention succeeds.

    Before the fix this raised an unhandled ``SiteResolutionError`` because the
    predicate-recorded target label went stale relative to the op's real final
    raw label (the dedup-key repr leaked a live "detach" orphan op in between).
    """

    model = _LinearModel()
    x = torch.randn(2, 4)
    direction = torch.randn(4)

    trace = tl.trace(
        model,
        x,
        intervene=tl.when(tl.func("linear"), tl.steer(direction, feature_axis=1)),
    )

    spec_path = tmp_path / "steer_intervention.tlspec"
    # Must not raise SiteResolutionError.
    trace.save_intervention(spec_path)
    assert spec_path.exists()

    loaded = tl.io.load_intervention_spec(spec_path)
    # The recorded target must match the op's real final label, not a stale one.
    recorded_labels = {target.selector_value for target in loaded.targets}
    real_labels = {layer.layer_label for layer in trace if layer.layer_label.startswith("linear")}
    assert recorded_labels
    assert recorded_labels <= real_labels


def test_steer_intervene_multiple_firing_sites_do_not_compound(tmp_path: Path) -> None:
    """BLOCKER: multiple firing sites each resolve to their own real label.

    Regression guard for the dedup-key mechanism itself: two distinct "linear"
    ops both matching the predicate must each get their own correctly-resolved
    target label, with no cross-contamination or stale reuse between them.
    """

    model = _LinearModel()
    x = torch.randn(2, 4)
    direction = torch.randn(4)

    trace = tl.trace(
        model,
        x,
        intervene=tl.when(tl.func("linear"), tl.steer(direction, feature_axis=1)),
    )

    spec = trace._intervention_spec
    target_labels = sorted(target.selector_value for target in spec.targets)
    assert target_labels == ["linear_1_1", "linear_2_2"]

    spec_path = tmp_path / "steer_intervention_multi.tlspec"
    trace.save_intervention(spec_path)
    loaded = tl.io.load_intervention_spec(spec_path)
    loaded_labels = sorted(target.selector_value for target in loaded.targets)
    assert loaded_labels == target_labels


def test_plain_trace_of_torch_max_has_no_orphan_ops() -> None:
    """SWEEP: plain (non-intervention) trace of a multi-output torch op is clean.

    Before the ``print_override`` fix, formatting ``torch.return_types.max``'s
    repr during active logging leaked four ``detach_*_raw`` orphan ops and
    skipped raw-index slots (raw indices jumped 1, 2, 7, 8, 9 instead of a
    contiguous 1..5). Only ``linear`` and the two ``max`` outputs (plus input and
    output boundary ops) should ever be logged.
    """

    model = _MaxModel()
    x = torch.randn(2, 4)

    trace = tl.trace(model, x)

    assert trace._orphan_labels == []

    raw_indices = sorted(
        layer.raw_index for layer in trace if getattr(layer, "raw_index", None) is not None
    )
    # Contiguous starting at 1 -- no gaps left behind by a leaked "detach" op.
    assert raw_indices == list(range(1, len(raw_indices) + 1))

    func_names = [layer.func_name for layer in trace]
    assert "detach" not in func_names


class _ReprInForwardModel(nn.Module):
    """Calls ``repr()`` on a live tensor mid-forward, while logging is active."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.last_repr: str | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        # Exercises TorchLens's intercepted Tensor.__repr__ (print_override)
        # directly, with capture actively logging -- the shared root cause.
        self.last_repr = repr(x)
        return x


def test_repr_of_tensor_during_active_logging_does_not_log_detach() -> None:
    """SWEEP: repr()'ing a live tensor mid-capture never logs a real detach op.

    Directly exercises ``print_override`` (the shared root cause): calling
    ``repr()`` on a tensor while logging is active must not create any
    additional logged op or orphan, regardless of caller. Before the fix this
    leaked a ``detach_*_raw`` orphan (verified by reverting the fix locally --
    the orphan reappears and a raw-index slot is skipped).
    """

    model = _ReprInForwardModel()
    x = torch.randn(2, 4)

    trace = tl.trace(model, x)

    assert model.last_repr is not None
    assert "tensor(" in model.last_repr
    assert trace._orphan_labels == []
    func_names = [layer.func_name for layer in trace]
    assert "detach" not in func_names
    raw_indices = sorted(
        layer.raw_index for layer in trace if getattr(layer, "raw_index", None) is not None
    )
    assert raw_indices == list(range(1, len(raw_indices) + 1))
