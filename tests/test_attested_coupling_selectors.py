"""Step-qualified selectors (lane F42, attested coupling).

The trace-verb verdict's evidence-bar item "selectors can name step versus
pass": ``at_step(...)`` matches ops recorded inside declared episode steps at
capture time (the armed join session's live step position) and post hoc (the
persisted ``Op.episode_step`` stamps), composes with the existing
pass-qualified addressing for the step-x-pass cross-product, and refuses
typed everywhere a step qualifier has no steps to qualify.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import ArgumentTypeError
from torchlens.intervention import at_step
from torchlens.intervention.errors import SelectorCapabilityError, SiteResolutionError
from torchlens.ir.selector_eval import normalize_selector_like
from torchlens.options import EpisodeSpec


class Step(nn.Module):
    """One stepped block with a nested submodule (nesting must stamp too)."""

    def __init__(self, width: int = 4):
        super().__init__()
        self.inner = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.inner(x))


class Root(nn.Module):
    """Float episode root: three stepped calls with root-loop ops between."""

    def __init__(self, n_steps: int = 3):
        super().__init__()
        self.step = Step()
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self.step(x) + 0.0
        return x


def _episode_trace(model: Root, x: torch.Tensor, **kwargs) -> tl.Trace:
    return tl.trace(
        model,
        x,
        episode=EpisodeSpec(
            stepped_module=model.step, n_steps=model.n_steps, step_output_kind="digest"
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_at_step_argument_validation_refuses_typed():
    """Empty, negative, and bool step declarations refuse with the code."""

    for args in [(), (-1,), (2, -1), (True,), (1.5,)]:
        with pytest.raises(ArgumentTypeError) as excinfo:
            at_step(*args)
        assert excinfo.value.fields["code"] == "episode_step_selector_invalid"
        assert excinfo.value.fields["remedy"]


def test_at_step_normalizes_sorted_unique():
    """Steps deduplicate and sort; the spec payload is the ordered tuple."""

    selector = at_step(3, 1, 1)
    assert selector.steps == (1, 3)
    assert selector.selector_kind == "episode_step"
    assert selector.selector_value == (1, 3)


@pytest.mark.smoke
def test_at_step_spec_round_trip():
    """TargetSpec round-trip reconstructs the same ordered steps."""

    spec = at_step(2, 0).to_target_spec()
    back = normalize_selector_like(spec, lifecycle="site")
    assert back.steps == (0, 2)


# ---------------------------------------------------------------------------
# Capture-time qualification (the live step position)
# ---------------------------------------------------------------------------


def test_save_predicate_step_qualifies_live():
    """save= with at_step(k) retains exactly step k's matching pass."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4), save=tl.func("relu") & at_step(1))

    def has_payload(op) -> bool:
        try:
            return op.out is not None
        except Exception:
            return False

    saved_relu = [op.label for op in log.layer_list if "relu" in op.label and has_payload(op)]
    assert saved_relu == ["relu_1_2:2"]  # 0-based step 1 = pass 2


def test_root_loop_ops_between_steps_never_match():
    """Ops outside every stepped call belong to no step (never match)."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4), save=at_step(0, 1, 2))

    def has_payload(op) -> bool:
        try:
            return op.out is not None
        except Exception:
            return False

    # The final add feeds the model output, whose payload is always retained;
    # the interior root-loop adds are the step-membership probe.
    add_saved = [op.label for op in log.layer_list if "add" in op.label and has_payload(op)]
    assert add_saved == ["add_1_3:3"]


def test_capture_step_selector_without_episode_refuses_typed():
    """at_step in save= on a PLAIN capture refuses at the point of failure."""

    with pytest.raises(SelectorCapabilityError) as excinfo, pytest.warns(Warning):
        tl.trace(Root(), torch.randn(2, 4), save=at_step(0))
    assert excinfo.value.fields["code"] == "episode_step_selector_without_episode"
    assert "episode" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Post-hoc qualification (the persisted stamps)
# ---------------------------------------------------------------------------


def test_episode_step_stamps_land_on_stepped_ops_only():
    """Ops inside stepped call k (nested included) read step k; others None."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    by_label = {op.label: op.episode_step for op in log.layer_list}
    assert by_label["linear_1_1:1"] == 0  # nested submodule op stamps
    assert by_label["relu_1_2:2"] == 1
    assert by_label["relu_1_2:3"] == 2
    assert by_label["add_1_3:1"] is None
    assert by_label["input_1:1"] is None
    assert by_label["output_1:1"] is None


def test_post_hoc_step_resolution_matches_one_pass():
    """find_sites with a step qualifier resolves the step's pass exactly."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    table = log.find_sites(tl.func("relu") & at_step(2))
    assert [site.label for site in table] == ["relu_1_2:3"]


def test_post_hoc_step_and_pass_cross_product():
    """Step and pass qualification compose (the step-x-pass cross-product)."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    agree = log.find_sites(tl.label("relu_1_2:2") & at_step(1))
    assert [site.label for site in agree] == ["relu_1_2:2"]
    disagree = log.find_sites(tl.label("relu_1_2:2") & at_step(2))
    assert list(disagree) == []  # step and pass disagree: zero sites


def test_post_hoc_plain_capture_refuses_typed():
    """A step qualifier against a plain product refuses, never matches nothing."""

    log = tl.trace(Root(), torch.randn(2, 4))
    with pytest.raises(SiteResolutionError) as excinfo:
        log.find_sites(at_step(0))
    assert excinfo.value.fields["code"] == "episode_step_selector_without_episode"


def test_post_hoc_unstamped_episode_artifact_refuses_typed():
    """A stamp-less episode product (pre-F42 artifact) refuses typed."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    for op in log.layer_list:
        op.episode_step = None  # simulate the pre-stamping artifact
    with pytest.raises(SiteResolutionError) as excinfo:
        log.find_sites(at_step(0))
    assert excinfo.value.fields["code"] == "episode_step_unstamped"


@pytest.mark.smoke
def test_stamps_survive_save_load(tmp_path):
    """Op.episode_step persists (tlspec v9 slot) and serves post-hoc queries."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    target = tmp_path / "episode.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    assert {op.label: op.episode_step for op in loaded.layer_list} == {
        op.label: op.episode_step for op in log.layer_list
    }
    table = loaded.find_sites(tl.func("relu") & at_step(1))
    assert [site.label for site in table] == ["relu_1_2:2"]
