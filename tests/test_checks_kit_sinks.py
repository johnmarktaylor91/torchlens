"""Attention-sink and outlier-residual-dimension probe over the ``capture.hooks`` seam.

Toy tests pin the statistic and its stated conventions (skipped positions, the query
mask, the probability check, refusals); the slow test pins the GPT-2 small row of
docs/reference/checks_kit.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.errors
from torchlens.debug._checks._sinks import SinkProbe


class _Block(nn.Module):
    """Stand-in residual block: returns its input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the residual through."""

        return x * 1.0


class _ToyAttention(nn.Module):
    """Two heads over five positions: head 0 sinks on key 0, head 1 is uniform."""

    def __init__(self) -> None:
        super().__init__()
        scores = torch.zeros(1, 2, 5, 5)
        scores[:, 0, :, 0] = 6.0
        self.register_buffer("scores", scores)
        self.block = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend with the fixed scores and add the residual's mean."""

        probs = torch.softmax(self.scores + 0.0 * x.mean(), dim=-1)
        return probs.sum() + self.block(x).mean()


def _residual() -> torch.Tensor:
    """``1 x 5 x 8`` residual: dimension 3 is 40 everywhere, dimension 5 only at position 0."""

    torch.manual_seed(0)
    x = torch.randn(1, 5, 8)
    x[..., 3] = 40.0
    x[:, 0, 5] = 400.0
    return x


def _run(probe: SinkProbe, x: torch.Tensor | None = None) -> tl.Trace:
    """Trace the toy with the probe's hooks."""

    capture = tl.options.CaptureOptions(hooks=probe.hook_plan())
    return tl.trace(_ToyAttention(), _residual() if x is None else x, capture=capture)


def test_sink_head_and_outlier_dim_are_flagged_without_adding_ops() -> None:
    """Head 0 sinks on position 0; dimension 3 is an outlier; the trace is unchanged."""

    x = _residual()
    bare = len(tl.trace(_ToyAttention(), x).ops)
    probe = SinkProbe(attention=[tl.func("softmax")], residual=[tl.module("block")])
    trace = _run(probe, x)
    assert len(trace.ops) == bare

    assert [(head.head, head.position) for head in probe.sink_heads] == [(0, 0)]
    assert probe.sink_heads[0].mass > 0.95
    assert [dim.dim for dim in probe.outlier_dims] == [3]
    attention = next(f for f in probe.facts().values() if f["kind"] == "attention")
    assert attention["top_mass"][1] == pytest.approx(0.2, abs=1e-4)
    assert sorted(probe.observed_fires.values()) == [1, 1]


def test_skip_positions_zero_lets_the_first_token_dominate() -> None:
    """With position 0 kept, its massive activation on dimension 5 is flagged too."""

    probe = SinkProbe(residual=[tl.module("block")], skip_positions=0)
    _run(probe)
    assert {dim.dim for dim in probe.outlier_dims} == {3, 5}


def test_query_mask_removing_every_kept_query_skips_the_site() -> None:
    """The mask and skip_positions together leave no query: stated, not guessed."""

    mask = torch.tensor([[True, False, False, False, False]])
    probe = SinkProbe(attention=[tl.func("softmax")], query_mask=mask)
    _run(probe)
    (facts,) = probe.facts().values()
    assert "no query kept" in facts["skipped"]
    assert probe.sink_heads == ()


def test_query_mask_restricts_the_average() -> None:
    """Keeping only query 1 averages over that one row."""

    mask = torch.tensor([[False, True, False, False, False]])
    probe = SinkProbe(attention=[tl.func("softmax")], query_mask=mask)
    _run(probe)
    (facts,) = probe.facts().values()
    assert facts["top_mass"][1] == pytest.approx(0.2, abs=1e-4)
    assert facts["sink_heads"] == [0]


def test_non_probability_values_are_skipped_with_a_reason() -> None:
    """A residual tensor named as an attention site is reported, never scored."""

    probe = SinkProbe(attention=[tl.module("block")])
    _run(probe)
    (facts,) = probe.facts().values()
    assert "not probabilities" in facts["skipped"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "at least one attention or residual site"),
        ({"residual": [tl.module("block")], "outlier_factor": 1.0}, "outlier_factor"),
        ({"residual": [tl.module("block")], "skip_positions": -1}, "0 or more"),
        ({"residual": [tl.module("block")], "skip_positions": True}, "must be an int"),
    ],
)
def test_bad_settings_are_refused(kwargs, match) -> None:
    """Each invalid convention is named in the refusal."""

    with pytest.raises(ValueError, match=match):
        SinkProbe(**kwargs)


def test_mass_threshold_outside_unit_interval_is_refused_typed() -> None:
    """The fraction knob reuses the kit's typed fraction refusal."""

    with pytest.raises(torchlens.errors.TorchLensError) as info:
        SinkProbe(attention=[tl.func("softmax")], mass_threshold=1.5)
    assert info.value.fields["code"] == "check_fraction_invalid"


def _common():
    """The replications' shared loader (pinned GPT-2 small revision, eager attention)."""

    path = Path(__file__).resolve().parents[1] / "examples" / "replications" / "_common.py"
    spec = importlib.util.spec_from_file_location("_sinks_test_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_gpt2_small_row() -> None:
    """GPT-2 small: most heads sink on position 0; dimension 447 stands out in every block."""

    model, tokenizer = _common().load_gpt2()
    prompt = "When Mary and John went to the store, John gave a drink to"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    blocks = [tl.module(f"transformer.h.{i}") for i in range(12)]
    probe = SinkProbe(attention=[tl.func("softmax")], residual=blocks)
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))
    trace.cleanup()

    heads = probe.sink_heads
    assert {head.position for head in heads} == {0}
    assert len(probe.observed_fires) == 24
    assert len(heads) >= 90
    residual = [facts for facts in probe.facts().values() if facts["kind"] == "residual"]
    assert len(residual) == 12
    assert all(447 in facts["outlier_dims"] for facts in residual)
    assert {447, 373, 64} <= {dim.dim for dim in probe.outlier_dims}
