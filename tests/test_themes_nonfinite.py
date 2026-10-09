"""F12 pins: the six-state nonfinite status pattern channel (N6).

The finite -> +Inf -> -Inf -> NaN corpus toy exercises the state
derivation; the two-mode degrade (zero coverage = one legend line, partial
coverage = per-node not_checked motifs) is the memo's debug-row contract.
NOT-CHECKED-as-FINITE is a zero-tolerance honesty class: an unchecked op
never presents as checked-clean.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses._nonfinite import (
    _MOTIFS,
    NONFINITE_STATES,
    derive_nonfinite_channel,
    nonfinite_spec_fn,
)
from torchlens.visualization.lenses.audit import CORPUS
from torchlens.visualization.node_spec import NodeSpec


def _member(name: str) -> Any:
    """Return one corpus member by name."""

    return next(member for member in CORPUS if member.name == name)


@pytest.fixture(scope="module")
def nonfinite_log() -> Any:
    """Full-save capture of the nonfinite chain toy."""

    model, x = _member("nonfinite_chain").build()
    log = tl.trace(model, x)
    yield log
    log.cleanup()


def test_vocabulary_is_the_closed_six(nonfinite_log: Any) -> None:
    """Six states, and the toy hits finite/pos_inf/neg_inf/nan."""

    assert NONFINITE_STATES == ("finite", "nan", "pos_inf", "neg_inf", "mixed", "not_checked")
    channel = derive_nonfinite_channel(nonfinite_log)
    present = set(channel.states.values())
    assert {"finite", "pos_inf", "neg_inf", "nan"} <= present
    assert not channel.zero_coverage


def test_states_key_both_label_spellings(nonfinite_log: Any) -> None:
    """Pass-qualified and bare labels both resolve (rendered nodes vary)."""

    channel = derive_nonfinite_channel(nonfinite_log)
    bare = {label for label in channel.states if ":" not in label}
    qualified = {label for label in channel.states if ":" in label}
    assert bare and qualified


@pytest.mark.smoke
def test_motifs_per_shape_table(nonfinite_log: Any) -> None:
    """box -> striped, oval -> wedged, box3d -> border-only degrade."""

    channel = derive_nonfinite_channel(nonfinite_log)
    spec_fn = nonfinite_spec_fn(channel)
    nan_label = next(label for label, state in channel.states.items() if state == "nan")
    layer = nonfinite_log[nan_label.split(":")[0]]

    boxed = spec_fn(layer, NodeSpec(lines=["x"], shape="box"))
    assert "striped" in boxed.style
    assert ":" in str(boxed.fillcolor)

    oval = spec_fn(layer, NodeSpec(lines=["x"], shape="oval"))
    assert "wedged" in oval.style

    degraded = spec_fn(layer, NodeSpec(lines=["x"], shape="box3d"))
    assert "striped" not in degraded.style and "wedged" not in degraded.style
    assert degraded.penwidth == 3.0  # the border motif still marks the state


def test_finite_ops_carry_no_motif(nonfinite_log: Any) -> None:
    """Default rendering IS the claim 'checked and clean'."""

    channel = derive_nonfinite_channel(nonfinite_log)
    spec_fn = nonfinite_spec_fn(channel)
    finite_label = next(
        label for label, state in channel.states.items() if state == "finite" and ":" not in label
    )
    spec = spec_fn(nonfinite_log[finite_label], NodeSpec(lines=["x"], shape="box"))
    assert spec.penwidth is None
    assert "striped" not in spec.style


@pytest.mark.smoke
def test_partial_coverage_marks_not_checked() -> None:
    """Selective save: unchecked ops get their own visible mark, never
    silence (the dangerous confusable case)."""

    model, x = _member("nonfinite_chain").build()
    log = tl.trace(model, x, save=tl.func("relu"))
    try:
        channel = derive_nonfinite_channel(log)
        assert not channel.zero_coverage
        assert "not_checked" in set(channel.states.values())
        spec_fn = nonfinite_spec_fn(channel)
        unchecked_label = next(
            label
            for label, state in channel.states.items()
            if state == "not_checked" and ":" not in label
        )
        spec = spec_fn(log[unchecked_label], NodeSpec(lines=["x"], shape="box"))
        assert "dashed" in spec.style
        assert any("checked 1 of" in line for line in channel.legend_lines)
    finally:
        log.cleanup()


def test_zero_coverage_channel_level_degrade() -> None:
    """Zero coverage: ONE prominent legend line, no per-node motifs, coded
    warning at the debug resolve."""

    model, x = _member("nonfinite_chain").build()
    with warnings.catch_warnings():
        # The zero-match save selector warning is the POINT of this fixture:
        # a capture retaining no payloads at all.
        warnings.simplefilter("ignore")
        log = tl.trace(model, x, save=tl.func("no_such_op_zzz"))
    try:
        channel = derive_nonfinite_channel(log)
        assert channel.zero_coverage
        assert "NOT CHECKED" in channel.legend_lines[0]
        spec_fn = nonfinite_spec_fn(channel)
        some_label = next(iter(channel.states))
        untouched = spec_fn(log[some_label.split(":")[0]], NodeSpec(lines=["x"], shape="box"))
        assert untouched.penwidth is None  # no per-node motifs in this mode
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolution = lenses.resolve_lens(log, "debug")
        codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
        assert "lens_nonfinite_not_checked" in codes
        assert any("NOT CHECKED" in line for line in resolution.disclosure)
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_debug_render_carries_motifs_end_to_end(nonfinite_log: Any, tmp_path: Any) -> None:
    """The rendered DOT carries the wedged motif and the legend lines."""

    resolution = lenses.resolve_lens(nonfinite_log, "debug")
    graph = nonfinite_log.draw(
        **resolution.draw_kwargs,
        vis_outpath=str(tmp_path / "nf"),
        vis_fileformat="svg",
        vis_save_only=True,
        return_graph=True,
    )
    assert "wedged" in graph.source
    assert "nonfinite status" in graph.source


def test_non_float_outputs_read_finite(nonfinite_log: Any) -> None:
    """Integer/bool payloads are finite by construction, never not_checked."""

    channel = derive_nonfinite_channel(nonfinite_log)
    assert channel.checked > 0


class _ExpTwice(torch.nn.Module):
    """One exp submodule called twice: pass 1 finite, pass 2 overflows to +Inf."""

    def __init__(self) -> None:
        """Build the shared exp submodule."""

        super().__init__()
        self.act = _Exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """exp(0) = 1, then exp(1000) = +Inf."""

        return self.act(self.act(x) * 1000.0)


class _Exp(torch.nn.Module):
    """A parameter-free exp, so both calls group into one two-pass layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Elementwise exp."""

        return torch.exp(x)


@pytest.mark.heavy
def test_unrolled_passes_carry_their_own_motif(tmp_path: Any) -> None:
    """Each unrolled pass node shows its OWN state, never a sibling pass's.

    The node-spec slot hands unrolled nodes their aggregate Layer, whose bare
    label keys the last pass's state; the finite first pass must stay clean.
    """

    log = tl.trace(_ExpTwice(), torch.zeros(1, 2))
    exp_ops = [op for op in log.ops if op.layer_label.startswith("exp")]
    assert len(exp_ops) == 2
    resolution = lenses.resolve_lens(log, "debug")
    source = log.draw(
        **{**resolution.draw_kwargs, "vis_mode": "unrolled"},
        vis_outpath=str(tmp_path / "nf_passes"),
        vis_fileformat="svg",
        vis_save_only=True,
    )
    inf_border = _MOTIFS["pos_inf"][1]
    statements = {}
    for op in exp_ops:
        name = op.label.replace(":", "pass")
        start = source.index(f"\t{name} [")
        statements[op.label] = source[start : source.index("]\n", start)]
    first, second = (statements[op.label] for op in exp_ops)
    assert inf_border not in first
    assert inf_border in second
    log.cleanup()
