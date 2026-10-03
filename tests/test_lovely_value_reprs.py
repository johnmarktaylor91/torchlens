"""F10 lovely item 6: value-record line/card grammar laws.

repr = ONE envelope+core line; str = bounded card whose FIRST line is the
repr (D15); multi-pass never pools (D32); honesty tokens are never
laundered; repr never raises and never copies payloads (voice rule 10).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _Mixed(nn.Module):
    """Small mixed model: params, module nesting, elementwise tail."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.fc(x)))


class _Recurrent(nn.Module):
    """Weight-reused loop: one Linear applied three times (multi-pass)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


@pytest.fixture(scope="module")
def mixed_trace():
    """One finished capture; the MODEL is kept alive (Params render live
    version-checked cores only while the source model exists -- D30)."""

    # Seeded (round-2 CI triage, 2026-10-01): an unseeded draw made the
    # layernorm output's near-zero per-row mean's magnitude -- and so the
    # "mean=" field's scientific-notation width in test_repr_is_one_bounded_line
    # -- depend on the shared global RNG stream's position, which pytest-randomly
    # varies with collection order. A fixed seed makes the fixture's data (and
    # every line-length assertion over it) reproducible regardless of order.
    torch.manual_seed(0)
    model = _Mixed().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    trace._keepalive_model = model  # pin lifetime for the live-param tests
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def recurrent_trace():
    """One finished multi-pass capture."""

    torch.manual_seed(0)  # reproducible regardless of collection order; see mixed_trace
    trace = tl.trace(_Recurrent().eval(), torch.randn(2, 4))
    yield trace
    trace.cleanup()


def _value_records(trace):
    """Every value-bearing record kind of one trace."""

    records = list(trace.layer_list)
    records.extend(trace.layers[label] for label in trace.layer_labels)
    records.extend(trace.params.values())
    records.extend(trace.modules.values())
    return records


def test_repr_is_one_bounded_line(mixed_trace) -> None:
    """Voice rule 9: repr is one line; the budget is 120 columns."""

    for record in _value_records(mixed_trace):
        line = repr(record)
        assert "\n" not in line, type(record).__name__
        assert len(line) <= 120, (type(record).__name__, len(line), line)


def test_card_first_line_is_repr_and_bounded(mixed_trace) -> None:
    """D15: str is a <=8-line card whose first line IS the repr."""

    for record in _value_records(mixed_trace):
        card = str(record)
        lines = card.splitlines()
        assert lines[0] == repr(record), type(record).__name__
        assert len(lines) <= 8, (type(record).__name__, len(lines))


def test_op_card_exits_and_content(mixed_trace) -> None:
    """The Op card names its exits; the core carries real numbers."""

    op = mixed_trace["relu_1_2"].ops[0]
    line = repr(op)
    assert "relu_1_2" in line and "mean=" in line and "@cpu" in line
    card = str(op)
    assert "More:" in card
    assert ".lookup_keys" in card  # the eleven keys moved behind the exit
    assert "graph" in card


def test_multipass_layer_never_pools(recurrent_trace) -> None:
    """D32: a k-pass Layer shows per-pass cores or a count, never pooled."""

    multi = next(
        layer
        for label in recurrent_trace.layer_labels
        for layer in [recurrent_trace.layers[label]]
        if layer.num_passes > 1
    )
    line = repr(multi)
    assert f"x{multi.num_passes} passes" in line
    assert "mean=" not in line  # a pooled statistic would be a plausible lie
    card = str(multi)
    assert "pass 1/" in card  # per-pass cores are the honest form
    per_pass_lines = [row for row in card.splitlines() if row.strip().startswith("pass ")]
    assert 1 <= len(per_pass_lines) <= 3


def test_predicate_save_not_saved_token() -> None:
    """(not saved) is distinguishable from saved-and-boring (memo 4.4)."""

    trace = tl.trace(_Mixed().eval(), torch.randn(2, 8), save=tl.func("relu"))
    unsaved = next(op for op in trace.layer_list if not op.has_saved_activation)
    assert "(not saved)" in repr(unsaved.ops[0])
    saved = trace["relu_1_2"].ops[0]
    assert "(not saved)" not in repr(saved)
    trace.cleanup()


def test_param_line_live_versioned_with_tie_disclosure(mixed_trace) -> None:
    """D30: Params render live version-checked cores; ties are NAMED."""

    param = mixed_trace.params["fc.weight"]
    line = repr(param)
    assert "live v" in line
    assert "trainable" in line
    card = str(param)
    assert card.splitlines()[0] == line
    assert "used by" in card


def test_module_card_single_output_core(mixed_trace) -> None:
    """A Module with exactly one saved output site shows its core."""

    card = str(mixed_trace.modules["norm"])
    assert "out layernorm_1_3" in card
    assert "mean=" in card


def test_repr_never_raises_on_detached_records(mixed_trace) -> None:
    """Voice rule 10 + R52: detached records degrade, never raise."""

    import pickle

    layer = pickle.loads(pickle.dumps(mixed_trace["relu_1_2"]))
    assert "detached" in repr(layer)
    op = pickle.loads(pickle.dumps(mixed_trace["relu_1_2"].ops[0]))
    assert "detached" in repr(op)


def test_edited_mark_on_intervened_op() -> None:
    """Composition row: post-edit values carry (edited), never observed."""

    trace = tl.trace(
        _Mixed().eval(),
        torch.randn(2, 8),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    edited = trace["relu_1_2"].ops[0]
    assert "(edited)" in repr(edited)
    trace.cleanup()
