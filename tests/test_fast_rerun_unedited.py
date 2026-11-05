"""``trace.run(model, x)`` on an unedited selective-save trace takes the guarded fast engine.

Contract pinned here: a trace captured with a ``save=`` selection and nothing
staged reruns through the guarded fast engine (``last_run["engine"] ==
"guarded_fast"``, ``fast_refused`` None), exactly: every saved site equals a
fresh ``tl.trace`` of the new input, the source is untouched and no hook stays
on the model. A changed call structure falls back to the capture rerun with the
refusing guard in ``last_run["fast_refused"]``; a save-everything trace with
nothing staged keeps the capture rerun. A tamper test proves the parity check
fails on a fast path that leaves stale values, and the run-time ``save=``
refusals name ``save=``, not ``until=``.
"""

from __future__ import annotations

import dataclasses
import enum
import weakref
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_rerun_state import copy_rerun_value
from torchlens._fast_live_steer import legacy_rerun_tries_fast_engine, saved_a_selection
from torchlens.errors import InvalidArgumentError, RunCapabilityUnavailableError
from torchlens.intervention.errors import ControlFlowDivergenceWarning

_DIM = 8


class _Net(nn.Module):
    """Stem, two residual blocks and a head."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.stem = nn.Linear(_DIM, _DIM)
        self.block1 = nn.Sequential(nn.LayerNorm(_DIM), nn.Linear(_DIM, _DIM), nn.GELU())
        self.block2 = nn.Sequential(nn.LayerNorm(_DIM), nn.Linear(_DIM, _DIM), nn.GELU())
        self.head = nn.Linear(_DIM, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stem, the blocks with residuals, and the head."""

        h = self.stem(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


class _BranchNet(nn.Module):
    """A model whose call structure depends on the input sign."""

    def __init__(self) -> None:
        """Build both arms and the head."""

        super().__init__()
        self.stem = nn.Linear(_DIM, _DIM)
        self.pos = nn.Linear(_DIM, _DIM)
        self.neg = nn.Sequential(nn.Linear(_DIM, _DIM), nn.ReLU(), nn.Linear(_DIM, _DIM))
        self.head = nn.Linear(_DIM, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the ``pos`` arm on a positive-sum input, the ``neg`` arm otherwise."""

        h = self.stem(x)
        h = self.pos(h) if bool(x.sum() > 0) else self.neg(h)
        return self.head(h)


def _x(seed: int, *, sign: float = 1.0) -> torch.Tensor:
    """Return a deterministic batch; ``sign`` fixes the sign of its sum."""

    x = torch.randn(3, _DIM, generator=torch.Generator().manual_seed(seed))
    return sign * x.abs()


def _saved(trace: Any) -> dict[str, torch.Tensor]:
    """Return the saved activation of every non-boundary saved op by label."""

    return {
        op.label: op.out
        for op in trace._layer_list
        if bool(getattr(op, "has_saved_activation", False)) and op.type not in ("input", "output")
    }


def _assert_sites_match(result: Any, fresh: Any) -> None:
    """Assert every saved site of ``result`` equals the same site of ``fresh``, exactly."""

    got, want = _saved(result), _saved(fresh)
    assert got, "the result saved no site"
    assert set(got) == set(want), (sorted(got), sorted(want))
    for label, value in got.items():
        assert torch.equal(value, want[label]), label
    assert torch.equal(result.output, fresh.output)


def _hook_count(model: nn.Module) -> int:
    """Count the forward hooks and pre-hooks on every module of ``model``."""

    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


_SELECTION = tl.module("block1") | tl.module("head")


@pytest.mark.parametrize("one_argument", [False, True], ids=["model_and_x", "x_only"])
def test_unedited_selective_save_rerun_is_fast_and_exact(one_argument: bool) -> None:
    """An unedited selective-save ``trace.run`` is the guarded fast engine, exact at every site."""

    torch.manual_seed(0)
    model = _Net().eval()
    x1, x2 = _x(1), _x(2)
    source = tl.trace(model, x1, save=_SELECTION)
    assert source.last_run is None
    source_sites = {label: value.clone() for label, value in _saved(source).items()}
    hooks_before = _hook_count(model)

    result = source.run(x2) if one_argument else source.run(model, x2)

    assert result is not source
    assert result.last_run["engine"] == "guarded_fast", result.last_run.get("fast_refused")
    assert result.last_run["fast_refused"] is None
    assert result.last_run["hooks"] == 0
    fresh = tl.trace(model, x2, save=_SELECTION)
    _assert_sites_match(result, fresh)
    with torch.no_grad():
        assert torch.equal(result.output, model(x2))
    assert result.input is x2
    # The source is unchanged and no hook stays on the model between calls.
    assert source.last_run is None
    for label, value in _saved(source).items():
        assert torch.equal(value, source_sites[label]), label
    assert _hook_count(model) == hooks_before


def test_repeated_unedited_reruns_stay_fast_and_exact() -> None:
    """Each rerun of a fast result is itself fast and exact on its own input."""

    torch.manual_seed(0)
    model = _Net().eval()
    trace = tl.trace(model, _x(1), save=_SELECTION)
    for seed in range(2, 6):
        x = _x(seed)
        trace = trace.run(model, x)
        assert trace.last_run["engine"] == "guarded_fast", trace.last_run.get("fast_refused")
        _assert_sites_match(trace, tl.trace(model, x, save=_SELECTION))


def test_stale_fast_values_fail_the_parity_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tamper: a fast engine that never writes a payload leaves stale sites the check rejects."""

    import torchlens._fast_run as fast_run

    torch.manual_seed(0)
    model = _Net().eval()
    source = tl.trace(model, _x(1), save=_SELECTION)
    x2 = _x(2)
    monkeypatch.setattr(fast_run, "save_payload_paused", lambda session, op, value: None)

    result = source.run(model, x2)

    assert result.last_run["engine"] == "guarded_fast"
    with pytest.raises(AssertionError):
        _assert_sites_match(result, tl.trace(model, x2, save=_SELECTION))


def test_changed_call_structure_falls_back_to_the_capture_rerun() -> None:
    """A different taken path refuses the fast engine; the recapture is exact and says why."""

    torch.manual_seed(0)
    model = _BranchNet().eval()
    selection = tl.module("stem") | tl.module("head")
    source = tl.trace(model, _x(1, sign=1.0), save=selection)
    x_neg = _x(2, sign=-1.0)

    # The recapture's own detector reports the changed graph, as every structural rerun does.
    with pytest.warns(ControlFlowDivergenceWarning):
        result = source.run(model, x_neg)

    assert result.last_run["engine"] == "rerun"
    refused = result.last_run["fast_refused"]
    assert isinstance(refused, str) and refused.endswith(":fast_live_call_fingerprint"), refused
    _assert_sites_match(result, tl.trace(model, x_neg, save=selection))
    # The same arm as the capture is fast again.
    again = result.run(model, _x(3, sign=-1.0))
    assert again.last_run["engine"] == "guarded_fast", again.last_run.get("fast_refused")


def test_save_everything_rerun_keeps_the_capture_engine() -> None:
    """A trace that saved every op, nothing staged, is re-recorded by the capture engine."""

    torch.manual_seed(0)
    model = _Net().eval()
    source = tl.trace(model, _x(1))
    assert not saved_a_selection(source)
    assert not legacy_rerun_tries_fast_engine(source)
    x2 = _x(2)

    result = source.run(model, x2)

    assert result.last_run["engine"] == "rerun"
    assert result.last_run["fast_refused"] is None
    _assert_sites_match(result, tl.trace(model, x2))


def test_selection_eligibility_reads_the_saved_ops() -> None:
    """The eligibility helper sees a selective save and a save-everything capture apart."""

    model = _Net().eval()
    assert saved_a_selection(tl.trace(model, _x(1), save=_SELECTION))
    assert legacy_rerun_tries_fast_engine(tl.trace(model, _x(1), save=_SELECTION))
    assert not saved_a_selection(tl.trace(model, _x(1)))


def test_run_time_save_refusals_name_save_not_until() -> None:
    """A predicate or empty run-time ``save=`` is refused under its own keyword."""

    model = _Net().eval()
    trace = tl.trace(model, _x(1), save=_SELECTION)
    with pytest.raises(RunCapabilityUnavailableError) as predicate:
        trace.run_transaction(inputs=_x(2), save=tl.module("block2"))
    assert "save=" in str(predicate.value)
    assert "until=" not in str(predicate.value)
    assert predicate.value.fields["detection_stage"] == "predicate_surface_pending"
    with pytest.raises(InvalidArgumentError) as empty:
        trace.run_transaction(inputs=_x(2), save=[])
    assert empty.value.fields["argument"] == "save"
    assert "save=" in str(empty.value)
    with pytest.raises(RunCapabilityUnavailableError) as until:
        trace.run_transaction(inputs=_x(2), until=tl.module("block2"))
    assert "until=" in str(until.value)


class _Kind(enum.Enum):
    """A two-member enum for the copy test."""

    A = 1
    B = 2


def _helper() -> None:
    """A function value for the copy test."""


@dataclasses.dataclass(frozen=True)
class _FrozenRecord:
    """A frozen record for the copy test."""

    name: str
    sizes: tuple[int, ...]


@dataclasses.dataclass
class _MutableRecord:
    """A mutable record for the copy test."""

    name: str


def test_copy_rerun_value_hands_atomic_values_over_and_rebuilds_sets() -> None:
    """Values a deep copy returns unchanged pass through; sets are rebuilt, same type."""

    ref = weakref.ref(_Net())
    for value in (_Kind.A, int, _helper, len, range(3), slice(1, 2), Ellipsis, NotImplemented, ref):
        assert copy_rerun_value(None, value) is value, value
    tensor = torch.zeros(2)
    assert copy_rerun_value(None, tensor) is tensor
    plain = {1, "a", _Kind.B}
    copied = copy_rerun_value(None, plain)
    assert copied == plain and copied is not plain and type(copied) is set
    frozen = frozenset({(1, 2), 3})
    assert copy_rerun_value(None, frozen) == frozen
    assert type(copy_rerun_value(None, frozen)) is frozenset
    nested = {"k": [1, {2}]}
    copied_nested = copy_rerun_value(None, nested)
    assert copied_nested == nested and copied_nested["k"] is not nested["k"]
    frozen_record = _FrozenRecord("a", (1, 2))
    assert copy_rerun_value(None, frozen_record) is frozen_record
    mutable_record = _MutableRecord("a")
    copied_record = copy_rerun_value(None, mutable_record)
    assert copied_record == mutable_record and copied_record is not mutable_record
    from torchlens._quantities import Bytes

    size = Bytes(12)
    assert copy_rerun_value(None, size) is size
