"""W051-TRACK / AUD-CODE 2.13: the emission ledger is keyed by sink OBJECT.

Two sinks of one class (two JSONL files: local + NFS) used to share ONE row
keyed by class name: a failure in either latched both, the healthy sink was
silently starved from that step on, and the counts merged.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.trackers as trk

pytestmark = pytest.mark.smoke


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


class _Flaky(trk.MemorySink):
    """A MemorySink that raises on its ``fail_at``-th scalar."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.seen = 0

    def emit_scalar(self, point):  # noqa: ANN001
        self.seen += 1
        if self.seen == self.fail_at:
            raise OSError("flaky filesystem")
        super().emit_scalar(point)


def test_same_class_sinks_get_separate_rows_and_the_healthy_one_is_never_starved() -> None:
    """One sink of a class failing mid-run leaves its sibling fully served."""

    model, opt = _mlp()
    failing, healthy = _Flaky(40), _Flaky(10**9)
    session = trk.watch(
        model, to=(failing, healthy), signals=("gradients",), optimizer=opt, every=1
    )
    with session:
        for step in range(5):
            with session.step(step):
                opt.zero_grad(set_to_none=True)
                model(torch.randn(4, 8)).sum().backward()
                opt.step()
    report = session.close()
    rows = {row["sink"]: row for row in report.sink_rows}
    assert set(rows) == {"_Flaky", "_Flaky#2"}
    assert rows["_Flaky"]["failed"] is True
    assert rows["_Flaky#2"]["failed"] is False
    # The healthy sink saw EVERY step and its own count, not the merged one.
    assert sorted({p.step for p in healthy.scalars if p.tag.startswith("gradients/")}) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert rows["_Flaky#2"]["emitted_scalars"] == len(healthy.scalars)
    assert rows["_Flaky"]["emitted_scalars"] == len(failing.scalars)
    assert rows["_Flaky#2"]["emitted_data_points"] > 0
    assert any("_Flaky latched failed" in skip for skip in report.named_skips)


def test_ledger_rows_are_identity_keyed() -> None:
    """The ledger keys on object identity; display names disambiguate."""

    ledger = trk._protocol.EmissionLedger()
    first, second = trk.MemorySink(), trk.MemorySink()
    assert ledger.row(first) is ledger.row(first)
    assert ledger.row(first) is not ledger.row(second)
    assert ledger.row(first).name == "MemorySink"
    assert ledger.row(second).name == "MemorySink#2"
    assert set(ledger.sinks) == {id(first), id(second)}


def test_same_sink_object_listed_twice_refuses_typed() -> None:
    """One object twice would double-emit while its row counted once."""

    model, opt = _mlp()
    sink = trk.MemorySink()
    with pytest.raises(trk.SinkProtocolError) as info:
        trk.watch(model, to=(sink, sink), optimizer=opt)
    assert info.value.fields["code"] == "tracker_sink_duplicate"
    assert sink.scalars == []
