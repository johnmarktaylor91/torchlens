"""WandbSink under ``watch`` against the real wandb package (offline only).

With no descriptor and no settings grid, ``watch(..., to=WandbSink(run),
hist_every=...)`` must attach on the sink's cap-safe grid and deliver
histograms that fit wandb's 512-bucket cap; an explicit over-cap grid still
refuses at attach.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens.trackers as trk
from torchlens.observability._kernels import DEFAULT_DESCRIPTOR
from torchlens.trackers._errors import TrackersError

wandb = pytest.importorskip("wandb")

from _wandb_offline_log import history_rows  # noqa: E402

pytestmark = [pytest.mark.optional, pytest.mark.heavy]


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _train(session: trk.WatchSession, model: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
    for step in (10, 11):
        with session.step(step):
            opt.zero_grad(set_to_none=True)
            model(torch.randn(4, 8)).sum().backward()
            opt.step()


def test_default_watch_attaches_and_fits_the_cap(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("WANDB_MODE", "offline")
    model, opt = _mlp()
    run = wandb.init(dir=str(tmp_path), mode="offline")
    try:
        session = trk.watch(model, to=trk.WandbSink(run), optimizer=opt, hist_every=1)
        try:
            _train(session, model, opt)
        finally:
            session.close()
    finally:
        run.finish()
    rows = history_rows(tmp_path)
    hist_bins = [
        len(value)
        for row in rows
        for key, value in row.items()
        if key.endswith("/bins") and row.get(key.rsplit("/", 1)[0] + "/_type") == "histogram"
    ]
    expected_buckets = 2 * trk.WANDB_SAFE_DESCRIPTOR.bins_per_side + 1
    assert hist_bins, "no histogram reached the offline run"
    # Edges = buckets + 1; every histogram arrived un-rebinned on the safe grid.
    assert set(hist_bins) == {expected_buckets + 1}
    assert expected_buckets <= trk.WANDB_BUCKET_CAP
    steps = {row["_step"] for row in rows if any(k.endswith("/_type") for k in row)}
    assert steps <= {10, 11}
    print(f"\nhistogram rows={len(hist_bins)} buckets={expected_buckets} steps={sorted(steps)}")


def test_explicit_over_cap_grid_still_refuses(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("WANDB_MODE", "offline")
    model, opt = _mlp()
    run = wandb.init(dir=str(tmp_path), mode="offline")
    try:
        with pytest.raises(TrackersError) as info:
            trk.watch(
                model,
                to=trk.WandbSink(run),
                optimizer=opt,
                hist_every=1,
                descriptor=DEFAULT_DESCRIPTOR,
            )
        assert info.value.fields["code"] == "tracker_histogram_bucket_cap"
    finally:
        run.finish()
