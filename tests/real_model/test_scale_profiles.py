"""Scale-profile budgets hold under their ceilings; SP1 runs under its own (U40.3).

The ceilings (10 CPU-minutes a night, 30 a week) and the per-profile budgets
live in ``tests/real_model/scale_profiles.py``. The SP1 quantities are
closed forms of the fixture's dimensions, never TorchLens's own report.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.real_model.scale_profiles import (
    NIGHTLY_CEILING_CPU_MIN,
    SCALE_PROFILES,
    SCALE_PROFILES_BY_ID,
    STATUSES,
    VENUES,
    WEEKLY_CEILING_CPU_MIN,
    nightly_cpu_minutes,
    weekly_cpu_minutes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.real_model]


def test_ceilings_are_the_ruled_numbers() -> None:
    assert NIGHTLY_CEILING_CPU_MIN == 10.0
    assert WEEKLY_CEILING_CPU_MIN == 30.0


def test_budgets_fit_under_both_ceilings() -> None:
    assert nightly_cpu_minutes() <= NIGHTLY_CEILING_CPU_MIN, (
        f"nightly scale profiles budget {nightly_cpu_minutes()} CPU-min a night,"
        f" over the {NIGHTLY_CEILING_CPU_MIN} ceiling"
    )
    assert weekly_cpu_minutes() <= WEEKLY_CEILING_CPU_MIN, (
        f"the week's scale load is {weekly_cpu_minutes()} CPU-min (weekly profiles"
        f" plus seven nightly runs), over the {WEEKLY_CEILING_CPU_MIN} ceiling"
    )


def test_ceiling_check_refuses_an_overweight_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANT: the week's arithmetic counts a nightly profile seven times."""

    from tests.real_model import scale_profiles as module

    heavy = module.ScaleProfile(**{**vars(SCALE_PROFILES[0]), "profile_id": "SP9-plant"})
    baseline_week = weekly_cpu_minutes()
    monkeypatch.setattr(module, "SCALE_PROFILES", (*SCALE_PROFILES, heavy))
    assert module.weekly_cpu_minutes() == pytest.approx(baseline_week + 7 * heavy.cpu_minutes)
    monkeypatch.setattr(module, "SCALE_PROFILES", (*SCALE_PROFILES, *(heavy,) * 10))
    assert module.nightly_cpu_minutes() > NIGHTLY_CEILING_CPU_MIN


@pytest.mark.parametrize("profile", SCALE_PROFILES, ids=lambda p: p.profile_id)
def test_every_profile_is_fully_named(profile) -> None:
    assert profile.profile_id.startswith("SP") and profile.discriminating_property
    assert profile.venue in VENUES and profile.status in STATUSES
    assert profile.cpu_minutes > 0 and profile.wall_budget_s > 0 and profile.rss_budget_mib > 0
    assert profile.owner.strip(), "an unowned profile says 'unassigned', never blank"
    if profile.status == "built":
        path, _, name = profile.test_node.partition("::")
        source = (REPO_ROOT / path).read_text()
        assert f"def {name}(" in source, f"{profile.profile_id}: {profile.test_node} is missing"
    else:
        assert profile.test_node == ""


def _current_rss_mib() -> float:
    """Current resident set size; a pytest worker's lifetime peak is not this run's."""

    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
    except OSError:
        pytest.skip("SP1's RSS budget reads /proc/self/statm (Linux-only venue)")
    return pages * resource.getpagesize() / (1 << 20)


def _real_retained_bytes(trace) -> int:
    """Physical storage bytes of retained payloads, each storage once."""

    seen: set[tuple[int, int]] = set()
    total = 0
    for op in trace.ops:
        payload = op.out
        if not isinstance(payload, torch.Tensor):
            continue
        storage = payload.untyped_storage()
        key = (storage.data_ptr(), int(storage.nbytes()))
        if key not in seen:
            seen.add(key)
            total += int(storage.nbytes())
    return total


def test_sp1_wide_output_linear_quantities() -> None:
    """SP1: the wide head's quantities equal their closed forms, within budget."""

    import torchlens as tl

    profile = SCALE_PROFILES_BY_ID["SP1-wide-output-linear"]
    batch, d_in, d_out, fp32 = 4, 8, 50_000, 4
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    rss_before = _current_rss_mib()
    wall_before = time.monotonic()
    torch.manual_seed(0)
    model = nn.Linear(d_in, d_out).eval()
    trace = tl.trace(model, torch.randn(batch, d_in))
    try:
        op = next(o for o in trace.ops if o.layer_label == "linear_1_1")
        n_params = d_in * d_out + d_out
        macs = batch * d_in * d_out
        assert int(trace.num_params) == n_params
        assert int(op.param_memory) == n_params * fp32
        assert int(op.macs_forward) == macs
        # fma2 convention: 2 x MACs plus the bias adds (batch x d_out).
        assert int(op.flops_forward) == 2 * macs + batch * d_out
        assert int(op.bytes_read) == (batch * d_in + n_params) * fp32
        assert int(op.bytes_written) == batch * d_out * fp32
        assert int(op.activation_memory) == batch * d_out * fp32
        assert int(trace.saved_activation_memory) == _real_retained_bytes(trace)
        rss_growth_mib = _current_rss_mib() - rss_before
    finally:
        trace.cleanup()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (usage_after.ru_utime - usage_before.ru_utime) + (
        usage_after.ru_stime - usage_before.ru_stime
    )
    assert cpu_s <= profile.cpu_minutes * 60, f"SP1 used {cpu_s:.1f} CPU-s"
    assert time.monotonic() - wall_before <= profile.wall_budget_s
    assert rss_growth_mib <= profile.rss_budget_mib, f"SP1 grew RSS by {rss_growth_mib:.0f} MiB"
