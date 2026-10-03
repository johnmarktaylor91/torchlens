"""Zero-match disclosure for the halt selector slot and preview backends.

``halt=`` was the ONE selector slot in the public surface outside the
zero-match disclosure family: a typo'd op/module name silently ran the FULL
forward (spending the memory/latency the halt was meant to bound) and handed
back the model's real outputs where the caller expected a frontier, with
``outcome.status == complete`` as the only implicit signal. The tf/mlx
intervene zero-match legs are the preview half of the torch-side 7969aca8
fix (fired accounting written but never read on tf; no counter at all on
mlx).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl


def _model() -> nn.Module:
    """Tiny two-op model."""

    return nn.Sequential(nn.Linear(3, 3), nn.ReLU())


def test_halt_zero_match_warns_on_completed_capture() -> None:
    """A halt selector that never fires discloses instead of staying silent."""

    with pytest.warns(UserWarning, match="halt selector .* matched zero sites"):
        log = tl.trace(_model(), torch.randn(2, 3), halt=tl.func("nosuchopzzz"))
    assert log.outcome.status.name == "COMPLETE"


def test_halt_that_fires_does_not_warn() -> None:
    """A genuinely-halting selector emits no zero-match warning."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(_model(), torch.randn(2, 3), halt=tl.func("linear"))
    assert log.outcome.status.name == "HALTED"
    assert not any("matched zero sites" in str(item.message) for item in caught)


def test_value_dependent_halt_callable_is_not_judged() -> None:
    """A non-selector halt callable legitimately never firing stays silent."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(_model(), torch.randn(2, 3), halt=lambda ctx: False)
    assert not any(
        "halt selector" in str(item.message) and "matched zero" in str(item.message)
        for item in caught
    )


@pytest.mark.smoke
def test_tf_zero_fire_site_warns_in_reachability_audit() -> None:
    """A planned tf site with zero fires warns at the post-forward audit.

    ``fired_site_labels`` was written by the fire path and read by NOTHING;
    a selector matching zero captured ops left the audit's ``unreachable``
    list empty and returned cleanly, and module sites got no audit at all.
    The audit itself is import-safe without TensorFlow, so this leg runs on
    torch-only hosts.
    """

    from torchlens.backends.tf.interventions import (
        TFInterventionPlan,
        TFInterventionSite,
        audit_tf_site_reachability,
    )

    fired_site = TFInterventionSite(
        plan_id="site_0",
        predicate=lambda ctx: None,
        selector=lambda ctx: False,
        decision=None,
        hook=lambda tensor: tensor,
        level="op",
    )
    silent_module_site = TFInterventionSite(
        plan_id="site_1",
        predicate=lambda ctx: None,
        selector=lambda ctx: False,
        decision=None,
        hook=lambda tensor: tensor,
        level="module",
    )
    plan = TFInterventionPlan(
        sites=(fired_site, silent_module_site),
        op_sites=(fired_site,),
        module_sites=(silent_module_site,),
    )
    plan.fired_site_labels.append(("site_0", "relu_1_1"))
    session = SimpleNamespace(events=SimpleNamespace(op_events=[]))

    with pytest.warns(UserWarning, match="site_1 fired at zero sites"):
        audit_tf_site_reachability(plan, session)

    # Every planned site fired: the audit stays silent.
    plan.fired_site_labels.append(("site_1", "module_1"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_tf_site_reachability(plan, session)
    assert not any("fired at zero sites" in str(item.message) for item in caught)


def test_zero_match_selectors_leave_a_persisted_trace_record(tmp_path) -> None:
    """A requested-but-unfired selector is recorded ON the trace (B3R4-R15-1).

    The transient UserWarning was the ONLY disclosure: the returned (and
    saved) Trace was indistinguishable from one where no intervention was
    requested, so an ablation sweep with one typo'd layer name read as
    "this layer does not matter". The zero-match fact now persists in
    ``trace.annotations["unmatched_capture_selectors"]``.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        log = tl.trace(
            _model(),
            torch.randn(2, 3),
            intervene=tl.when(tl.func("nosuchopzzz"), tl.zero_ablate()),
        )

    records = log.annotations.get("unmatched_capture_selectors")
    assert records, "zero-match intervention left no trace-side record"
    slots = {record["slot"] for record in records}
    assert "intervene" in slots
    entry = next(record for record in records if record["slot"] == "intervene")
    assert "nosuchopzzz" in entry["selector"]

    # The fact survives save/load: a re-analysis of the artifact can see it.
    path = tmp_path / "zero_match.tlspec"
    tl.save(log, str(path))
    loaded = tl.load(str(path))
    loaded_records = loaded.annotations.get("unmatched_capture_selectors")
    assert loaded_records
    assert any(record["slot"] == "intervene" for record in loaded_records)

    # Control: a FIRING intervention leaves no zero-match record.
    fired = tl.trace(
        _model(),
        torch.randn(2, 3),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    assert not (fired.annotations or {}).get("unmatched_capture_selectors")


def test_zero_match_halt_and_save_record_their_slots() -> None:
    """The halt and save selector slots persist the same zero-match fact."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        log = tl.trace(
            _model(),
            torch.randn(2, 3),
            save=tl.func("nosuchopzzz"),
            halt=tl.func("nosuchopyyy"),
        )
    records = log.annotations.get("unmatched_capture_selectors") or ()
    slots = {record["slot"] for record in records}
    assert slots == {"save", "halt"}


def test_secret_bearing_save_warns_the_saver(tmp_path) -> None:
    """R62: the save that embeds custom attributes is no longer silent.

    Fail-before: a model carrying ``self.api_token = ...`` saved with zero
    terminal output; the only disclosure landed INSIDE the artifact being
    handed out. Structural constructor echoes (in_features, ...) alone never
    trigger the warning, and include_custom_attributes=False silences it.
    """

    import warnings

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens.errors import TorchLensWarning

    class _Leaky(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.api_token = "sk-not-a-real-token"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    log = tl.trace(_Leaky(), torch.ones(1, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.save(log, tmp_path / "leaky.tlspec")
    matching = [
        w
        for w in caught
        if issubclass(w.category, TorchLensWarning) and "custom module attribute" in str(w.message)
    ]
    assert len(matching) == 1
    assert "api_token" in str(matching[0].message)

    clean = tl.trace(nn.Sequential(nn.Linear(3, 2)), torch.ones(1, 3))
    with warnings.catch_warnings(record=True) as caught_clean:
        warnings.simplefilter("always")
        tl.save(clean, tmp_path / "clean.tlspec")
    assert not [w for w in caught_clean if "custom module attribute" in str(w.message)]

    with warnings.catch_warnings(record=True) as caught_optout:
        warnings.simplefilter("always")
        tl.save(log, tmp_path / "optout.tlspec", include_custom_attributes=False)
    assert not [w for w in caught_optout if "custom module attribute" in str(w.message)]


def test_persisted_selector_repr_relativizes_absolute_paths() -> None:
    """R62 LOW: the zero-match ledger goes through the path-relativization belt."""

    import warnings

    import torch
    from torch import nn

    import torchlens as tl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(
            nn.Sequential(nn.Linear(3, 2)),
            torch.ones(1, 3),
            save=tl.func("/home/someone/secret/cfg.json"),
        )
    entries = log.annotations.get("unmatched_capture_selectors", [])
    assert entries, "expected a zero-match ledger entry"
    assert "/home/someone" not in entries[0]["selector"]
    assert "cfg.json" in entries[0]["selector"]


def test_layers_to_save_zero_match_warns_and_persists(tmp_path) -> None:
    """grind-r6 b3 R15 (opus MED, probe-proven; residual of B3R5-R15-1).

    ``layers_to_save`` resolves through its own predicate machinery, never a
    ``BaseSelector``, so a typo'd layer name sat OUTSIDE the zero-match
    disclosure family: the capture retained only the always-retained output
    tail and disclosed nothing, live and in the artifact -- while the same
    typo in the ``save=`` slot warned and persisted a record. This is the
    highest-traffic instance of the cc2cabbb class ("an ablation sweep with
    one typo'd layer name concluded 'this layer does not matter'").
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(
            _model(),
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(layers_to_save=["nosuchlayerzzz"]),
        )
    assert any("matched zero layers" in str(w.message) for w in caught), (
        "typo'd layers_to_save produced no zero-match warning"
    )
    entries = log.annotations.get("unmatched_capture_selectors", [])
    assert any(entry["slot"] == "layers_to_save" for entry in entries), (
        f"no persisted layers_to_save zero-match record: {entries}"
    )

    # The artifact carries the record too.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.save(log, tmp_path / "zm.tlspec")
        loaded = tl.load(tmp_path / "zm.tlspec")
    loaded_entries = loaded.annotations.get("unmatched_capture_selectors", [])
    assert any(entry["slot"] == "layers_to_save" for entry in loaded_entries)

    # Control: the correct spelling selects and stays silent.
    with warnings.catch_warnings(record=True) as clean:
        warnings.simplefilter("always")
        control = tl.trace(
            _model(),
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(layers_to_save=["linear"]),
        )
    assert not any("matched zero layers" in str(w.message) for w in clean)
    assert not control.annotations.get("unmatched_capture_selectors")
