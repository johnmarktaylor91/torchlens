"""Deep chains (compo memo stratum 3; F36 waves A-D).

One invariant at every hop: honesty and provenance facts SURVIVE the chain.
Each chain crosses at least three product transitions, and every
cross-member/ragged read is exercised AFTER a save/load round trip (foldB
D18 law 2). Toy models by design -- the chain topology, not the artifact
scale, is what these prove; the RG gallery owns realism.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "deep chains cross persistence hops (trace -> save -> load) by definition; every product is private and cleaned up"


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()


def _x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 4)


def test_chain_halted_settle_save_load_report_doors(tmp_path: Any) -> None:
    """Halt -> settle -> save -> load -> every report door still says HALTED."""

    import torchlens as tl

    halted = tl.trace(_model(), _x(), halt=tl.func("relu"))
    assert halted.outcome.status.name == "HALTED"
    path = tmp_path / "halted.tlspec"
    tl.save(halted, str(path))
    loaded = tl.load(str(path))
    assert loaded.outcome.status.name == "HALTED", "the halt fact fell off at the save/load hop"
    for door, text in (
        ("summary", str(loaded.summary())),
        ("explain", tl.report.explain(loaded)),
    ):
        assert "halt" in text.lower(), (
            f"the {door} door on the LOADED halted trace never says halted --"
            " the honesty fact was laundered at a chain hop"
        )
    payload = loaded.to_agent_json()
    assert "halted" in str(payload).lower()


def test_chain_structure_only_save_load_capability_doors(tmp_path: Any) -> None:
    """Structure-only -> save -> load -> hypothesis facts + refusals survive."""

    import torchlens as tl

    trace = tl.trace(_model(), _x(), capture=tl.options.CaptureOptions(structure_only=True))
    assert trace.structure_only
    path = tmp_path / "structure.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert loaded.structure_only, "structure_only flag fell off at the load hop"
    assert "hypothes" in tl.report.explain(loaded).lower(), (
        "the loaded structure-only trace's explain() no longer discloses the hypothesis ladder"
    )
    with pytest.raises(Exception) as excinfo:
        loaded.run(inputs=_x())
    code = (
        getattr(excinfo.value, "fields", {}).get("code")
        if hasattr(excinfo.value, "fields")
        else None
    )
    assert code is not None, (
        "run() on a LOADED structure-only capture refused without a stable"
        " code -- the capability chokepoint lost its typing across the hop"
    )


def test_chain_episode_ledger_survives_save_load_with_ragged_reads(tmp_path: Any) -> None:
    """Episode -> save -> load -> the per-step ledger reads back whole
    (foldB D18 law 2: ragged/cross-member reads AFTER a round trip)."""

    import torchlens as tl
    from tests.composition_expectations.test_proofnet_m1_product_verb import (
        build_episode_products,
    )

    products = build_episode_products()
    episode = products["episode"]
    ledger_before = episode.annotations["episode"]
    path = tmp_path / "episode.tlspec"
    tl.save(episode, str(path))
    loaded = tl.load(str(path))
    ledger_after = loaded.annotations["episode"]
    assert ledger_after["header"]["episode_ledger_version"] == 2
    assert ledger_after["header"]["capture_digest"] == ledger_before["header"]["capture_digest"], (
        "the minted capture digest changed across persistence"
    )
    statuses_before = [row["status"] for row in ledger_before["rows"]]
    statuses_after = [row["status"] for row in ledger_after["rows"]]
    assert statuses_after == statuses_before, (
        "per-step status rows (the ragged read) differ after the round trip"
    )
    outputs_after = [row["step_output"] for row in ledger_after["rows"]]
    assert all(outputs_after), "step-output evidence fell off at the load hop"


def test_chain_multipass_pass_qualified_edit_is_exact() -> None:
    """Recurrent multi-pass -> pass-qualified do() -> EXACT donor semantics:
    the addressed pass changes, the earlier pass does not, downstream
    recomputes (the pass-blind-donor regression class, toy-scale pin)."""

    import torchlens as tl

    class _Loop(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = torch.tanh(self.step(x))
            return x

    torch.manual_seed(0)
    model = _Loop().eval()
    example = torch.randn(1, 4)
    trace = tl.trace(model, example, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        tanh_layer = [label for label in trace.layer_labels if "tanh" in label][0]
        assert len(trace[tanh_layer].ops) == 3, "recurrence grouping lost the 3 passes"
        with pytest.raises(Exception) as bare_exc:
            trace.fork().do(tanh_layer, tl.zero_ablate())
        assert "pass" in str(bare_exc.value).lower(), (
            "a bare multi-pass label must refuse teaching the pass-qualified"
            " spellings (multipass_bare_label_ambiguous)"
        )
        fork = trace.fork()
        try:
            pass_one_before = trace[tanh_layer].ops[0].out.clone()
            fork.do(f"{tanh_layer}:2", tl.zero_ablate())
            assert torch.equal(fork[tanh_layer].ops[0].out, pass_one_before), (
                "editing pass 2 CHANGED pass 1 -- donor semantics corrupted"
            )
            assert torch.count_nonzero(fork[tanh_layer].ops[1].out) == 0, (
                "the addressed pass 2 was not edited"
            )
            # After the pass-2 OUTPUT is zeroed, exactly one step remains:
            # output == tanh(step(zeros)).
            with torch.no_grad():
                manual = torch.tanh(model.step(torch.zeros(1, 4)))
            assert torch.allclose(fork.output_ops[0].out, manual, atol=1e-6), (
                "downstream recompute after the pass-2 edit disagrees with the"
                " hand-coded continuation (independent oracle)"
            )
        finally:
            fork.cleanup()
    finally:
        trace.cleanup()


def test_chain_slice_selection_feeds_do_on_the_source() -> None:
    """Live -> between() slice -> __selection__ -> do() on a fork: the slice
    presenter composes back into the edit algebra."""

    import torchlens as tl

    trace = tl.trace(_model(), _x(), capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        labels = trace.layer_labels
        chain_slice = trace.between([labels[1]], [labels[-2]])
        selection = chain_slice.__selection__()
        fork = trace.fork()
        try:
            fork.do(selection.resolve(fork), tl.zero_ablate())
            assert torch.count_nonzero(fork[labels[1]].out) == 0, (
                "the slice-derived selection edit did not land on a member op"
            )
            assert fork.intervention_audit, "no audit row for the slice-lane edit"
        finally:
            fork.cleanup()
    finally:
        trace.cleanup()
