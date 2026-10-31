"""The episode-option composition table (foldA F36 delta, verbatim mandate).

Every listed (episode x option) combination is ORTHOGONAL (both features
serve their full contract side by side), COUPLED (the composition mints its
own evidence: attribution, disclosure rows, digests), or TYPED-REFUSED
(stable code, teaching message). UNLISTED cells refuse: the closure gate
fails on any composition axis value the table does not classify.

Both mandated episode/intervention cells are recorded:

- F40 arm -- intervention during an episode REFUSED before execution. That
  arm was SUPERSEDED by F42's evidence gate for capture-time coupling; the
  refuse-before-execution posture survives at the replay door, recorded here
  as EP-REPLAY-COUPLED (``episode_coupled_replay_underivable`` fires before
  any replay computation).
- F42 arm -- episode= x intervene= runs COUPLED after its evidence gate:
  the ledger header carries ``intervention_digest``, rows carry measured
  ``fire_count``, and ``trace.episode_coupling`` CONSUMES the capture digest.

C07-A grammar gates (ledger version quarantine, bundle relation grammar v2)
ride their owning lanes' suites; this table owns the option-composition seam.
Ground truth probed live on the merged tree, 2026-08-30.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.composition_expectations.test_proofnet_m1_product_verb import (
    build_episode_products,
)

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "episode-option cells construct per-cell captures by definition (each cell IS a capture-option composition); toy root, module-scoped rig"

ORTHOGONAL = "ORTHOGONAL"
COUPLED = "COUPLED"
TYPED_REFUSED = "TYPED-REFUSED"
UNTYPED_REFUSED = "UNTYPED-REFUSED"  # ledgered gap, never a steady state

#: The closed cell list: (cell_id, classification, refusal code or "").
#: Cell ids are the option axis values; the closure test derives the tested
#: axis set from THIS table, so an axis value tested below without a row
#: here (or vice versa) is red.
EPISODE_OPTION_TABLE: tuple[tuple[str, str, str], ...] = (
    ("intervene_capture", COUPLED, ""),
    ("coupled_replay_run", TYPED_REFUSED, "episode_coupled_replay_underivable"),
    ("halt_predicate", COUPLED, ""),
    ("save_predicate_retaining_source", ORTHOGONAL, ""),
    ("storage_disk_digest_source_dropped", TYPED_REFUSED, "episode_declaration_invalid"),
    ("structure_only", TYPED_REFUSED, "structure_only_episode_unsupported"),
    ("raise_on_nan", ORTHOGONAL, ""),
    ("log_injections", ORTHOGONAL, ""),
    ("record_entry", UNTYPED_REFUSED, "TypeError"),
    ("module_root_without_stepped_module", TYPED_REFUSED, "episode_declaration_invalid"),
    ("forced_tokens_on_float_root", TYPED_REFUSED, "episode_declaration_invalid"),
    ("bound_method_default_owner", COUPLED, ""),
    ("declared_steps_never_run", TYPED_REFUSED, "episode_ledger_incoherent"),
)


class _StepModule(torch.nn.Module):
    """One toy step for episode-option cells."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _Root(torch.nn.Module):
    """Module root looping its stepped submodule three times."""

    def __init__(self) -> None:
        super().__init__()
        self.step = _StepModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.step(x)
        return x


class _SelfSteppingGenerator(torch.nn.Module):
    """Owner whose ``generate`` steps ITSELF (the F41 default-owner cell)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self(x)
        return x


@pytest.fixture(scope="module")
def episode_rig() -> dict[str, Any]:
    """Shared toy root + spec factory for the option cells."""

    import torchlens as tl

    torch.manual_seed(0)
    root = _Root().eval()

    def spec(**overrides: Any) -> Any:
        base: dict[str, Any] = {
            "stepped_module": root.step,
            "n_steps": 3,
            "step_output_kind": "digest",
        }
        base.update(overrides)
        return tl.options.EpisodeSpec(**base)

    return {"root": root, "example": torch.randn(1, 4), "spec": spec}


def _expect(cell_id: str) -> tuple[str, str]:
    rows = [row for row in EPISODE_OPTION_TABLE if row[0] == cell_id]
    assert len(rows) == 1, f"cell {cell_id!r} must appear exactly once in the table"
    return rows[0][1], rows[0][2]


def _assert_refuses(cell_id: str, invoke: Any) -> None:
    """Drive a refusal cell: right state, right code (foldB D18 law 3)."""

    classification, code = _expect(cell_id)
    with pytest.raises(Exception) as excinfo:
        invoke()
    exc = excinfo.value
    got_code = getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
    if classification == TYPED_REFUSED:
        assert got_code == code, (
            f"cell {cell_id}: refusal carries code={got_code!r}, table says {code!r}"
        )
    elif classification == UNTYPED_REFUSED:
        assert type(exc).__name__ == code and not got_code, (
            f"cell {cell_id}: the ledgered untyped refusal changed"
            f" ({type(exc).__name__}, code={got_code!r}) -- if a lane typed it,"
            " promote the cell to TYPED-REFUSED deliberately"
        )
    else:  # pragma: no cover - table drift guard
        raise AssertionError(f"cell {cell_id} declared {classification}, but it raised")


def test_episode_x_intervene_runs_coupled_with_evidence(episode_rig: dict[str, Any]) -> None:
    """The F42 cell: COUPLED after the evidence gate, never silent.

    Evidence asserted, not inferred from absence: the ledger header mints
    ``intervention_digest``, at least one started row measured a nonzero
    ``fire_count``, and ``episode_coupling`` consumes the capture digest.
    """

    classification, _ = _expect("intervene_capture")
    assert classification == COUPLED
    products = build_episode_products()
    coupled = products["coupled"]
    ledger = coupled.annotations["episode"]
    header = ledger["header"]
    assert header["intervention_digest"], (
        "COUPLED capture minted no intervention_digest -- the F42 evidence gate"
        " is not stamping the episode ledger header"
    )
    assert header["fidelity_basis"] == "perturbed", (
        f"a fired edit must outrank other bases; header says {header['fidelity_basis']!r}"
    )
    coupling = coupled.episode_coupling
    assert coupling is not None
    fire_counts = [row["fire_count"] for row in ledger["rows"] if row["fire_count"] is not None]
    assert fire_counts and any(count > 0 for count in fire_counts), (
        f"coupled episode measured fire counts {fire_counts}: the tanh ablation"
        " fired on no step row, so attribution evidence is absent"
    )


def test_episode_coupled_replay_refuses_before_execution(episode_rig: dict[str, Any]) -> None:
    """The surviving F40 posture: coupled replay refuses BEFORE running."""

    products = build_episode_products()
    coupled = products["coupled"]
    _assert_refuses("coupled_replay_run", lambda: coupled.run(inputs=products["example"]))


def test_episode_x_halt_is_coupled_via_step_status_rows(episode_rig: dict[str, Any]) -> None:
    """A mid-episode halt lands in the per-step status ledger, disclosed."""

    import torchlens as tl

    rig = episode_rig
    trace = tl.trace(rig["root"], rig["example"], episode=rig["spec"](), halt=tl.func("tanh"))
    classification, _ = _expect("halt_predicate")
    assert classification in {ORTHOGONAL, COUPLED}
    assert trace.outcome.status.name == "HALTED"
    ledger = trace.annotations["episode"]
    statuses = [row["status"] for row in ledger["rows"]]
    assert statuses, "halted episode carries no per-step rows"
    assert any(status in {"interrupted", "absent"} for status in statuses), (
        f"halt fired inside the episode but every row reads {statuses}: the"
        " coupling (halt -> step status) is not being recorded"
    )


def test_episode_x_save_predicate_retaining_source_is_orthogonal(
    episode_rig: dict[str, Any],
) -> None:
    """save= that retains the step-output source composes cleanly."""

    import torchlens as tl

    rig = episode_rig
    trace = tl.trace(rig["root"], rig["example"], episode=rig["spec"](), save=tl.func("tanh"))
    classification, _ = _expect("save_predicate_retaining_source")
    assert classification == ORTHOGONAL
    assert trace.outcome.status.name == "COMPLETE"
    assert trace.annotations["episode"] is not None
    tanh_labels = [label for label in trace.layer_labels if "tanh" in label]
    assert tanh_labels, "no tanh layer captured"
    first_pass = trace[tanh_labels[0]].ops[0]
    assert first_pass.out is not None, "the save predicate retained nothing"


def test_episode_x_disk_storage_with_dropped_source_refuses(
    episode_rig: dict[str, Any], tmp_path: Any
) -> None:
    """Declared digest evidence + a save policy dropping the source: refuse."""

    import torchlens as tl

    rig = episode_rig
    _assert_refuses(
        "storage_disk_digest_source_dropped",
        lambda: tl.trace(
            rig["root"],
            rig["example"],
            episode=rig["spec"](),
            storage=tl.to_disk(str(tmp_path / "episode.tlspec")),
        ),
    )


def test_episode_x_structure_only_refuses_typed(episode_rig: dict[str, Any]) -> None:
    """Structure-only episodes are out of scope: typed, teaching."""

    import torchlens as tl

    rig = episode_rig
    _assert_refuses(
        "structure_only",
        lambda: tl.trace(
            rig["root"],
            rig["example"],
            episode=rig["spec"](),
            capture=tl.options.CaptureOptions(structure_only=True),
        ),
    )


def test_episode_x_raise_on_nan_is_orthogonal(episode_rig: dict[str, Any]) -> None:
    """The nonfinite tripwire arms per step without disturbing the ledger."""

    import torchlens as tl

    rig = episode_rig
    trace = tl.trace(
        rig["root"],
        rig["example"],
        episode=rig["spec"](),
        capture=tl.options.CaptureOptions(raise_on_nan=True),
    )
    classification, _ = _expect("raise_on_nan")
    assert classification in {ORTHOGONAL, COUPLED}
    assert trace.outcome.status.name == "COMPLETE"
    rows = trace.annotations["episode"]["rows"]
    assert rows and all(row["status"] == "complete" for row in rows)


def test_episode_x_log_injections_is_orthogonal(episode_rig: dict[str, Any]) -> None:
    """The F44 injection log coexists with episode capture."""

    import torchlens as tl

    rig = episode_rig
    trace = tl.trace(
        rig["root"],
        rig["example"],
        episode=rig["spec"](),
        capture=tl.options.CaptureOptions(log_injections=True),
    )
    classification, _ = _expect("log_injections")
    assert classification == ORTHOGONAL
    assert trace.outcome.status.name == "COMPLETE"
    assert trace.annotations["episode"] is not None


def test_episode_kwarg_on_record_is_a_ledgered_untyped_refusal(
    episode_rig: dict[str, Any],
) -> None:
    """tl.record has no episode arm; the bare TypeError is a ledgered gap
    (DIGEST-AUDIT: record refuses trace-only kwargs untyped, B6 D20 class)."""

    import torchlens as tl

    rig = episode_rig
    _assert_refuses(
        "record_entry",
        lambda: tl.record(rig["root"], rig["example"], save=tl.func("tanh"), episode=rig["spec"]()),
    )


def test_episode_module_root_requires_stepped_module(episode_rig: dict[str, Any]) -> None:
    """EpisodeSpec(stepped_module=None) on a module root refuses typed (F41)."""

    import torchlens as tl

    rig = episode_rig
    _assert_refuses(
        "module_root_without_stepped_module",
        lambda: tl.trace(
            rig["root"],
            rig["example"],
            episode=tl.options.EpisodeSpec(n_steps=3, step_output_kind="digest"),
        ),
    )


def test_episode_forced_tokens_on_float_root_refuses_declaration(
    episode_rig: dict[str, Any],
) -> None:
    """Teacher forcing declares tokens; a float root fails the declaration."""

    import torchlens as tl

    rig = episode_rig
    _assert_refuses(
        "forced_tokens_on_float_root",
        lambda: tl.trace(
            rig["root"],
            rig["example"],
            episode=rig["spec"](step_output_kind="tokens", forced_tokens=(1, 2, 3)),
        ),
    )


def test_bound_method_episode_defaults_stepped_module_to_owner() -> None:
    """F41 x episode: on a bound-method root the owner IS the default step."""

    import torchlens as tl

    torch.manual_seed(0)
    generator = _SelfSteppingGenerator().eval()
    example = torch.randn(1, 4)
    trace = tl.trace(
        generator.generate,
        example,
        episode=tl.options.EpisodeSpec(n_steps=3, step_output_kind="digest"),
    )
    classification, _ = _expect("bound_method_default_owner")
    assert classification == COUPLED
    assert trace.outcome.status.name == "COMPLETE"
    assert str(trace.root_entry_point).startswith("bound_method:")
    rows = trace.annotations["episode"]["rows"]
    assert len(rows) == 3 and all(row["status"] == "complete" for row in rows), (
        "default-owner stepping did not attribute the three generate steps"
    )


def test_episode_declared_steps_that_never_run_quarantine_typed() -> None:
    """Declared n_steps with zero stepped calls is a typed incoherence, never
    a silently-empty ledger (measured live: the owner default on a root whose
    generate steps a SUBMODULE runs the owner zero times)."""

    import torchlens as tl

    class _SubSteppingGenerator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = _StepModule()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.step(x)

        def generate(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = self.step(x)
            return x

    torch.manual_seed(0)
    generator = _SubSteppingGenerator().eval()
    _assert_refuses(
        "declared_steps_never_run",
        lambda: tl.trace(
            generator.generate,
            torch.randn(1, 4),
            episode=tl.options.EpisodeSpec(n_steps=3, step_output_kind="digest"),
        ),
    )


def test_episode_option_table_is_closed() -> None:
    """Unlisted cells refuse: every table row has a driver and vice versa."""

    import inspect
    import sys

    module = sys.modules[__name__]
    source = inspect.getsource(module)
    for cell_id, classification, code in EPISODE_OPTION_TABLE:
        assert classification in {ORTHOGONAL, COUPLED, TYPED_REFUSED, UNTYPED_REFUSED}
        if classification == TYPED_REFUSED:
            assert code, f"{cell_id}: TYPED-REFUSED without a pinned code"
        # Every cell id must be exercised by a driver in this module.
        occurrences = source.count(f'"{cell_id}"')
        assert occurrences >= 2, (
            f"table cell {cell_id!r} has no driver in this module -- a listed"
            " combination without an executed cell is an unwitnessed claim"
        )
    cell_ids = [row[0] for row in EPISODE_OPTION_TABLE]
    assert len(cell_ids) == len(set(cell_ids)), "duplicate cell ids in the table"
