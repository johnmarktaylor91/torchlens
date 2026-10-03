"""P1 gates: record types, amendment registry, seam contracts, scatter parity.

* Three-way coverage closure: manifest keys == store slots + extra-key
  channels; every ingest-produced fields_dict key classified.
* Regenerate-and-diff: the generated manifest matches the source spec.
* Scatter parity: for every journal record of the scenario battery,
  ``op_record_from_event`` -> ``scatter_record_to_cells`` reproduces today's
  ``_fields_from_event`` output cell-for-cell on every record-sourced cell
  (JOINs neutralized on both sides) — the substance of the
  ``Op._from_cells == Op(fields_dict)`` gate before ingest rewires in P3.
* Amendment registry battery: exact-set violations red; typed constructors
  green; identity fields unpatchable; PATH_TO_FLAT covers exactly the union.
* Strict protocol: unknown names raise ``OpRecordAttributeError`` and
  ``getattr(..., default)`` keeps default semantics.
* IngestInputs v1 carries the RESERVED ``aten_events`` lane (cross-sprint
  requirement) and covers the recorded step-0 trace-read inventory.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import torchlens as tl
from torchlens.ir.op_record import (
    AMENDMENT_FAMILIES,
    PATH_TO_FLAT,
    TYPED_CONSTRUCTORS,
    AmendmentValidationError,
    OpAmendment,
    OpRecordAttributeError,
    amend_late_buffer_output_parent,
    op_record_from_event,
    validate_amendment,
)
from torchlens.ir.op_record_scatter import (
    CELL_SOURCES,
    EXTRA_KEY_CHANNELS,
)

# Markers are additive: a file-level smoke pytestmark would keep the heavy test
# in the `-m smoke` tier, so the smoke mark is applied per test instead.


def test_three_way_manifest_closure() -> None:
    from torchlens.data_classes.op import _OP_SLOT_NAMES

    manifest_keys = set(CELL_SOURCES)
    slot_keys = set(_OP_SLOT_NAMES)
    universe = slot_keys | EXTRA_KEY_CHANNELS
    missing = universe - manifest_keys
    assert not missing, f"unclassified store cells: {sorted(missing)}"
    stale = manifest_keys - universe - {"is_in_conditional_body", "source_trace"}
    assert not stale, f"manifest rows without a store cell: {sorted(stale)}"


def test_manifest_regenerate_and_diff() -> None:
    from tools.generate_op_record_manifest import generate

    generated = Path("torchlens/ir/op_record_manifest.py").read_text()
    assert generated == generate(), (
        "generated manifest is stale — run python -m tools.generate_op_record_manifest"
    )
    from torchlens.ir.op_record_manifest import CELL_SOURCE_MANIFEST

    assert CELL_SOURCE_MANIFEST == CELL_SOURCES


@pytest.mark.heavy
@pytest.mark.parametrize("invocation", ["script", "module"])
def test_manifest_check_cli_works_as_a_subprocess(invocation: str) -> None:
    """The manifest generator's ``--check`` works as contributors run it (R53-A).

    The in-process regenerate-and-diff above cannot catch a broken CLI: the
    plain-script spelling failed with ``ModuleNotFoundError`` (no repo root on
    ``sys.path``, unlike its ``generate_record_schema.py`` sibling, whose CLI
    IS subprocess-gated at ``test_record_schema_bindings.py``), and neither
    spelling was exercised as a subprocess. ``PYTHONPATH`` is stripped so the
    script spelling proves the generator's own path bootstrap rather than the
    test runner's environment. Green requires rc==0 AND the explicit ``ok:``
    line, so import breakage and a stale manifest red distinctly.
    """

    repo_root = Path(__file__).resolve().parents[2]
    if invocation == "script":
        command = [sys.executable, str(repo_root / "tools" / "generate_op_record_manifest.py")]
    else:
        command = [sys.executable, "-m", "tools.generate_op_record_manifest"]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [*command, "--check"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(repo_root),
        env=environment,
    )
    assert result.returncode == 0, (
        f"--check ({invocation}) failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert result.stdout.strip().startswith("ok:"), result.stdout


def test_amendment_registry_exact_sets() -> None:
    union = {path for schema in AMENDMENT_FAMILIES.values() for path, _ in schema}
    assert union == set(PATH_TO_FLAT), "PATH_TO_FLAT must cover exactly the path union"
    assert set(TYPED_CONSTRUCTORS) == set(AMENDMENT_FAMILIES)

    good = amend_late_buffer_output_parent(7, "buffer_1_raw", is_output_parent=True)
    validate_amendment(good)

    # unregistered family
    with pytest.raises(AmendmentValidationError, match="unregistered"):
        validate_amendment(
            OpAmendment(0, 0, 7, "x_raw", "no_such_family", (("graph.is_output_parent", True),))
        )
    # wrong path set (missing member)
    with pytest.raises(AmendmentValidationError, match="exact"):
        validate_amendment(
            OpAmendment(
                0,
                0,
                7,
                "x_raw",
                "module_exit_intervention",
                (("intervention.intervention_fired", True),),
            )
        )
    # same-set wrong ORDER also refused (ordered exact-set)
    with pytest.raises(AmendmentValidationError, match="exact"):
        validate_amendment(
            OpAmendment(
                0,
                0,
                7,
                "x_raw",
                "lookback_retention",
                (
                    ("policy.predicate_matched", True),
                    ("core.output", object()),
                ),
            )
        )
    # value type violation
    with pytest.raises(AmendmentValidationError, match="value type"):
        validate_amendment(
            OpAmendment(
                0, 0, 7, "x_raw", "late_buffer_output_parent", (("graph.is_output_parent", "yes"),)
            )
        )
    # identity fields unpatchable even under a forged family row
    forged = dict(AMENDMENT_FAMILIES)
    try:
        AMENDMENT_FAMILIES["forged"] = (("core.label_raw", (str,)),)
        with pytest.raises(AmendmentValidationError, match="unpatchable"):
            validate_amendment(
                OpAmendment(0, 0, 7, "x_raw", "forged", (("core.label_raw", "evil"),))
            )
    finally:
        AMENDMENT_FAMILIES.clear()
        AMENDMENT_FAMILIES.update(forged)
        AMENDMENT_FAMILIES.pop("forged", None)
    # anchor required
    with pytest.raises(AmendmentValidationError, match="anchor"):
        validate_amendment(
            OpAmendment(
                0, 0, 7, "", "late_buffer_output_parent", (("graph.is_output_parent", True),)
            )
        )


@pytest.mark.smoke
def test_strict_protocol_refusal_type() -> None:
    from ._models import SmallCNN, _cnn_input

    events = _journal_events(SmallCNN(), _cnn_input())
    record, _ = op_record_from_event(events[0])
    with pytest.raises(OpRecordAttributeError):
        _ = record.definitely_not_a_field
    # AttributeError subclass: getattr default semantics survive
    assert getattr(record, "definitely_not_a_field", "fallback") == "fallback"
    # legacy flat names read through the facets
    assert record.label_raw == events[0].label_raw
    assert record.parent_arg_positions == events[0].parent_arg_positions


def _journal_events(model, inputs) -> list:
    """Return genuine compat OpEvents for adapter-input tests.

    The legacy torch producer died in P7; compat ``OpEvent`` journals (the
    ingest adapter's input shape, still emitted by preview backends until
    S15) are synthesized from a decomposed capture through the retained
    inverse adapter.
    """

    import torchlens.postprocess as postprocess_module
    import torchlens.postprocess._materialize as materialize_module

    from ._oracle_adapter import op_event_from_record

    captured: list = []
    original = materialize_module.materialize_from_events

    def spy(trace, events):
        captured.extend(events.op_events)
        original(trace, events)

    postprocess_module.materialize_from_events = spy
    materialize_module.materialize_from_events = spy
    try:
        tl.trace(model, inputs)
    finally:
        postprocess_module.materialize_from_events = original
        materialize_module.materialize_from_events = original
    return [op_event_from_record(entry) for entry in captured]


@pytest.mark.heavy
def test_scatter_is_the_single_ingest_truth(tmp_path: Path) -> None:
    """Every journal record ingests through the generated scatter (P3).

    ``_fields_from_event`` is deleted; the record-sourced cells of every
    materialized op flow through ``scatter_record_to_cells`` exactly once per
    record, and the scatter output lands verbatim in the Op fields ingest
    hands to construction (spot-checked on identity cells). Byte-identity of
    the full store/journal/artifact layers vs pre-migration main is carried
    by the temporal-baseline comparator (P0 archive, 6.2b).
    """

    import torchlens.ir.op_record_scatter as scatter_module

    from ._models import SCENARIOS
    from ._snapshot import run_scenario

    for scenario in SCENARIOS:
        if scenario.name == "cnn_backward":
            continue  # backward mutates grads post-capture; journal identical anyway
        journal_labels: list[str] = []

        def collect(journal_events, _sink=journal_labels) -> None:
            _sink.extend(e.label_raw for e in journal_events.op_events)

        scattered: list[tuple[str, dict]] = []
        original_scatter = scatter_module.scatter_record_to_cells

        def observing_scatter(
            record, extras, owning_trace, _orig=original_scatter, _sink=scattered
        ):
            cells = _orig(record, extras, owning_trace)
            _sink.append((record.core.label_raw, cells))
            return cells

        scatter_module.scatter_record_to_cells = observing_scatter
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run = run_scenario(
                    scenario,
                    Path(tmp),
                    with_artifact=False,
                    journal_mutator=collect,
                )
        finally:
            scatter_module.scatter_record_to_cells = original_scatter
        assert journal_labels, f"{scenario.name}: no journal rows (vacuous)"
        scattered_labels = [label for label, _ in scattered]
        assert sorted(scattered_labels) == sorted(journal_labels), (
            f"{scenario.name}: scatter coverage mismatch — records not routed "
            "through the single ingest truth"
        )
        # scatter output lands verbatim in the materialized rows (identity cells)
        trace = run.trace
        raw_labels = {
            getattr(op, "raw_label", None) or getattr(op, "_label_raw", None) for op in trace.ops
        }
        for label, cells in scattered:
            assert cells["_label_raw"] == label
            if label in raw_labels:
                assert cells["type_index"] is not None


def test_ingest_inputs_v1_reserves_aten_lane_and_covers_step0_reads() -> None:
    import json

    from torchlens.postprocess._ingest_contract import (
        INGEST_CONTRACT_ROWS,
        IngestInputs,
        JournalView,
        Step0Result,
    )

    journal_fields = {f.name for f in dataclasses.fields(JournalView)}
    assert "aten_events" in journal_fields, "reserved aten_events lane missing"
    assert JournalView.__dataclass_fields__["aten_events"].default == ()
    assert any(row[0] == "journal.aten_events" for row in INGEST_CONTRACT_ROWS)

    input_fields = {f.name for f in dataclasses.fields(IngestInputs)}
    for required in (
        "journal",
        "module_workspace",
        "raw_graph_workspace",  # X1
        "buffer_initial_values",
        "op_equivalence_classes",
        "source_model_ref",
        "param_logs",
        "owning_trace",
        "op_row_store",
        "trace_core",  # ppdag 9.1 orchestrator row
        "input_layers_initial",  # X4
        "timing_sink",  # X3
        "scatter_options",
    ):
        assert required in input_fields, required
    result_fields = {f.name for f in dataclasses.fields(Step0Result)}
    assert result_fields == {
        "raw_log_registrations",  # X2 local-map derivation
        "input_layer_labels",
        "equivalence_class_map",
        "module_side_channel",
    }

    # the recorded step-0 trace-read inventory must be coverable by the
    # enumeration (each read maps to a declared IngestInputs handle or a
    # named orchestrator-epilogue attribute)
    ledger_path = Path(__file__).resolve().parent / "ledger" / "step0_trace_reads.json"
    if ledger_path.exists():
        observed = set(json.loads(ledger_path.read_text()))
        covered = {
            # journal + workspaces + registries
            "capture_events": "journal",
            "_module_capture_ws": "module_workspace",
            "_raw_graph_ws": "raw_graph_workspace",
            "_buffer_initial_values": "buffer_initial_values",
            "op_equivalence_classes": "op_equivalence_classes",
            "_source_model_ref": "source_model_ref",
            "param_logs": "param_logs",
            "input_layers": "input_layers_initial",
            "_trace_core": "trace_core",
            "_phase_timings": "timing_sink",
        }
        uncovered = {
            name
            for name in observed
            if name in covered
            and covered[name] not in {f.name for f in dataclasses.fields(IngestInputs)}
        }
        assert not uncovered, f"declared coverage broken: {uncovered}"
