"""Frozen step-0 ingest seam contracts: ``IngestInputs`` v1 / ``Step0Result`` v1.

The producer-unification / postprocess-DAG joint seam (producer DoR v4
section 5.2; ppdag v3 section 9). The orchestrator (DAG lane) constructs
``IngestInputs``, invokes ``ingest_op_records(inputs, manifest)``, and applies
the returned ``Step0Result`` payloads; ingest itself reads NOTHING off the
trace outside this bundle (contract clause I7, enforced by the P0/P1
instrumentation: the recorded step-0 trace-read set must be covered by this
enumeration before anything freezes).

Version 1 is FROZEN jointly with ``CellSourceManifest`` v1 (see
``torchlens/ir/op_record_manifest.py``); the DAG contract shape lands first
per ppdag v3 section 9.3. Field additions before the joint freeze are the
sanctioned evolution path; post-freeze changes are versioned (v2+), never
silent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

INGEST_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class JournalView:
    """The enumerated read-only journal lanes ingest may consume.

    One field per lane; the ``aten_events`` lane is RESERVED (named-but-empty)
    for the aten/kernel follow-up sprint — naming it before the v1 freeze
    costs one row now and saves a schema break later (cross-sprint
    requirement, 2026-08-12).
    """

    op_events: tuple[Any, ...]  # the FOLDED reducer view (amended facts)
    op_amendments: tuple[Any, ...]  # provenance lane; already folded into op_events
    module_prep_events: tuple[Any, ...]
    module_enter_events: tuple[Any, ...]
    module_exit_events: tuple[Any, ...]
    pre_hook_events: tuple[Any, ...]  # module provenance (review round-3 omission)
    buffer_write_events: tuple[Any, ...]
    output_version_events: tuple[Any, ...]
    grad_fn_handles_by_label_raw: Mapping[str, Any]
    # RESERVED: immutable AtenCallEvent records (future aten sprint). Always
    # empty in v1; consumers must tolerate the lane existing and being empty.
    aten_events: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestInputs:
    """Everything step-0 ingest is allowed to read (contract I7).

    Read-only inputs enumerated from the instrumented inventory (P0 ledger
    ``step0_trace_reads.json``) plus the ppdag cross-design findings X1-X5:

    * ``journal`` — the lane view above (X5: ``pre_hook_events`` included).
    * ``module_workspace`` — ``ModuleCaptureWorkspace`` handle; module
      side-channel rebuild target AND ``module_forward_args`` read source.
    * ``raw_graph_workspace`` — X1: declared READ handle
      (``input_tensor_addresses`` consumed by input-role assignment), not
      merely a mutation-payload target.
    * ``buffer_initial_values`` — declared buffer state universe.
    * ``op_equivalence_classes`` — the shared-set identity source (M7 keys on
      object identity, so the SAME mapping object must flow through).
    * ``source_model_ref`` — weakref to the source model.
    * ``param_logs`` — the params registry.
    * ``owning_trace`` — the ONE trace-identity reference (source_trace cell
      default join target), declared explicitly.
    * ``op_row_store`` / ``trace_core`` — created by the ORCHESTRATOR (ppdag
      v3 section 9.1 ownership row); ingest receives them ready-made.
    * ``input_layers_initial`` — X4: the dedup baseline; the Step0Result
      payload spec is "append with dedup against this declared initial state".
    * ``timing_sink`` — X3: the declared timing disposition; the orchestrator
      passes a trace-backed sink so ingest never touches
      ``trace._phase_timings`` directly.
    * ``scatter_options`` — mechanically inventoried trace-level options the
      converters read (closed by the same instrumentation gate).
    """

    journal: JournalView
    module_workspace: Any
    raw_graph_workspace: Any
    buffer_initial_values: Mapping[str, Any]
    op_equivalence_classes: Mapping[str, set]
    source_model_ref: Any
    param_logs: Any
    owning_trace: Any
    op_row_store: Any
    trace_core: Any
    input_layers_initial: tuple[str, ...]
    timing_sink: Callable[[str, float], None]
    scatter_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Step0Result:
    """Mutations restaged as payloads the ORCHESTRATOR applies.

    * ``raw_log_registrations`` — (label_raw, op_log) rows for the
      ``RawGraphWorkspace`` raw-layer dict/list (X2: ingest builds and
      consumes its own LOCAL registration map; this payload is derived from
      it, so no in-out workspace is needed).
    * ``input_layer_labels`` — append-with-dedup against
      ``IngestInputs.input_layers_initial`` (X4).
    * ``equivalence_class_map`` — the populated equivalence map (same shared
      set objects as ``op_equivalence_classes``).
    * ``module_side_channel`` — the module side-channel rebuild output the
      orchestrator applies to the module workspace.

    Named fallback (DoR 5.2): anything that proves too entangled at P3 moves
    to a DECLARED in-out parameter on ``IngestInputs`` with a documented
    mutation set — never silent trace access.
    """

    raw_log_registrations: tuple[tuple[str, Any], ...]
    input_layer_labels: tuple[str, ...]
    equivalence_class_map: Mapping[str, set]
    module_side_channel: Any


# Contract rows: one line per lane/handle, mirrored into the joint seam note.
INGEST_CONTRACT_ROWS: tuple[tuple[str, str], ...] = (
    ("journal.op_events", "read-only op lane (reducer view once P4 lands)"),
    ("journal.op_amendments", "read-only amendment provenance lane (pre-folded)"),
    ("journal.module_prep_events", "read-only"),
    ("journal.module_enter_events", "read-only"),
    ("journal.module_exit_events", "read-only"),
    ("journal.pre_hook_events", "read-only module provenance"),
    ("journal.buffer_write_events", "read-only"),
    ("journal.output_version_events", "read-only"),
    ("journal.grad_fn_handles_by_label_raw", "read-only handle side index"),
    ("journal.aten_events", "RESERVED empty lane for the aten follow-up sprint"),
    ("module_workspace", "read handle + side-channel rebuild TARGET (via Step0Result)"),
    ("raw_graph_workspace", "read handle (input_tensor_addresses); X1"),
    ("buffer_initial_values", "read-only"),
    ("op_equivalence_classes", "read-only identity-bearing mapping"),
    ("source_model_ref", "read-only weakref"),
    ("param_logs", "read-only registry"),
    ("owning_trace", "identity join target ONLY (source_trace default)"),
    ("op_row_store", "destination store handle (orchestrator-created)"),
    ("trace_core", "orchestrator-created core handle (ppdag 9.1)"),
    ("input_layers_initial", "read-only dedup baseline; X4"),
    ("timing_sink", "declared timing disposition; X3"),
    ("scatter_options", "mechanically inventoried option reads"),
)
