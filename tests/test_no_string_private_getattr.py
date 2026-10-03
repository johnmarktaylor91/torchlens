"""b5 R45-2 / SF-41: string private-attribute reach-ins are ledgered, not free.

``getattr(trace, "_raw_to_final_op_labels", {})`` is a private reach-in that
the linter cannot see: SLF001 flags ``trace._raw_to_final_op_labels`` but NOT
its string spelling, so the measured 1771-1777 SLF001 hits UNDERCOUNT the real
encapsulation surface by this whole class. Worse, the three-argument form is
fail-OPEN by construction -- a renamed or dropped field silently becomes the
default instead of an error, which is exactly how the merged presenter came to
resolve boundary ops against a possibly-empty mapping (fixed in the same
change as this gate).

This module is two things:

1. an AST ratchet over ``getattr``/``setattr``/``hasattr``/``delattr`` calls
   whose attribute name is a private string literal and whose base is a
   TRACE-shaped identifier -- reach-ins into TorchLens's own central object.
   The per-package ledger is exact-equality in BOTH directions: adding one
   fails, and fixing one requires lowering the row (shrink-only). A package
   with NO row must stay at zero, which is how ``merged/`` (fail-closed by
   ethos) and the clean packages are held.
2. the regression pins for the fixed presenter seam.

Whole-package context at seeding time (2026-08-14, b5 Lane B5-GOV): 678
private string reach-ins on non-``self`` bases overall, of which 366 are on
trace-shaped bases (ledgered below). The remainder are reach-ins into torch
internals, records, workspaces, and modules; the torch-private subset is
separately gated by ``tests/test_private_probe_gate.py``.
"""

from __future__ import annotations

import ast
import collections
from functools import lru_cache
from pathlib import Path

import pytest
from _source_corpus import package_ast, package_files

from torchlens.merged._errors import MergeInputError
from torchlens.merged._presenter import MergedTrace, _rank_raw_to_final_op_labels

# Module-wide smoke dropped (r3settle2 budget lint): the repo-wide reach-in
# count scan below measures over the 5s smoke partition; per-test marks.

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "torchlens"

#: Builtins that take an attribute NAME as a string, bypassing SLF001.
_REACHIN_BUILTINS = frozenset({"getattr", "setattr", "hasattr", "delattr"})

#: Identifiers that denote a ``Trace`` in this codebase. Deliberately a closed
#: vocabulary: `trace`/`log`/`ml` are the three historical spellings, the rest
#: are the qualified locals the capture and refresh paths use. A new spelling
#: for the same object should be added here, not used to dodge the gate.
_TRACE_IDENTIFIERS = frozenset(
    {
        "trace",
        "new_trace",
        "log",
        "model_log",
        "ml",
        "target_trace",
        "source_trace",
        "refreshed",
    }
)

#: Packages whose reach-in count MUST stay zero. ``merged/`` is fail-closed by
#: contract (docs/reference/merged_trace_contract.md): a rank core that cannot
#: answer refuses typed, so no silent-default read belongs there.
#: ``distributed/`` is the positive control the b5 hunt found clean.
_FAIL_CLOSED_PACKAGES = frozenset({"merged", "distributed"})

#: Per-package count of private string reach-ins on trace-shaped bases, seeded
#: from the tree at b5 fixplan time. EXACT in both directions; a package
#: without a row must stay at zero. To discharge rows, replace the reach-in
#: with a declared seam (see ``merged/_presenter.py``'s
#: ``_rank_raw_to_final_op_labels``: direct private read, absence typed) or
#: with a public accessor, then lower the count here.
_TRACE_REACHIN_LEDGER: dict[str, int] = {
    # 14 -> 15 (2026-08-14 fix-wave reconcile, f2bc65a6 fix/interv F2): the
    # live-refresh run report reads the optional `_runnable` seam in
    # _runnable_transaction.py; absent on non-runnable traces, so the None
    # default is the correct "no runnable seam" reading.
    # 15 -> 16 (2026-08-15 fw3settle reconcile, fe2444e3 fix/runnable-r4): the
    # fast-provider host-RNG declaration reads the optional `_runnable` seam in
    # _fast_run.py, the same f2bc65a6 idiom already ledgered here and at _io.
    # 16 -> 19 (2026-08-17 privprobes reconcile): kernel_telemetry.py
    # (a56cef72, L3 telemetry lane) reads the optional DROP-gated
    # `_primitive_op_profile` store (x2; absent on non-aten / loaded traces,
    # so the None default is the correct "no primitive layer" reading -- the
    # idiom already ledgered at postprocess/validation for that exact field),
    # and _runnable_transaction.py (79607e17, declared-state snapshot-restore)
    # latches state_compromised onto the optional `_runnable` seam after a
    # failed live restore (absent = the trace never ran, nothing to latch).
    "<root>": 19,
    # 24 -> 25 (2026-08-15 r3settle reconcile): bundle.py reads the optional
    # `_runnable` seam (absent on non-runnable traces; None default correct),
    # the same f2bc65a6 idiom already ledgered at <root>.
    # 25 -> 26 (2026-08-16 fw7settle reconcile, fcf0e97c buffer-value channel
    # gate): accessor_rebuild.py/bundle.py read the optional
    # `_buffer_initial_values` channel; absent unless the capture recorded
    # pre-forward buffer values, so the None default is the correct "no
    # channel" reading (net +1 against a discharged older site).
    "_io": 26,
    "autoroute": 2,
    "backends/jax": 7,
    "backends/mlx": 21,
    "backends/paddle": 20,
    "backends/tf": 9,
    "backends/tinygrad": 5,
    # 111 -> 112 (2026-08-14 fixwave-2 reconcile): heavy site churn from the
    # ops.py/completeness_witness.py file splits nets +1, dominated by the
    # intended safetynet stage-2 rescue path (rescue.py) and backward-projection
    # additions against removed `_layer_counter` reads.
    # 112 -> 114 (2026-08-14 fix-wave reconcile, 6d5fdf64 fix/bwgrad): the
    # op-gradient budget charge reads the optional session-time
    # `_save_budget_accountant` in tensor_tracking.py (x2), the same idiom as
    # the pre-existing _ops_retention.py reads of the same field.
    # 114 -> 117 (2026-08-15 r3settle reconcile, b9937876 fix/backward): the
    # per-call save_grads policy reads the optional session-time
    # `_active_save_grads_policy` (tensor_tracking.py hasattr+getattr,
    # backward.py getattr); absent outside a managed backward window, so the
    # attribute-fallback default is the correct reading.
    # 117 -> 118 (2026-08-16 fw6settle reconcile, aa4e874a fix/bwgrad): the
    # output-parent promotion charge reads the optional session-time
    # `_save_budget_accountant` in backend.py, the same idiom as the
    # pre-existing _ops_retention.py/tensor_tracking.py reads of that field.
    # 118 -> 120 (2026-08-16 fw7settle reconcile, 76fe9c94 r8 R16+R34
    # accountant settle): two more `_save_budget_accountant` reads in
    # _ops_retention.py, the same optional session-time idiom already
    # ledgered for that exact field.
    # 120 -> 124 (2026-08-16 l2/l3 merge-gate reconcile, 5f0a4f8d aten wave 0):
    # _aten_capture.py reads the optional session-time `_module_capture_ws`
    # workspace (x2) and `_capture_events` stream (x2); both are absent
    # outside a live capture window, so the None default is the correct
    # "no active capture" reading (aten recording then no-ops).
    # 124 -> 123 (2026-08-17 privprobes reconcile, d4a2b8a6): the standalone
    # `_warned_implicit_backward_pass` read in tensor_tracking.py was
    # discharged by the one-shot-warning consolidation into `_warned_once`.
    # 123 -> 116 (fast2 ci-fix reconcile): nine sites from the F01
    # injection/offload-shim/FLIP-2 lanes converted to direct private reads
    # or a shared `peek_*` seam (buffer_writes.py's
    # `peek_buffer_write_tracker`, now reused by six call sites that were
    # each a separate string reach-in before this reconcile, net -7 there
    # alone), discharging all nine new sites plus six pre-existing ones.
    "backends/torch": 116,
    "bridge": 1,
    "bundle": 1,
    # 20 -> 22 (2026-08-16 l2 episode, f2228f46 S7 suite): the episode
    # partial-failure disclosure in _episode_ledger.py best-effort reads the
    # optional `_capture_events` stream and `_raw_graph_ws` workspace off a
    # FAILED partial trace; either may legitimately be absent at the point of
    # failure, so the None default is the correct "nothing to disclose"
    # reading (the row degrades to interrupted/absent, never guesses).
    # 22 -> 23 (2026-08-17 privprobes reconcile, a5eced2d D18 buffer-sink
    # projector): projectors.py reads `module_training_modes` through the
    # optional `_runnable` seam (f2bc65a6 idiom); absent on never-run traces,
    # so the empty-modes default is the correct "no recorded modes" reading.
    # 23 -> 21 (fast2 ci-fix reconcile): the episode-failed/nonfinite-prefix/
    # peak-memory sites converted to direct private reads guarded by
    # try/except AttributeError, net -2 after the new episode/peak-memory
    # additions.
    "capture": 21,
    # 25 -> 27 (2026-08-14 fix-wave reconcile, 7f90a885 fix/walkers): the
    # linear ordinal_index cache keys its per-trace memo on the session-time
    # `_backward_projection_revision` counter in grad_fn_call.py (x2); absent
    # on loaded traces, so the None default is the correct reading.
    # 27 -> 29 (2026-08-15 r3settle reconcile): op.py's out-dedup path reads
    # the optional session-time `_out_dedup_mode` / `_out_identity_cache`
    # knobs; absent on default captures, so the identity/None defaults are the
    # correct "dedup off" reading.
    # 29 -> 30 (2026-08-16 fw6settle reconcile, aa4e874a fix/bwgrad): the
    # reused charged-grad-payload path in op.py adds a third read of the
    # optional session-time `_defer_streaming_bundle_finalization` flag, the
    # same idiom as the two reads already ledgered for that file.
    # 30 -> 31 (2026-08-16 fw7settle reconcile, 76fe9c94 r8 R16+R34
    # accountant settle): op.py's release path reads the optional
    # session-time `_save_budget_accountant`, the idiom already ledgered at
    # backends/torch for that exact field.
    # 31 -> 32 (2026-08-17 privprobes reconcile, 79607e17 declared-state
    # snapshot-restore): _trace_validation.py reads the state-compromised
    # latch through the optional `_runnable` seam; absent means no prior
    # run() ever latched, so the None default correctly passes the gate.
    # 32 -> 25 (fast2 ci-fix reconcile): _op_dedup.py's seven dedup-cache
    # reach-ins (getattr + setattr) and _trace_validation.py's new
    # `_loaded_from_bundle` read converted to direct private reads guarded
    # by try/except AttributeError.
    "data_classes": 25,
    "experimental": 1,
    "fastlog": 2,
    # 43 -> 44 (2026-08-17 privprobes reconcile, d4a2b8a6 one-shot-warning
    # consolidation): the replay.py refresh scrub discards one key from the
    # consolidated `_warned_once` set; absent on traces that never warned, so
    # the empty-set default is the correct "nothing to discard" reading.
    # 44 -> 41 (fast2 ci-fix reconcile): the F01 injection-state reads across
    # injection.py/runtime.py/_module_boundary.py/_ops_interventions.py
    # (backends/torch) converted to the shared `peek_injection_state` seam
    # or a direct read guarded by try/except AttributeError.
    "intervention": 41,
    "ir": 2,
    # 5 -> 8 (2026-08-16 l2/l3 merge-gate reconcile, 5f0a4f8d aten wave 0):
    # _primitive_profile.py reads the optional DROP-gated
    # `_primitive_op_profile` store (x2) and `_capture_events` stream; absent
    # on non-aten / loaded traces, so the None default is the correct
    # "no primitive layer" reading.
    "postprocess": 8,
    "report": 1,
    # 7 -> 6 (2026-08-14 fixwave-2 reconcile): one reach-in discharged upstream.
    # 6 -> 0 (fast2 ci-fix reconcile): every repgeom reach-in (the
    # `_annotation_blobs`/`_validate_annotation_tensor`/
    # `_mark_annotations_mutated` trio across _annotation_gate.py,
    # _node_visuals.py, and _trace_views.py) converted to direct private
    # reads guarded by try/except AttributeError; no row left to hold --
    # same zero-row convention as a package that never had a reach-in.
    # 0 -> 2 (2026-08-14 fixwave-2 reconcile, R40 viz hardening): the shared
    # `_visualizer_dir` scratch-dir helpers moved into utils/display.py; the
    # attribute is genuinely optional session state (absent until the first
    # render), so the None default is the correct "no scratch dir yet" reading.
    "utils": 2,
    # 23 -> 24 (2026-08-15 r3settle reconcile): _invariants_payloads.py reads
    # the optional `_trace_core` op-store seam, absent on loaded/preview
    # traces, so the None default is the correct "no sealed core" reading.
    # 24 -> 25 (2026-08-16 fw7settle reconcile, 5ef9cf21 loaded-artifact
    # false-fire fix): _invariants_payloads.py reads the optional
    # `_loaded_from_bundle` marker to scope op_log_fields to live captures;
    # absent on live traces, so the False default is the correct reading.
    # 25 -> 30 (2026-08-16 l2/l3 merge-gate reconcile, 5f0a4f8d aten wave 0):
    # _invariants_primitive_ops.py reads the optional DROP-gated
    # `_primitive_op_profile` store (x3, absent = no primitive layer to
    # check, and the non-torch invariant asserts it IS absent) and the
    # `_tracing_finished` marker (x2) that scopes the unresolved-ownership
    # check to completed traces; both defaults are the correct readings.
    # 30 -> 29 (2026-08-17 privprobes reconcile): the l2/l3 row above landed
    # with ONE `_tracing_finished` read in _invariants_primitive_ops.py, not
    # two -- correcting the overcount (profile x3 + tracing_finished x1).
    "validation": 29,
    # 20 -> 23 (2026-08-14 fixwave-2 reconcile): intended R19/R40 rendering
    # additions (node-overlay names/scores, source-code blob, `_visualizer_dir`
    # consolidation into _render_dot.py) against removed `_raw_layer_dict` /
    # _render_nodes.py sites.
    # 23 -> 22 (2026-08-14 fix-wave reconcile, fix/vizr2 2d1371ac): one
    # reach-in discharged with the imagepath-out-of-saved-DOT rework.
    "visualization": 22,
    # 2 -> 1 (2026-08-14 fixwave-2 reconcile): one reach-in discharged upstream.
    "viz": 1,
}


def _is_private_name(name: str) -> bool:
    """Whether a literal attribute name is single-underscore private."""

    return name.startswith("_") and not name.startswith("__")


def _package_of(path: Path) -> str:
    """Ledger key for one module: its top-level package (backends split one deeper)."""

    parts = path.relative_to(_PACKAGE_ROOT).parts
    if len(parts) == 1:
        return "<root>"
    if parts[0] == "backends" and len(parts) > 2:
        return f"backends/{parts[1]}"
    return parts[0]


def _trace_reachins(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Collect ``(lineno, builtin, attr)`` string reach-ins on trace-shaped bases."""

    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _REACHIN_BUILTINS
            and len(node.args) >= 2
        ):
            continue
        name_arg = node.args[1]
        if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
            continue
        if not _is_private_name(name_arg.value):
            continue
        base = ast.unparse(node.args[0])
        if base in {"self", "cls"}:
            continue
        # `a.b.trace`, `state.trace`, `handles[r].trace` all denote a Trace.
        tail = base.split(".")[-1].split("[")[0]
        if tail in _TRACE_IDENTIFIERS:
            found.append((node.lineno, node.func.id, name_arg.value))
    return found


@lru_cache(maxsize=1)
def _scan_package() -> dict[str, list[str]]:
    """Package -> sorted ``file:line builtin attr`` reach-in sites."""

    found: dict[str, list[str]] = collections.defaultdict(list)
    for path in package_files():
        rel = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        tree = package_ast(path)
        for lineno, builtin, attr in _trace_reachins(tree):
            found[_package_of(path)].append(f"{rel}:{lineno} {builtin}(..., {attr!r})")
    return {package: sorted(sites) for package, sites in found.items()}


@pytest.mark.heavy
def test_trace_reachin_counts_match_the_ledger_exactly() -> None:
    """Per-package trace reach-in counts equal the seeded ledger, both ways."""

    found = _scan_package()
    counts = {package: len(sites) for package, sites in found.items()}
    grew = {
        package: (_TRACE_REACHIN_LEDGER.get(package, 0), count)
        for package, count in counts.items()
        if count > _TRACE_REACHIN_LEDGER.get(package, 0)
    }
    assert not grew, (
        "New private string reach-in(s) into a Trace (package: ledgered -> found): "
        f"{grew}. `getattr(trace, '_x', default)` is invisible to SLF001 and "
        "fail-open: a renamed field degrades silently. Read the declared "
        "attribute directly (SLF001-visible, AttributeError on rename) or add a "
        "typed seam like merged/_presenter.py::_rank_raw_to_final_op_labels. "
        f"Offending sites: { {p: found[p] for p in grew} }"
    )
    shrank = {
        package: (ledgered, counts.get(package, 0))
        for package, ledgered in _TRACE_REACHIN_LEDGER.items()
        if counts.get(package, 0) < ledgered
    }
    assert not shrank, (
        f"Reach-in ledger rows are stale (package: ledgered -> found): {shrank}. "
        "Lower the counts in this test -- the ledger is shrink-only, so a fixed "
        "reach-in must be recorded to keep the ratchet tight."
    )


def test_fail_closed_packages_hold_zero_reachins() -> None:
    """`merged/` and `distributed/` carry no silent-default Trace reads."""

    found = _scan_package()
    offenders = {
        package: found[package] for package in sorted(_FAIL_CLOSED_PACKAGES) if package in found
    }
    assert not offenders, (
        f"Fail-closed package(s) grew a private string reach-in: {offenders}. "
        "These packages refuse typed when a rank core cannot answer; a "
        "silent-default read contradicts that contract (b5 R45-2)."
    )
    assert not _FAIL_CLOSED_PACKAGES & set(_TRACE_REACHIN_LEDGER), (
        "A fail-closed package must not have a ledger row -- it is held at zero."
    )


def test_gate_scanner_detects_planted_offenders() -> None:
    """Planted positives/negatives: the scanner sees the four builtin forms only."""

    planted = ast.parse(
        "a = getattr(trace, '_raw_to_final_op_labels', {})\n"
        "b = hasattr(log, '_runnable')\n"
        "setattr(ml, '_validation_replay_status', 'ok')\n"
        "delattr(state.trace, '_mlx_module_stack')\n"
        # Negatives: own state, public name, dunder, non-trace base, non-literal.
        "c = getattr(self, '_private', None)\n"
        "d = getattr(trace, 'public_field', None)\n"
        "e = getattr(trace, '__class__', None)\n"
        "f = getattr(some_module, '_private', None)\n"
        "g = getattr(trace, name, None)\n"
    )
    attrs = [attr for _, _, attr in _trace_reachins(planted)]
    assert attrs == [
        "_raw_to_final_op_labels",
        "_runnable",
        "_validation_replay_status",
        "_mlx_module_stack",
    ]


class _MissingSeamTrace:
    """A rank-core stand-in that never had the declared label seam."""


class _WrongTypeSeamTrace:
    _raw_to_final_op_labels = ["not", "a", "mapping"]


class _UnresolvableTrace:
    """Carries the seam but cannot resolve the label it maps to."""

    _raw_to_final_op_labels = {"raw_1": "final_1"}

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)


class _ResolvingTrace:
    _raw_to_final_op_labels = {"raw_1": "final_1"}
    sentinel = object()

    def __getitem__(self, key: str) -> object:
        assert key == "final_1"
        return self.sentinel


class _Join:
    """Minimal duck-typed join; the real path is pinned by test_merged_gloo.py."""

    key = ("digest", 0, "channel", 0)
    presence = (0,)

    def op_labels_raw(self, rank: int) -> tuple[str, ...]:
        assert rank == 0
        return ("raw_1",)


class _Presenter:
    """Minimal duck-typed presenter exposing only what `join_ops` reads."""

    def __init__(self, trace: object) -> None:
        self.ranks = {0: trace}

    def _require_members(self, surface: str) -> None:
        """No-op release guard: join_ops() calls it (f96c67cc); the stub never releases."""


@pytest.mark.smoke
def test_declared_seam_refuses_typed_when_absent_or_wrong_type() -> None:
    """The raw->final seam is fail-closed: absence and wrong type both refuse."""

    with pytest.raises(MergeInputError) as absent:
        _rank_raw_to_final_op_labels(3, _MissingSeamTrace())
    assert absent.value.fields["code"] == "merged_schema_invalid"
    assert absent.value.fields["rank"] == 3

    with pytest.raises(MergeInputError):
        _rank_raw_to_final_op_labels(0, _WrongTypeSeamTrace())

    assert _rank_raw_to_final_op_labels(0, _ResolvingTrace()) == {"raw_1": "final_1"}


def test_join_ops_refuses_an_unresolvable_recorded_boundary_label() -> None:
    """A recorded back-reference the core cannot resolve refuses, never drops."""

    with pytest.raises(MergeInputError) as caught:
        MergedTrace.join_ops(_Presenter(_UnresolvableTrace()), _Join())  # type: ignore[arg-type]
    fields = caught.value.fields
    assert fields["code"] == "merged_schema_invalid"
    assert (fields["rank"], fields["raw_label"], fields["final_label"]) == (0, "raw_1", "final_1")

    resolved = MergedTrace.join_ops(_Presenter(_ResolvingTrace()), _Join())  # type: ignore[arg-type]
    assert resolved == {0: (_ResolvingTrace.sentinel,)}
