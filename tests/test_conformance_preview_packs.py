"""Conformance packs and scorecards for the preview backends (F32 B4).

Framework-free cases run everywhere: every preview backend resolves to its
adapter, an absent framework turns every model case into a named skip and
every scorecard cell into ``not_run`` (all still in the denominator), a typed
backend refusal becomes an ``unsupported`` cell, and the canonical scorecard
bytes are pinned against a payload derived independently of the folding code.
The live packs run each backend's full C0-C2 ladder on its config-built
roster when the framework is installed (``backend_<name>`` markers).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from torchlens.backends.registry import BackendUnsupportedError
from torchlens.conformance import (
    PREVIEW_BACKENDS,
    CaseResult,
    ConformanceReport,
    PreviewScorecard,
    _preview_adapters,
    claim_string,
    preview_scorecard,
    run_conformance,
)
from torchlens.conformance._adapters import RosterModel, resolve_adapter
from torchlens.conformance._cases import run_c1_cases
from torchlens.conformance._protocol_cases import (
    c1_row_case_ids,
    c1_tail_case_ids,
    c2_row_case_ids,
)
from torchlens.conformance._scorecard import SCORECARD_VERSION, scorecard_from_report
from torchlens.errors import ConfigurationError

#: Cells a full run over a two-family protocol-free roster defines: 3 C0, 2x3 + 1 C1, 2x3 C2.
_FULL_DENOMINATOR = 16
#: Every registered backend defines the four protocol-level C0 cells.
_PROTOCOL_C0_CELLS = 4
_PREVIEW_C0_CELLS = 3 + _PROTOCOL_C0_CELLS
#: Per preview roster row at C1: capture, inventory, parity, repeat, graph, shape, gap,
#: validation passes, validation names reason; at C2 the v1 trio plus payload equality
#: and validation after load. The refusal tail varies per backend (its False flags and
#: the public options it does not implement), so the denominator is enumerated per backend.
_PREVIEW_C1_ROW_CELLS = 9
_PREVIEW_C2_ROW_CELLS = 5
_PREVIEW_DENOMINATOR_FLOOR = (
    _PREVIEW_C0_CELLS + 2 * _PREVIEW_C1_ROW_CELLS + 1 + 2 * _PREVIEW_C2_ROW_CELLS
)


def _preview_denominator(backend: str) -> int:
    """Every cell the suite defines for ``backend``'s own roster."""

    adapter = resolve_adapter(backend)
    roster = adapter.default_roster()
    cells = _PREVIEW_C0_CELLS + len(c1_tail_case_ids(adapter))
    cells += sum(len(c1_row_case_ids(adapter, row)) for row in roster)
    cells += sum(len(c2_row_case_ids(adapter, row)) for row in roster)
    assert cells >= _PREVIEW_DENOMINATOR_FLOOR
    return cells


def _golden_report() -> ConformanceReport:
    """A fixed report exercising every cell state and the claim-word redaction."""

    built = "config_built"
    rows = (
        CaseResult("c0_capability_flags_boolean", "C0", "passed"),
        CaseResult("c0_provider_identity", "C0", "passed"),
        CaseResult("c0_surface_complete", "C0", "passed"),
        CaseResult("c1_capture_completes", "C1", "passed", "fn", built),
        CaseResult("c1_reference_parity", "C1", "passed", "fn", built),
        CaseResult("c1_deterministic_repeat", "C1", "passed", "fn", built),
        CaseResult(
            "c1_capture_completes",
            "C1",
            "failed",
            "mod",
            built,
            "unsupported[backend_unsupported]: no module roots",
        ),
        CaseResult("c1_capabilities_consumed", "C1", "passed"),
        CaseResult("c2_roundtrip_completes", "C2", "passed", "fn", built),
        CaseResult("c2_reload_is_distinct", "C2", "passed", "fn", built),
        CaseResult(
            "c2_structure_equal",
            "C2",
            "failed",
            "fn",
            built,
            "reloaded structure differs from live",
        ),
        CaseResult(
            "c2_roundtrip_completes",
            "C2",
            "skipped",
            "mod",
            built,
            "framework_not_installed: TorchLens-conformant x",
        ),
    )
    return ConformanceReport(
        suite_version="torchlens.conformance.v1",
        backend="golden",
        tiers=("C0", "C1", "C2"),
        results=rows,
        funnel={},
        realism={},
        earned_tier="",
        is_claim_eligible=False,
        environment={"torchlens_version": "0.0.0"},
        attestation_sha256="",
    )


def _golden_payload() -> dict[str, Any]:
    """The golden scorecard's canonical payload, written by hand (no folding code involved)."""

    def cell(tier: str, case_id: str, family: str, state: str, detail: str = "") -> dict[str, str]:
        group = "config_built" if family else ""
        return {
            "tier": tier,
            "case_id": case_id,
            "family": family,
            "state": state,
            "detail": detail,
            "group": group,
        }

    not_reached = "not reached: {} did not pass"
    skipped = "framework_not_installed: TorchLens-[claim word] x"
    return {
        "scorecard_version": SCORECARD_VERSION,
        "backend": "golden",
        "framework_version": "1.2.3",
        "suite_version": "torchlens.conformance.v1",
        "torchlens_version": "0.0.0",
        "cells": [
            cell("C0", "c0_capability_flags_boolean", "", "passed"),
            cell("C0", "c0_provider_identity", "", "passed"),
            cell("C0", "c0_surface_complete", "", "passed"),
            cell("C1", "c1_capture_completes", "fn", "passed"),
            cell("C1", "c1_reference_parity", "fn", "passed"),
            cell("C1", "c1_deterministic_repeat", "fn", "passed"),
            cell(
                "C1",
                "c1_capture_completes",
                "mod",
                "unsupported",
                "unsupported[backend_unsupported]: no module roots",
            ),
            cell(
                "C1",
                "c1_reference_parity",
                "mod",
                "not_run",
                not_reached.format("c1_capture_completes"),
            ),
            cell(
                "C1",
                "c1_deterministic_repeat",
                "mod",
                "not_run",
                not_reached.format("c1_capture_completes"),
            ),
            cell("C1", "c1_capabilities_consumed", "", "passed"),
            cell("C2", "c2_roundtrip_completes", "fn", "passed"),
            cell("C2", "c2_reload_is_distinct", "fn", "passed"),
            cell(
                "C2", "c2_structure_equal", "fn", "failed", "reloaded structure differs from live"
            ),
            cell("C2", "c2_roundtrip_completes", "mod", "not_run", skipped),
            cell(
                "C2",
                "c2_reload_is_distinct",
                "mod",
                "not_run",
                not_reached.format("c2_roundtrip_completes"),
            ),
            cell(
                "C2",
                "c2_structure_equal",
                "mod",
                "not_run",
                not_reached.format("c2_roundtrip_completes"),
            ),
        ],
        "counts": {"denominator": 16, "failed": 1, "not_run": 5, "passed": 9, "unsupported": 1},
        "highest_tier": "C0",
        "maturity": "",
        "framework_requirement": {},
        "capabilities": {},
        "roster_groups": {"config_built": 2},
        "graduation": [],
    }


def test_golden_scorecard_canonical_bytes() -> None:
    """The canonical bytes and the embedded digest match the independent derivation."""

    import json

    roster = tuple(
        RosterModel(family, "config_built", lambda: (abs, -1.0)) for family in ("fn", "mod")
    )
    card = scorecard_from_report(_golden_report(), roster, framework_version="1.2.3")
    text = card.to_json()
    payload = _golden_payload()
    digest_free = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected_digest = hashlib.sha256(digest_free.encode("utf-8")).hexdigest()
    payload["sha256"] = expected_digest
    expected_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert text == expected_text
    assert card.sha256 == expected_digest
    assert card.counts == {
        "passed": 9,
        "failed": 1,
        "unsupported": 1,
        "not_run": 5,
        "denominator": _FULL_DENOMINATOR,
    }
    assert card.highest_tier == "C0"
    assert "conformant" not in text.lower()


def test_every_preview_backend_resolves_to_its_adapter() -> None:
    """Each preview name resolves; an unknown name still refuses typed."""

    for backend in PREVIEW_BACKENDS:
        assert resolve_adapter(backend).name == backend
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_adapter("mystery-provider")
    assert excinfo.value.fields["code"] == "conformance_backend_unknown"


def test_absent_framework_skips_every_case_and_scores_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A framework that is not importable earns no tier beyond C0 and no claim."""

    absent = _preview_adapters.PreviewFramework("paddle", "torchlens_no_such_framework", "paddle")
    monkeypatch.setitem(_preview_adapters.PREVIEW_FRAMEWORKS, "paddle", absent)
    report = run_conformance("paddle", tiers=("C0", "C1", "C2"), workdir=tmp_path)
    model_rows = [row for row in report.results if row.tier != "C0"]
    assert model_rows and all(row.outcome == "skipped" for row in model_rows)
    assert all("install torchlens[paddle]" in row.detail for row in model_rows)
    denominator = _preview_denominator("paddle")
    assert report.funnel["skipped"] == len(model_rows) == denominator - _PREVIEW_C0_CELLS
    assert report.earned_tier == "C0"
    assert "conformant" not in claim_string(report)

    card = preview_scorecard("paddle", workdir=tmp_path)
    assert card.counts["not_run"] == denominator - _PREVIEW_C0_CELLS
    assert card.counts["denominator"] == denominator
    assert card.highest_tier == "C0"
    assert "conformant" not in card.to_json().lower()


class _RefusingAdapter:
    """Provider double whose capture refuses with a typed backend error."""

    name = "refusing"
    claim_eligible = False

    def capabilities(self) -> dict[str, bool]:
        """No gated capability is declared."""

        return {"capture": True, "durable_save": False, "validation_replay": False}

    def trace(self, model: Any, example_input: Any) -> Any:
        """Refuse the way a backend lacking a capability does."""

        raise BackendUnsupportedError("module roots are not supported")

    def reference_output(self, model: Any, example_input: Any) -> Any:
        """Never reached."""

        return model(example_input)


def test_typed_backend_refusal_scores_unsupported() -> None:
    """A BackendUnsupportedError becomes an unsupported cell, still in the denominator."""

    roster = (RosterModel("mod", "config_built", lambda: (abs, -1.0)),)
    rows = run_c1_cases(_RefusingAdapter(), roster)
    assert rows[0].outcome == "failed"
    assert rows[0].detail.startswith("unsupported[backend_unsupported]: ")
    report = ConformanceReport(
        suite_version="torchlens.conformance.v1",
        backend="refusing",
        tiers=("C0", "C1"),
        results=tuple(rows),
        funnel={},
        realism={},
        earned_tier="",
        is_claim_eligible=False,
        environment={},
        attestation_sha256="",
    )
    card = scorecard_from_report(report, roster, framework_version="")
    states = {(cell.tier, cell.case_id): cell.state for cell in card.cells}
    assert states[("C1", "c1_capture_completes")] == "unsupported"
    assert states[("C1", "c1_reference_parity")] == "not_run"
    assert card.counts["unsupported"] == 1
    assert card.highest_tier == ""


#: MLX's red cells on its packet roster, keyed ``(case_id, family)``, with the cause
#: substring each detail must carry. One cause, not an MLX capture gap: loaded
#: non-torch traces drop the runtime capture replay needs. The list holding a
#: Sequential's children has its own module record (``block.layers``, class
#: ``list``), so ``mod_nested_residual`` passes the module-hierarchy invariant. The
#: shared preview option policy refuses with the per-cause code, so every
#: ``c1_refusal_*`` cell passes.
_MLX_ROSTER_FAMILIES: tuple[str, ...] = (
    "fn_chain_branch",
    "fn_multi_output",
    "fn_loop3",
    "fn_norm_softmax",
    "fn_reshape_transpose",
    "fn_where_clip",
    "mod_mlp",
    "mod_shared_twice",
    "mod_cnn",
    "mod_nested_residual",
)
MLX_KNOWN_RED_CELLS: dict[tuple[str, str], str] = {
    **{
        ("c2_validation_after_load", family): "runtime capture data was stripped"
        for family in _MLX_ROSTER_FAMILIES
    },
}


def _assert_full_ladder(card: PreviewScorecard, backend: str) -> None:
    """A live pack passes every one of its cells and publishes C2."""

    assert card.backend == backend
    assert card.framework_version
    failing = [cell for cell in card.cells if cell.state != "passed"]
    assert not failing, failing
    assert card.counts["denominator"] == _preview_denominator(backend)
    assert card.highest_tier == "C2"
    assert "conformant" not in card.to_json().lower()


def _after_load_reds(families: tuple[str, ...]) -> dict[tuple[str, str], str]:
    """Loaded non-torch traces drop the runtime capture replay needs (a format gap)."""

    return {
        ("c2_validation_after_load", family): "runtime capture data was stripped"
        for family in families
    }


def _known_red_cells(
    refusals: dict[str, str], families: tuple[str, ...], extra: dict[tuple[str, str], str]
) -> dict[tuple[str, str], str]:
    """One backend's red cells keyed ``(case_id, family)`` with each cause substring."""

    return {
        **{(case_id, ""): cause for case_id, cause in refusals.items()},
        **_after_load_reds(families),
        **extra,
    }


def _assert_named_reds(
    pack: tuple[str, str],
    tmp_path: Path,
    known: dict[tuple[str, str], str],
    rows: int,
    tier: str,
) -> None:
    """The pack's non-passed cells equal ``known`` exactly, each failed with its cause.

    ``pack`` is the ``(backend, framework version)`` under test.
    """

    from torchlens.conformance import _roster_for

    backend, version = pack
    adapter = resolve_adapter(backend)
    roster = _roster_for(adapter, None)
    report = run_conformance(backend, tiers=("C0", "C1", "C2"), roster=roster, workdir=tmp_path)
    card = scorecard_from_report(report, roster, framework_version=version, adapter=adapter)
    assert card.backend == backend
    assert card.roster_groups == {"config_built": rows}
    red = {(cell.case_id, cell.family): cell for cell in card.cells if cell.state != "passed"}
    assert set(red) == set(known), sorted(set(red) ^ set(known))
    for key, cause in known.items():
        assert red[key].state == "failed", red[key]
        assert cause in red[key].detail, (key, red[key].detail)
    assert card.counts["denominator"] == _preview_denominator(backend)
    # The ladder stops below the first tier holding a red cell.
    assert card.highest_tier == tier
    assert "conformant" not in card.to_json().lower()


#: JAX's red cells on its packet roster (switch off). One cause, not a JAX
#: capture gap: loaded traces without the runtime capture replay needs. Every
#: C0 and C1 cell passes, so the pack earns C1.
def _jax_known_red_cells() -> dict[tuple[str, str], str]:
    """JAX's named red cells, built from the roster module's family list."""

    from torchlens.conformance._rosters import jax as jax_roster

    return _known_red_cells({}, jax_roster.CONFIG_BUILT_FAMILIES, {})


#: TensorFlow's red cells on its packet roster (switch off). The JAX cause plus
#: the Keras module rows: their parameter reads (``ReadVariableOp``) are effect
#: regions validation reports ``unverified`` (a ruled honest state, never
#: exempted), so the ladder stops at C0.
_TF_KERAS_MODULE_ROWS: tuple[str, ...] = (
    "mlp_module",
    "shared_weight_module",
    "cnn_module",
    "nested_residual_module",
)


def _tf_known_red_cells() -> dict[tuple[str, str], str]:
    """TensorFlow's named red cells, built from the roster module's family list."""

    from torchlens.conformance._rosters import tf as tf_roster

    keras = {
        ("c1_validation_passes", family): "effect regions unverified"
        for family in _TF_KERAS_MODULE_ROWS
    }
    return _known_red_cells({}, tf_roster.CONFIG_BUILT_FAMILIES, keras)


@pytest.mark.heavy
@pytest.mark.backend_jax
def test_jax_pack_earns_c1_with_its_named_red_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The jax pack on its nine-row packet roster earns C1; every red cell is named.

    A new red or a fixed one both fail here; the fix removes its entry (and
    rewrites the shipped scorecard) in the same change.
    """

    jax = pytest.importorskip("jax")
    pytest.importorskip("equinox")
    pytest.importorskip("flax")
    from torchlens.conformance._rosters import ECOSYSTEM_ENV

    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    _assert_named_reds(("jax", str(jax.__version__)), tmp_path, _jax_known_red_cells(), 9, "C1")


@pytest.mark.heavy
@pytest.mark.tf_backend
def test_tf_pack_earns_c0_with_its_named_red_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tf pack on its nine-row packet roster earns C0; every red cell is named.

    A new red or a fixed one both fail here; the fix removes its entry (and
    rewrites the shipped scorecard) in the same change.
    """

    tf = pytest.importorskip("tensorflow")
    from torchlens.conformance._rosters import ECOSYSTEM_ENV

    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    _assert_named_reds(("tf", str(tf.__version__)), tmp_path, _tf_known_red_cells(), 9, "C0")


@pytest.mark.heavy
@pytest.mark.backend_mlx
def test_mlx_pack_earns_c1_with_its_named_red_cells(tmp_path: Path) -> None:
    """The mlx pack on its ten-row packet roster earns C1; every red cell is named.

    The set of non-passed cells must equal ``MLX_KNOWN_RED_CELLS`` exactly, each
    with its cause: a new red or a fixed one both fail here, and the fix removes
    its entry (and lifts the rung) in the same change.
    """

    mx = pytest.importorskip("mlx.core")
    from torchlens.conformance import _roster_for

    adapter = resolve_adapter("mlx")
    roster = _roster_for(adapter, None)
    report = run_conformance("mlx", tiers=("C0", "C1", "C2"), roster=roster, workdir=tmp_path)
    card = scorecard_from_report(
        report, roster, framework_version=str(mx.__version__), adapter=adapter
    )
    assert card.backend == "mlx"
    assert card.roster_groups == {"config_built": 10}
    red = {(cell.case_id, cell.family): cell for cell in card.cells if cell.state != "passed"}
    assert set(red) == set(MLX_KNOWN_RED_CELLS), sorted(set(red) ^ set(MLX_KNOWN_RED_CELLS))
    for key, cause in MLX_KNOWN_RED_CELLS.items():
        assert red[key].state == "failed", red[key]
        assert cause in red[key].detail, (key, red[key].detail)
    assert card.counts["denominator"] == _preview_denominator("mlx")
    # Only the after-load reds remain and they sit in C2, so the ladder stops at C1.
    assert card.highest_tier == "C1"
    # C1 needs the pretrained-family floor; the config-built roster has none.
    with pytest.raises(ConfigurationError) as excinfo:
        claim_string(report)
    assert excinfo.value.fields["code"] == "conformance_claim_unearned"
    assert "conformant" not in card.to_json().lower()


@pytest.mark.heavy
@pytest.mark.backend_tinygrad
def test_tinygrad_pack_full_ladder(tmp_path: Path) -> None:
    """The tinygrad pack passes C0-C2 on its config-built roster."""

    pytest.importorskip("tinygrad")
    _assert_full_ladder(preview_scorecard("tinygrad", workdir=tmp_path), "tinygrad")


@pytest.mark.heavy
@pytest.mark.backend_paddle
def test_paddle_pack_full_ladder(tmp_path: Path) -> None:
    """The paddle pack passes C0-C2 on its config-built roster."""

    pytest.importorskip("paddle")
    _assert_full_ladder(preview_scorecard("paddle", workdir=tmp_path), "paddle")
