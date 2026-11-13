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

from tests.conformance import (
    CaseResult,
    ConformanceReport,
    claim_string,
    run_conformance,
)
from tests.conformance._adapters import RosterModel, resolve_adapter
from tests.conformance._protocol_cases import (
    c1_row_case_ids,
    c1_tail_case_ids,
    c2_row_case_ids,
)
from tests.conformance._scorecard import SCORECARD_VERSION, scorecard_from_report
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


def _after_load_reds(families: tuple[str, ...]) -> dict[tuple[str, str], str]:
    """Loaded non-torch traces drop the runtime capture replay needs (a format gap)."""

    return {
        ("c2_validation_after_load", family): "runtime capture data was stripped"
        for family in families
    }


#: A red cell's expected cause substring, or ``(state, cause)`` for a cell that is
#: ``unsupported`` or ``not_run`` rather than ``failed``.
_RedCause = str | tuple[str, str]


def _known_red_cells(
    refusals: dict[str, str],
    families: tuple[str, ...],
    extra: dict[tuple[str, str], _RedCause],
) -> dict[tuple[str, str], _RedCause]:
    """One backend's red cells keyed ``(case_id, family)`` with each cause substring."""

    return {
        **{(case_id, ""): cause for case_id, cause in refusals.items()},
        **_after_load_reds(families),
        **extra,
    }


def _assert_named_reds(
    pack: tuple[str, str],
    tmp_path: Path,
    known: dict[tuple[str, str], _RedCause],
    rows: int,
    tier: str,
) -> None:
    """The pack's non-passed cells equal ``known`` exactly, each in its state with its cause.

    A bare cause means the cell ``failed``; a ``(state, cause)`` pair names another state.

    ``pack`` is the ``(backend, framework version)`` under test.
    """

    from tests.conformance import _roster_for

    backend, version = pack
    adapter = resolve_adapter(backend)
    roster = _roster_for(adapter, None)
    report = run_conformance(backend, tiers=("C0", "C1", "C2"), roster=roster, workdir=tmp_path)
    card = scorecard_from_report(report, roster, framework_version=version, adapter=adapter)
    assert card.backend == backend
    assert card.roster_groups == {"config_built": rows}
    red = {(cell.case_id, cell.family): cell for cell in card.cells if cell.state != "passed"}
    assert set(red) == set(known), sorted(set(red) ^ set(known))
    for key, expected in known.items():
        state, cause = expected if isinstance(expected, tuple) else ("failed", expected)
        assert red[key].state == state, red[key]
        assert cause in red[key].detail, (key, red[key].detail)
    assert card.counts["denominator"] == _preview_denominator(backend)
    # The ladder stops below the first tier holding a red cell.
    assert card.highest_tier == tier
    assert "conformant" not in card.to_json().lower()


#: tinygrad's red cells on its fifteen rows: the JAX after-load cause on every
#: row. The three random-init rows capture too: realize-once capture draws a
#: ``tinygrad.nn`` layer's lazy random weights in the same single realization
#: as the forward, so they pass every C0 and C1 cell and the pack earns C1.
_TINYGRAD_PROBE_ROWS: tuple[str, ...] = ("matmul_relu_function", "mlp_module")
_TINYGRAD_RANDOM_INIT_ROWS: tuple[str, ...] = (
    "nn_linear_random_init",
    "nn_conv2d_random_init",
    "nn_layernorm_random_init",
)


def _tinygrad_known_red_cells() -> dict[tuple[str, str], _RedCause]:
    """tinygrad's named red cells, built from the roster module's family lists."""

    from tests.conformance._rosters import tinygrad as tinygrad_roster

    captured = (
        _TINYGRAD_PROBE_ROWS
        + tuple(entry[0] for entry in tinygrad_roster._CONFIG_BUILT)
        + _TINYGRAD_RANDOM_INIT_ROWS
    )
    return _known_red_cells({}, captured, {})


#: JAX's red cells on its packet roster (switch off). One cause, not a JAX
#: capture gap: loaded traces without the runtime capture replay needs. Every
#: C0 and C1 cell passes, so the pack earns C1.
def _jax_known_red_cells() -> dict[tuple[str, str], str]:
    """JAX's named red cells, built from the roster module's family list."""

    from tests.conformance._rosters import jax as jax_roster

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

    from tests.conformance._rosters import tf as tf_roster

    keras = {
        ("c1_validation_passes", family): "effect regions unverified"
        for family in _TF_KERAS_MODULE_ROWS
    }
    return _known_red_cells({}, tf_roster.CONFIG_BUILT_FAMILIES, keras)


@pytest.mark.rare
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
    from tests.conformance._rosters import ECOSYSTEM_ENV

    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    _assert_named_reds(("jax", str(jax.__version__)), tmp_path, _jax_known_red_cells(), 9, "C1")


@pytest.mark.rare
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
    from tests.conformance._rosters import ECOSYSTEM_ENV

    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    _assert_named_reds(("tf", str(tf.__version__)), tmp_path, _tf_known_red_cells(), 9, "C0")


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_mlx
def test_mlx_pack_earns_c1_with_its_named_red_cells(tmp_path: Path) -> None:
    """The mlx pack on its ten-row packet roster earns C1; every red cell is named.

    The set of non-passed cells must equal ``MLX_KNOWN_RED_CELLS`` exactly, each
    with its cause: a new red or a fixed one both fail here, and the fix removes
    its entry (and lifts the rung) in the same change.
    """

    mx = pytest.importorskip("mlx.core")
    from tests.conformance import _roster_for

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


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_tinygrad
def test_tinygrad_pack_earns_c1_with_its_named_red_cells(tmp_path: Path) -> None:
    """The tinygrad pack on its fifteen-row roster earns C1; every red cell is named.

    A new red or a fixed one both fail here; the fix removes its entry (and
    rewrites the shipped scorecard) in the same change.
    """

    from importlib import metadata

    pytest.importorskip("tinygrad")
    pack = ("tinygrad", metadata.version("tinygrad"))
    _assert_named_reds(pack, tmp_path, _tinygrad_known_red_cells(), 15, "C1")


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_paddle
def test_paddle_pack_earns_c1_with_its_named_red_cells(tmp_path: Path) -> None:
    """The paddle pack on its twelve-row config-built roster earns C1; every red cell is named.

    The only reds are the after-load cells JAX and MLX also name: loaded non-torch
    traces drop the runtime capture replay needs. A new red or a fixed one both
    fail here; the fix removes its entry (and rewrites the shipped scorecard) in
    the same change.
    """

    paddle = pytest.importorskip("paddle")
    from tests.conformance._rosters.paddle import config_built_specs

    families = tuple(spec.family for spec in config_built_specs())
    known = _known_red_cells({}, families, {})
    _assert_named_reds(("paddle", str(paddle.__version__)), tmp_path, known, 12, "C1")
