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
from torchlens.conformance._scorecard import scorecard_from_report
from torchlens.errors import ConfigurationError

#: Cells a full run over a two-family roster defines: 3 C0, 2x3 + 1 C1, 2x3 C2.
_FULL_DENOMINATOR = 16

#: SHA-256 of the complete canonical bytes of the golden scorecard below, and
#: the digest the scorecard embeds (over its digest-free form). Both were
#: derived from a hand-written payload serialized with ``json.dumps`` alone.
_GOLDEN_BYTES_SHA256 = (
    "81603e35bcc9118bb98db51753d9ad2b218e8ab1eac72684f70ba5782d169304"  # pragma: allowlist secret
)
_GOLDEN_EMBEDDED_SHA256 = (
    "28cbbe86eed6e10b35251954e96371fe1706848fdab91c50bc5e70358ff38b92"  # pragma: allowlist secret
)


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
        claim_eligible=False,
        environment={"torchlens_version": "0.0.0"},
        attestation_sha256="",
    )


def test_golden_scorecard_canonical_bytes() -> None:
    """The canonical bytes and the embedded digest match the independent derivation."""

    card = scorecard_from_report(_golden_report(), ("fn", "mod"), framework_version="1.2.3")
    text = card.to_canonical_json()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == _GOLDEN_BYTES_SHA256
    assert card.sha256 == _GOLDEN_EMBEDDED_SHA256
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
    assert report.funnel["skipped"] == len(model_rows) == _FULL_DENOMINATOR - 3
    assert report.earned_tier == "C0"
    assert "conformant" not in claim_string(report)

    card = preview_scorecard("paddle", workdir=tmp_path)
    assert card.counts["not_run"] == _FULL_DENOMINATOR - 3
    assert card.counts["denominator"] == _FULL_DENOMINATOR
    assert card.highest_tier == "C0"
    assert "conformant" not in card.to_canonical_json().lower()


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
        claim_eligible=False,
        environment={},
        attestation_sha256="",
    )
    card = scorecard_from_report(report, ("mod",), framework_version="")
    states = {(cell.tier, cell.case_id): cell.state for cell in card.cells}
    assert states[("C1", "c1_capture_completes")] == "unsupported"
    assert states[("C1", "c1_reference_parity")] == "not_run"
    assert card.counts["unsupported"] == 1
    assert card.highest_tier == ""


def _assert_full_ladder(card: PreviewScorecard, backend: str) -> None:
    """A live pack passes every one of its cells and publishes C2."""

    assert card.backend == backend
    assert card.framework_version
    failing = [cell for cell in card.cells if cell.state != "passed"]
    assert not failing, failing
    assert card.counts["denominator"] == _FULL_DENOMINATOR
    assert card.highest_tier == "C2"
    assert "conformant" not in card.to_canonical_json().lower()


@pytest.mark.heavy
@pytest.mark.backend_jax
def test_jax_pack_full_ladder(tmp_path: Path) -> None:
    """The jax pack passes C0-C2 on its config-built roster."""

    pytest.importorskip("jax")
    pytest.importorskip("equinox")
    _assert_full_ladder(preview_scorecard("jax", workdir=tmp_path), "jax")


@pytest.mark.heavy
def test_tf_pack_full_ladder(tmp_path: Path) -> None:
    """The tf pack passes C0-C2 on its config-built roster."""

    pytest.importorskip("tensorflow")
    _assert_full_ladder(preview_scorecard("tf", workdir=tmp_path), "tf")


@pytest.mark.heavy
@pytest.mark.backend_mlx
@pytest.mark.xfail(
    raises=AssertionError,
    reason=(
        "MLX backend defect found by this pack: tracing an mlx.nn.Sequential(Linear, ReLU, "
        "Linear) whose parameters are still unevaluated leaves the model unusable; the "
        "direct call after the trace raises '[eval] Attempting to eval an array without a "
        "primitive' (c1_capture_completes on mlp_module). Remove this marker with the fix."
    ),
)
def test_mlx_pack_full_ladder(tmp_path: Path) -> None:
    """The mlx pack passes C0-C2 on its config-built roster."""

    pytest.importorskip("mlx.core")
    _assert_full_ladder(preview_scorecard("mlx", workdir=tmp_path), "mlx")


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
