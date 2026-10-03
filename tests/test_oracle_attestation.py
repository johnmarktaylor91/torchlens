"""Attestation that the byte-identity oracles are ENFORCEABLE, not skippable.

The b10 audit found the oracle families could go green without ever
enforcing anything: off-canonical environments recorded-and-skipped forever
(ephemeral CI checkouts re-recorded every run), update flags produced
vacuous compare-to-self passes, and nothing asserted the canonical goldens
even exist where the enforcing tier runs (R78-4, R78-8, R53-3). These tests
pin the governance itself:

* every oracle family's CANONICAL golden inventory is committed;
* the fail-closed resolver refuses to self-baseline (record needs the
  explicit opt-in, CI never writes, canonical-missing is a hard failure);
* golden update/regen flags hard-error under CI.

The per-leg executed-floor (scripts/check_ci_executed_tests.py in the
workflows) is the runtime half: with these two together a skipping oracle
is visible instead of green.
"""

from __future__ import annotations

from pathlib import Path

import _oracle_env
import pytest
from _oracle_env import (
    ENFORCE_ENV_VAR,
    REASON_ENV_VAR,
    RECORD_ENV_VAR,
    env_fingerprint,
    flag_armed,
    golden_mutation_flags_armed_under_ci,
    guard_wrap_state_for_golden_update,
    require_env_golden,
    require_update_reason,
    resolve_env_golden,
    write_provenance,
)

_TESTS_DIR = Path(__file__).resolve().parent

#: Every byte-oracle family's canonical committed golden inventory.
#: A missing entry here means the enforcing tier CANNOT enforce — the
#: record-and-skip path (now fail-closed) would have been the only outcome.
_CANONICAL_GOLDENS: dict[str, tuple[Path, ...]] = {
    "surface_oracle": tuple(
        _TESTS_DIR / "surface_oracle" / "goldens" / f"{axis}.json"
        for axis in (
            "plain_cnn",
            "train_batchnorm",
            "recurrent",
            "conditional",
            "in_place",
            "tiny_transformer",
        )
    ),
    "godobject_viz": tuple(
        _TESTS_DIR / "godobject_oracle" / "goldens" / f"viz_{key}_{mode}.gv"
        for key in ("viz_cnn", "viz_recurrent")
        for mode in ("unrolled", "rolled")
    ),
    "godobject_legacy": (
        _TESTS_DIR / "godobject_oracle" / "goldens" / "legacy_baseline_cnn.tlspec",
        _TESTS_DIR / "godobject_oracle" / "goldens" / "legacy_baseline_cnn_runnable.tlspec",
        _TESTS_DIR / "godobject_oracle" / "goldens" / "legacy_baseline_cnn_loaded.json",
    ),
    "state_keysets": (_TESTS_DIR / "godobject_oracle" / "goldens" / "state_keysets.json",),
    "selector_matrix": (_TESTS_DIR / "golden" / "selector_semantics_matrix.json",),
    "viz_render_identity": (_TESTS_DIR / "golden" / "viz_render_identity_oracle.json",),
    "rank_render_ir": (_TESTS_DIR / "golden" / "rank_render_ir_semantics.json",),
}


@pytest.mark.parametrize("family", sorted(_CANONICAL_GOLDENS))
def test_canonical_golden_inventory_is_committed(family: str) -> None:
    """Every oracle family's canonical goldens exist in the checkout."""

    missing = [str(path) for path in _CANONICAL_GOLDENS[family] if not path.exists()]
    assert not missing, (
        f"oracle family {family!r} is missing canonical goldens — its byte "
        f"tests cannot enforce anything: {missing}"
    )


def test_env_markers_match_a_committed_baseline() -> None:
    """Each ENV marker names the fingerprint its canonical goldens carry.

    An ENV marker naming a fingerprint nobody records under would silently
    move EVERY environment onto the (fail-closed) env-keyed path.
    """

    for goldens_dir in (
        _TESTS_DIR / "surface_oracle" / "goldens",
        _TESTS_DIR / "godobject_oracle" / "goldens",
        _TESTS_DIR / "golden",
    ):
        marker = goldens_dir / "ENV"
        assert marker.exists(), f"missing ENV marker in {goldens_dir}"
        recorded = marker.read_text().strip()
        assert recorded, f"empty ENV marker in {goldens_dir}"


def test_viz_families_commit_emitter_version_markers() -> None:
    """Viz byte families record their DOT-emitter versions (b10 R78 round-3).

    The graphviz python package directly emits the DOT bytes these goldens
    freeze (pydot additionally parses the render-identity structural digest),
    so the family-scoped fingerprint extension needs committed ``ENV-<pkg>``
    markers naming the canonical emitter versions.
    """

    for marker in (
        _TESTS_DIR / "godobject_oracle" / "goldens" / "ENV-graphviz",
        _TESTS_DIR / "golden" / "ENV-graphviz",
        _TESTS_DIR / "golden" / "ENV-pydot",
    ):
        assert marker.exists(), f"missing emitter-version marker {marker}"
        assert marker.read_text().strip(), f"empty emitter-version marker {marker}"


def _fake_env(monkeypatch: pytest.MonkeyPatch, fingerprint: str, **env: str | None) -> None:
    """Pin the fingerprint and the relevant environment variables."""

    monkeypatch.setattr(_oracle_env, "env_fingerprint", lambda extra_packages=(): fingerprint)
    for name in ("CI", RECORD_ENV_VAR, ENFORCE_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        if value is not None:
            monkeypatch.setenv(name, value)


def _goldens_dir(tmp_path: Path, canonical: str) -> Path:
    """Create a goldens dir whose ENV marker names ``canonical``."""

    goldens = tmp_path / "goldens"
    goldens.mkdir()
    (goldens / "ENV").write_text(canonical + "\n")
    return goldens


def test_missing_canonical_golden_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the canonical environment a missing golden is a hard failure."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py9.9-torch9.9.9")
    with pytest.raises(pytest.fail.Exception, match="missing canonical golden"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")


def test_missing_off_canonical_golden_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off-canonical + missing golden REFUSES instead of self-baselining."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8")
    with pytest.raises(pytest.fail.Exception, match="Refusing to self-baseline"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert not (goldens / "env-py8.8-torch8.8.8").exists(), "refusal must not write"


def test_missing_off_canonical_golden_skips_visibly_under_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI legs without a committed baseline skip with a reason, never record."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8", CI="true")
    with pytest.raises(pytest.skip.Exception, match="no committed golden"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert not (goldens / "env-py8.8-torch8.8.8").exists(), "CI must never write"


def test_enforcing_leg_fails_closed_instead_of_ci_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared enforcing leg FAILS on a missing golden even under CI.

    Without the enforce declaration a matrix torch bump silently moves the
    one enforcing CI leg off-canonical, every golden case skips, and the
    byte oracles enforce on NO leg at all while staying green (T13.1).
    """

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8", CI="true", **{ENFORCE_ENV_VAR: "1"})
    with pytest.raises(pytest.fail.Exception, match="an enforcing leg never skips"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert not (goldens / "env-py8.8-torch8.8.8").exists(), "refusal must not write"


def test_enforcing_leg_refuses_the_record_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enforcement outranks recording: enforce + record opt-in still fails.

    A leg that both enforces and records would self-baseline the very bytes
    it claims to verify — the record opt-in is for provisioning NEW
    long-lived boxes, never the enforcing leg.
    """

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8", **{RECORD_ENV_VAR: "1", ENFORCE_ENV_VAR: "1"})
    with pytest.raises(pytest.fail.Exception, match="an enforcing leg never skips"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert not (goldens / "env-py8.8-torch8.8.8").exists(), "refusal must not write"


def test_record_opt_in_returns_recordable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit record opt-in provisions exactly one recordable path."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8", **{RECORD_ENV_VAR: "1"})
    path = require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert path == goldens / "env-py8.8-torch8.8.8" / "case.json"
    assert path.parent.is_dir(), "record opt-in prepares the env directory"
    assert not path.exists()


def test_committed_env_golden_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A committed env-keyed baseline resolves for plain byte enforcement."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    env_dir = goldens / "env-py8.8-torch8.8.8"
    env_dir.mkdir()
    (env_dir / "case.json").write_text("{}\n")
    _fake_env(monkeypatch, "py8.8-torch8.8.8")
    path = require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert path == env_dir / "case.json"


def test_canonical_environment_resolves_canonical_path(tmp_path: Path) -> None:
    """Matching the ENV marker keeps enforcement on the canonical goldens."""

    goldens = _goldens_dir(tmp_path, env_fingerprint())
    path, off_canonical = resolve_env_golden(goldens, "case.json")
    assert path == goldens / "case.json"
    assert off_canonical is False


def test_missing_env_marker_refuses(tmp_path: Path) -> None:
    """No committed ENV marker is a setup bug, never a canonical blessing.

    The pre-fix resolver treated a missing base marker as "canonical", so a
    family routed through the env-keyed resolver without one had its bytes
    enforced on EVERY environment (b10 R78-4 round 5, fail-open default).
    """

    goldens = tmp_path / "goldens"
    goldens.mkdir()
    with pytest.raises(RuntimeError, match="no committed ENV marker"):
        resolve_env_golden(goldens, "case.json")


def test_recorded_env_baselines_are_gitignored() -> None:
    """A recorded env-* baseline can never ride along in a broad git add."""

    import subprocess

    probe = "tests/surface_oracle/goldens/env-py0.0-torch0.0.0/probe.json"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=_TESTS_DIR.parent,
        check=False,
    )
    assert result.returncode == 0, f"{probe} is not gitignored (b10 R78-4)"


def test_golden_mutation_flags_hard_error_under_ci() -> None:
    """The conftest guard names every armed update/regen flag under CI."""

    armed = golden_mutation_flags_armed_under_ci(
        {
            "CI": "true",
            "TORCHLENS_UPDATE_SURFACE_ORACLE": "1",
            "TORCHLENS_REGEN_EXPORT_GOLDENS": "1",
            "TL_SELECTOR_MATRIX_REGEN": "1",
            "TORCHLENS_ORACLE_RECORD_ENV": "1",
            "TORCHLENS_CACHE_DIR": "/tmp/x",
        }
    )
    assert armed == [
        "TL_SELECTOR_MATRIX_REGEN",
        "TORCHLENS_ORACLE_RECORD_ENV",
        "TORCHLENS_REGEN_EXPORT_GOLDENS",
        "TORCHLENS_UPDATE_SURFACE_ORACLE",
    ]
    assert golden_mutation_flags_armed_under_ci({"TORCHLENS_UPDATE_SURFACE_ORACLE": "1"}) == []
    assert golden_mutation_flags_armed_under_ci({"CI": "true"}) == []


def test_flag_armed_requires_exact_one() -> None:
    """Golden flags arm on the exact value "1" ONLY (b10 R78 round-3).

    The pre-fix selector-matrix read (``bool(environ.get(...))``) armed
    regeneration on NAME=0 — the value a user sets to DISARM.
    """

    name = "TORCHLENS_UPDATE_X"
    assert flag_armed({name: "1"}, name) is True
    for disarmed in ("0", "", "true", "yes", "2", " 1"):
        assert flag_armed({name: disarmed}, name) is False, disarmed
    assert flag_armed({}, name) is False


def test_update_reason_is_required_and_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Update runs refuse to proceed without a non-empty golden reason."""

    monkeypatch.delenv(REASON_ENV_VAR, raising=False)
    with pytest.raises(pytest.fail.Exception, match="requires TORCHLENS_GOLDEN_REASON"):
        require_update_reason("TORCHLENS_UPDATE_X")
    monkeypatch.setenv(REASON_ENV_VAR, "   ")
    with pytest.raises(pytest.fail.Exception, match="requires TORCHLENS_GOLDEN_REASON"):
        require_update_reason("TORCHLENS_UPDATE_X")
    monkeypatch.setenv(REASON_ENV_VAR, "r3 fix: enumerated behavior change")
    assert require_update_reason("TORCHLENS_UPDATE_X") == "r3 fix: enumerated behavior change"


def test_write_provenance_appends_full_history(tmp_path: Path) -> None:
    """PROVENANCE keeps every record: last-writer-wins erased sibling families."""

    write_provenance(tmp_path, "family_a", "TORCHLENS_UPDATE_A", "first rebaseline")
    write_provenance(tmp_path, "family_b", "TORCHLENS_UPDATE_B", "second family, same dir")
    content = (tmp_path / "PROVENANCE").read_text()
    assert "generator: family_a" in content
    assert "generator: family_b" in content
    assert "flag: TORCHLENS_UPDATE_A=1" in content
    assert "flag: TORCHLENS_UPDATE_B=1" in content
    assert "reason: first rebaseline" in content
    assert "reason: second family, same dir" in content
    assert content.count("---\n") == 1, "records are separated, none overwritten"


def test_write_provenance_records_source_identity(tmp_path: Path) -> None:
    """Every PROVENANCE record ties the rebaseline to a HEAD sha + tree state.

    Without the source line a reviewed rebaseline could not be mechanically
    tied to the code that emitted it, and a dirty-tree generation went
    undisclosed (b10 R78 round 5).
    """

    import re as _re

    write_provenance(tmp_path, "family_a", "TORCHLENS_UPDATE_A", "source-identity probe")
    content = (tmp_path / "PROVENANCE").read_text()
    match = _re.search(r"^source: (.+)$", content, flags=_re.MULTILINE)
    assert match, f"PROVENANCE record carries no source line:\n{content}"
    assert _re.fullmatch(
        r"[0-9a-f]{40} \((clean|dirty)\)|unknown \(git unavailable\)", match.group(1)
    ), match.group(1)


def test_wrap_state_guard_refuses_wrapped_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process golden generation refuses to start on wrapped torch (SF-53)."""

    import torchlens._state as tl_state

    monkeypatch.setattr(tl_state, "_is_decorated", True)
    flag = "TORCHLENS_UPDATE_WRAP_GUARD_PROBE_RED"
    _oracle_env._WRAP_GUARD_CLEARED.discard(flag)
    with pytest.raises(pytest.fail.Exception, match="UNWRAPPED torch"):
        guard_wrap_state_for_golden_update(flag)
    assert flag not in _oracle_env._WRAP_GUARD_CLEARED, "refusal must not memoize"
    _oracle_env._WRAP_GUARD_CLEARED.discard(flag)


def test_wrap_state_guard_passes_clean_then_memoizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean start passes once per family; later intra-family wraps are inherent."""

    import torchlens._state as tl_state

    flag = "TORCHLENS_UPDATE_WRAP_GUARD_PROBE_GREEN"
    _oracle_env._WRAP_GUARD_CLEARED.discard(flag)
    monkeypatch.setattr(tl_state, "_is_decorated", False)
    guard_wrap_state_for_golden_update(flag)
    # The family's OWN captures wrap torch mid-generation; that is inherent
    # to in-process families and deterministic in a fresh single-family run.
    monkeypatch.setattr(tl_state, "_is_decorated", True)
    guard_wrap_state_for_golden_update(flag)
    _oracle_env._WRAP_GUARD_CLEARED.discard(flag)


def test_extended_fingerprint_appends_emitter_versions() -> None:
    """The family-scoped fingerprint extension stays base-compatible."""

    base = env_fingerprint()
    extended = env_fingerprint(extra_packages=("graphviz",))
    assert extended.startswith(base + "-graphviz")
    assert env_fingerprint(extra_packages=()) == base
    absent = env_fingerprint(extra_packages=("definitely-not-a-real-dist",))
    assert absent == f"{base}-definitely-not-a-real-distabsent"


def test_extras_markers_gate_canonical_resolution(tmp_path: Path) -> None:
    """Emitter markers must MATCH for canonical viz-golden enforcement.

    A missing or mismatched ``ENV-<pkg>`` marker moves the family off-
    canonical (fail-closed downstream) instead of silently comparing bytes
    emitted by a different generator version.
    """

    package = "definitely-not-a-real-dist"  # _package_version -> "absent"
    goldens = tmp_path / "goldens"
    goldens.mkdir()
    (goldens / "ENV").write_text(env_fingerprint() + "\n")

    # Missing extras marker: off-canonical even though the base ENV matches.
    path, off_canonical = resolve_env_golden(goldens, "case.gv", (package,))
    assert off_canonical is True
    assert path.parent.name == f"env-{env_fingerprint((package,))}"

    # Mismatched extras marker: off-canonical.
    (goldens / f"ENV-{package}").write_text("9.9.9\n")
    _, off_canonical = resolve_env_golden(goldens, "case.gv", (package,))
    assert off_canonical is True

    # Matching extras marker: canonical path, plain enforcement.
    (goldens / f"ENV-{package}").write_text("absent\n")
    path, off_canonical = resolve_env_golden(goldens, "case.gv", (package,))
    assert off_canonical is False
    assert path == goldens / "case.gv"

    # And the base ENV mismatch still dominates.
    (goldens / "ENV").write_text("py0.0-torch0.0.0\n")
    _, off_canonical = resolve_env_golden(goldens, "case.gv", (package,))
    assert off_canonical is True
