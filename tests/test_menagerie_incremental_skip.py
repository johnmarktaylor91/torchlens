"""Gate tests for the incremental-skip default and the force/force-full flags.

Sprint bucket D (FINAL_PLAN D4/D5). Incremental skip is the DEFAULT: a model
whose CURRENT identity tuple matches a genuine current-identity ``passed`` row in
the cascade-aware ``current_verification_real`` view is provably unchanged and is
skipped; everything else (changed identity, a degraded ``input_scale=NULL``
component, ANY non-``passed`` status, a cascade-suppressed row) re-validates.

This is a TRIPWIRE: a false skip would hide exactly the capture regression
validation exists to catch, so these gates assert the exact eligibility
boundaries (a)-(f) called out in the bucket-D brief:

* (a) matching identity + a current real pass -> SKIPPED;
* (b) a changed source/env/lock hash -> NOT skipped;
* (c) a degraded (``input_scale=NULL``) row -> NOT skipped;
* (d) a current ``failed``/``timeout`` row -> NOT skipped (retried);
* (e) ``--force-full`` re-validates a would-be-skipped model;
* (f) ``--force`` (targeted) re-validates only the selected models.
"""

from __future__ import annotations

from pathlib import Path

from menagerie.ledger import (
    CASCADE_ARTIFACT_ERROR_CLASS,
    LEGACY_UNKNOWN,
    VerificationRun,
    VerificationTarget,
    append_verification_run,
    base_env_hash,
    base_lock_hash,
    connect,
    incremental_skip_stable_ids,
    torchlens_source_hash,
)
from menagerie.catalog import CatalogRow
from menagerie.run_all import build_parser as run_all_parser
from menagerie.validate_menagerie import (
    _incremental_skip_stable_ids,
    _torchlens_version,
    build_parser as validate_parser,
)


def _run(**overrides: object) -> VerificationRun:
    """Build a compact passing verification run fixture.

    Defaults to a genuine current-identity ``passed`` row for ``m1`` at
    ``input_scale=1.0`` -- the exact shape that is skip-eligible.

    Parameters
    ----------
    overrides:
        Field overrides for the default run.

    Returns
    -------
    VerificationRun
        Test verification run.
    """

    data: dict[str, object] = {
        "stable_id": "m1",
        "recipe_revision_sha256": "recipe-a",
        "name": "ToyNet",
        "zoo": "unit-zoo",
        "variant": "",
        "scope": "forward",
        "status": "passed",
        "forward_pass": 1,
        "backward_pass": None,
        "backward_na_reason": None,
        "metadata_ok": 1,
        "n_ops": 4,
        "graph_shape_hash": "shape-a",
        "svg_sha256": None,
        "torchlens_version": "tl-current",
        "torch_version": "torch-test",
        "python_version": "py-test",
        "device_requested": "cpu",
        "device_actual": "cpu",
        "env_hash": base_env_hash(),
        "lock_hash": base_lock_hash(),
        "torchlens_source_hash": torchlens_source_hash(),
        "input_scale": 1.0,
        "runner_host": "unit-host",
        "started_at": "2026-06-22T00:00:00+00:00",
        "finished_at": "2026-06-22T00:00:01+00:00",
        "duration_sec": 1.0,
        "error_class": None,
        "error_message": None,
        "run_id": "run-a",
    }
    data.update(overrides)
    return VerificationRun(**data)  # type: ignore[arg-type]


def _cascade_artifact(**overrides: object) -> VerificationRun:
    """Build a synthesized pre-fix batch-cascade artifact row.

    Mirrors the frozen 290 job-6014456 rows: ``status=failed`` with
    ``CASCADE_ARTIFACT_ERROR_CLASS``, zero duration, started==finished, NULL
    ``n_ops``/``graph_shape_hash``, written on a non-cluster orchestrator host.

    Parameters
    ----------
    overrides:
        Field overrides for the default artifact.

    Returns
    -------
    VerificationRun
        Cascade-artifact verification run.
    """

    data: dict[str, object] = {
        "status": "failed",
        "forward_pass": None,
        "metadata_ok": None,
        "n_ops": None,
        "graph_shape_hash": None,
        "svg_sha256": None,
        "duration_sec": 0.0,
        "started_at": "2026-06-26T08:37:54+00:00",
        "finished_at": "2026-06-26T08:37:54+00:00",
        "runner_host": "zmachine",
        "error_class": CASCADE_ARTIFACT_ERROR_CLASS,
        "error_message": (
            "cluster job failed; sbatch --wait returncode=1; Submitted batch job 6014456"
        ),
    }
    data.update(overrides)
    return _run(**data)


def _target(**overrides: object) -> VerificationTarget:
    """Build the current identity target that matches the default pass row.

    Parameters
    ----------
    overrides:
        Field overrides for the default target.

    Returns
    -------
    VerificationTarget
        Current identity target.
    """

    data: dict[str, object] = {
        "recipe_revision_sha256": "recipe-a",
        "torchlens_source_hash": torchlens_source_hash(),
        "env_hash": base_env_hash(),
        "lock_hash": base_lock_hash(),
        "device_requested": "cpu",
        "scope": "forward",
    }
    data.update(overrides)
    return VerificationTarget(**data)  # type: ignore[arg-type]


def _skip(conn, *, targets=None, input_scale=None) -> set[str]:
    """Run the skip predicate with single-model defaults.

    Parameters
    ----------
    conn:
        Ledger connection.
    targets:
        Optional identity targets keyed by stable ID.
    input_scale:
        Optional current intended input scale for ``m1``.

    Returns
    -------
    set[str]
        Skip-eligible stable IDs.
    """

    targets = targets or {"m1": _target()}
    scales = {"m1": input_scale if input_scale is not None else 1.0}
    return incremental_skip_stable_ids(conn, "tl-current", targets, scales)


# ----- (a) matching identity + current real pass -> SKIPPED -----


def test_a_matching_identity_and_current_pass_is_skip_eligible(tmp_path: Path) -> None:
    """A genuine current-identity pass at the same input scale is skip-eligible."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run())

    assert _skip(conn) == {"m1"}


def test_a_empty_targets_skip_nothing(tmp_path: Path) -> None:
    """No targets means nothing is skip-eligible."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run())

    assert incremental_skip_stable_ids(conn, "tl-current", {}, {}) == set()


def test_a_machine_metadata_difference_does_not_prevent_skip(tmp_path: Path) -> None:
    """GATE (c): machine/hardware metadata is NOT part of identity.

    A model validated on machine A (one CPU/GPU/RAM fingerprint) is the SAME
    validation on machine B. The bucket-C machine fields must NEVER enter the
    identity tuple or the incremental-skip join: a row stamped with one machine
    fingerprint is STILL skip-eligible from a (machine-agnostic) target whose
    identity matches. The pass row carries fully-populated machine fields; the
    target -- which has no machine concept at all -- still yields a skip.
    """

    from menagerie.ledger import with_machine_metadata

    conn = connect(tmp_path / "verification.db")
    # Stamp the pass row with this host's REAL machine fingerprint, then with a
    # deliberately DIFFERENT one, to prove neither value is consulted by the join.
    append_verification_run(conn, with_machine_metadata(_run(run_id="real-machine")))

    assert _skip(conn) == {"m1"}

    other_db = tmp_path / "other.db"
    other = connect(other_db)
    append_verification_run(
        other,
        _run(
            run_id="other-machine",
            machine_cpu_model="Totally Different CPU",
            machine_cpu_cores_physical=128,
            machine_cpu_cores_logical=256,
            machine_total_ram_gb=2048.0,
            machine_gpu_models="A100 80GB;A100 80GB;A100 80GB;A100 80GB",
            machine_gpu_count=4,
            machine_platform="Linux-cluster-node-x86_64",
            machine_torch_num_threads=64,
        ),
    )

    # Same identity target -> still skip-eligible despite the divergent machine.
    assert _skip(other) == {"m1"}


# ----- (b) changed source/env/lock hash -> NOT skipped -----


def test_b_changed_source_hash_is_not_skipped(tmp_path: Path) -> None:
    """A pass under a different TorchLens source hash is stale -> validate."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(torchlens_source_hash="other-source"))

    assert _skip(conn) == set()


def test_b_changed_env_hash_is_not_skipped(tmp_path: Path) -> None:
    """A pass under a different env hash is stale -> validate."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(env_hash="other-env"))

    assert _skip(conn) == set()


def test_b_changed_lock_hash_is_not_skipped(tmp_path: Path) -> None:
    """A pass under a different lock hash is stale -> validate."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(lock_hash="other-lock"))

    assert _skip(conn) == set()


def test_b_changed_recipe_revision_is_not_skipped(tmp_path: Path) -> None:
    """A pass under a different recipe revision is stale -> validate."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(recipe_revision_sha256="recipe-b"))

    assert _skip(conn) == set()


def test_b_changed_device_is_not_skipped(tmp_path: Path) -> None:
    """A pass requested on a different device is stale -> validate."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(device_requested="cuda", device_actual="cuda"))

    assert _skip(conn) == set()


def test_b_changed_scope_is_not_skipped(tmp_path: Path) -> None:
    """A forward pass does not certify a forward+backward scope request."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run())

    assert _skip(conn, targets={"m1": _target(scope="forward+backward")}) == set()


def test_b_legacy_unknown_source_is_not_skipped(tmp_path: Path) -> None:
    """A legacy-unknown source pass never qualifies even if the tuple matches."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(torchlens_source_hash=LEGACY_UNKNOWN))

    assert _skip(conn, targets={"m1": _target(torchlens_source_hash=LEGACY_UNKNOWN)}) == set()


def test_b_legacy_unknown_lock_is_not_skipped(tmp_path: Path) -> None:
    """A legacy-unknown lock pass never qualifies even if the tuple matches."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(lock_hash=LEGACY_UNKNOWN))

    assert _skip(conn, targets={"m1": _target(lock_hash=LEGACY_UNKNOWN)}) == set()


# ----- (c) degraded (input_scale NULL) -> NOT skipped -----


def test_c_null_stored_input_scale_is_not_skipped(tmp_path: Path) -> None:
    """A degraded row (stored input_scale NULL) is never skip-eligible."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(input_scale=None))

    assert _skip(conn, input_scale=1.0) == set()


def test_c_null_target_input_scale_skips_nothing(tmp_path: Path) -> None:
    """When the current intended input scale is unknown (None), do not skip."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run())

    assert incremental_skip_stable_ids(conn, "tl-current", {"m1": _target()}, {"m1": None}) == set()


def test_c_mismatched_input_scale_is_not_skipped(tmp_path: Path) -> None:
    """A pass at a different input scale does not certify the current scale."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(input_scale=0.5))

    assert _skip(conn, input_scale=1.0) == set()


def test_c_matching_reduced_input_scale_is_skipped(tmp_path: Path) -> None:
    """A pass at the SAME reduced input scale is still skip-eligible."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(input_scale=0.5))

    assert _skip(conn, input_scale=0.5) == {"m1"}


# ----- (d) failed / timeout current row -> NOT skipped (retried) -----


def test_d_current_failed_is_not_skipped(tmp_path: Path) -> None:
    """A newer failure supersedes an older pass -> retry, never skip."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(run_id="old-pass"))
    append_verification_run(
        conn,
        _run(
            run_id="new-fail",
            status="failed",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            error_class="failed:replay",
            error_message="False",
            finished_at="2026-06-22T00:00:09+00:00",
        ),
    )

    assert _skip(conn) == set()


def test_d_current_timeout_is_not_skipped(tmp_path: Path) -> None:
    """A current timeout is retried, never skipped."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            status="timeout",
            forward_pass=None,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            error_class="failed:timeout",
            error_message="timed out",
        ),
    )

    assert _skip(conn) == set()


def test_d_current_install_failed_is_not_skipped(tmp_path: Path) -> None:
    """A current install_failed row is retried, never skipped."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            status="install_failed",
            forward_pass=None,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            error_class="install_failed",
            error_message="pin conflict",
        ),
    )

    assert _skip(conn) == set()


def test_d_cascade_suppressed_pass_is_skip_eligible(tmp_path: Path) -> None:
    """A cascade artifact NEWER than a real pass does not block the skip.

    The cascade-aware view selects the real pass, so a model whose latest raw row
    is a frozen cascade artifact is still skip-eligible on its real prior pass --
    and crucially the cascade row is NEVER itself promoted to skip-eligible.
    """

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(run_id="real-pass"))
    append_verification_run(
        conn,
        _cascade_artifact(run_id="cascade", finished_at="2026-06-26T08:37:54+00:00"),
    )

    assert _skip(conn) == {"m1"}


def test_d_cascade_only_model_is_not_skipped(tmp_path: Path) -> None:
    """A model whose ONLY row is a cascade artifact is never skipped."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _cascade_artifact(run_id="cascade-only"))

    assert _skip(conn) == set()


def test_d_wrong_torchlens_version_is_not_skipped(tmp_path: Path) -> None:
    """A pass under a different TorchLens VERSION does not certify the run."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(torchlens_version="tl-old"))

    assert _skip(conn) == set()


# ----- (e)/(f) flag wiring: force-full re-validates all; force re-validates selected -----


def test_e_force_full_flag_is_parsed_by_both_clis() -> None:
    """``--force-full`` is accepted by run-all AND the validator."""

    validate_args = validate_parser().parse_args(["--force-full"])
    run_all_args = run_all_parser().parse_args(["--force-full"])

    assert validate_args.force_full is True
    assert validate_args.force is False
    assert run_all_args.force_full is True


def test_f_force_flag_is_composable_with_selection() -> None:
    """``--force`` composes with --stable-ids and stays distinct from --force-full."""

    validate_args = validate_parser().parse_args(["--force", "--stable-ids", "m1", "m2"])
    run_all_args = run_all_parser().parse_args(["--force", "--family", "resnet"])

    assert validate_args.force is True
    assert validate_args.force_full is False
    assert validate_args.stable_ids == ["m1", "m2"]
    assert run_all_args.force is True
    assert run_all_args.family == "resnet"


def test_default_has_neither_force_flag_set() -> None:
    """Incremental skip is the DEFAULT: neither force flag is set without opt-in."""

    validate_args = validate_parser().parse_args([])
    run_all_args = run_all_parser().parse_args([])

    assert validate_args.force is False
    assert validate_args.force_full is False
    assert run_all_args.force is False
    assert run_all_args.force_full is False


def test_f_run_all_force_full_forwards_force_full_to_validator() -> None:
    """run-all --force-full forwards --force-full (not --force) to the validator."""

    from menagerie.run_all import _validation_args

    args = run_all_parser().parse_args(["--force-full"])
    forwarded = _validation_args(args, Path("/tmp/validation"))

    assert "--force-full" in forwarded
    assert "--force" not in forwarded


def test_f_run_all_force_forwards_force_to_validator() -> None:
    """run-all --force forwards the targeted --force to the validator."""

    from menagerie.run_all import _validation_args

    args = run_all_parser().parse_args(["--force", "--stable-ids", "m1"])
    forwarded = _validation_args(args, Path("/tmp/validation"))

    assert "--force" in forwarded
    assert "--force-full" not in forwarded
    assert "--stable-ids" in forwarded


# ----- integration: the validator-side skip seam (env + device + ledger) -----


def _catalog_row(**overrides: object) -> CatalogRow:
    """Build a small base-env CPU catalog row fixture.

    Parameters
    ----------
    overrides:
        Catalog field overrides.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    data: dict[str, object] = {
        "model_id": 1,
        "display_index": 1,
        "stable_id": "m1",
        "name": "UnitNet",
        "variant": "",
        "family": "unit",
        "family_normalized": "unit",
        "domain": "unit",
        "zoo": "unit-zoo",
        "constructor_call": "torch.nn.Identity()",
        "input_shape": "(1,)",
        "input_dtype": "float32",
        "era": "2026",
        "verified": True,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "recipe-a",
    }
    data.update(overrides)
    return CatalogRow(**data)  # type: ignore[arg-type]


def test_integration_skip_seam_skips_current_pass(tmp_path: Path) -> None:
    """The validator skip seam skips a base-env CPU row with a current pass.

    Exercises env-hash resolution (base), device routing (local-cpu -> cpu) and
    the ledger predicate together -- the seam ``run()`` uses to decide the skip.
    """

    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(conn, _run(torchlens_version=_torchlens_version()))
    conn.close()

    skip = _incremental_skip_stable_ids(
        [_catalog_row()],
        scope="forward",
        device="cpu",
        input_scale_by_id={"m1": 1.0},
        env_registry=None,
        ledger_db=ledger_db,
        local_gpu_vram_bytes=None,
    )

    assert skip == {"m1"}


def test_integration_skip_seam_validates_degraded_row(tmp_path: Path) -> None:
    """The validator skip seam refuses a degraded (input_scale NULL) row."""

    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(conn, _run(input_scale=None, torchlens_version=_torchlens_version()))
    conn.close()

    skip = _incremental_skip_stable_ids(
        [_catalog_row()],
        scope="forward",
        device="cpu",
        input_scale_by_id={"m1": 1.0},
        env_registry=None,
        ledger_db=ledger_db,
        local_gpu_vram_bytes=None,
    )

    assert skip == set()


def test_integration_skip_manifest_row_uses_real_pass_not_cascade(tmp_path: Path) -> None:
    """A skipped model's manifest row is reconstructed from the REAL pass.

    Even when the absolute-latest raw ledger row is a frozen cascade artifact,
    the manifest row that records the skip must come from the cascade-suppressed
    ``current_verification_real`` pass (validated, real n_ops), never the cascade
    (failed, NULL n_ops).
    """

    from menagerie.cluster_runner import current_real_verification_run_for_stable_id

    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(conn, _run(run_id="real-pass", torchlens_version=_torchlens_version()))
    append_verification_run(
        conn,
        _cascade_artifact(run_id="cascade", finished_at="2026-06-26T08:37:54+00:00"),
    )
    conn.close()

    run = current_real_verification_run_for_stable_id("m1", ledger_db=ledger_db)

    assert run.status == "passed"
    assert run.n_ops == 4
    assert run.error_class != CASCADE_ARTIFACT_ERROR_CLASS


def test_integration_skip_seam_validates_changed_source(tmp_path: Path) -> None:
    """The validator skip seam refuses a row whose source hash changed.

    The seam computes the CURRENT source hash, so a pass recorded under a
    different (stale) source hash is not skip-eligible.
    """

    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(
        conn, _run(torchlens_source_hash="stale-source", torchlens_version=_torchlens_version())
    )
    conn.close()

    skip = _incremental_skip_stable_ids(
        [_catalog_row()],
        scope="forward",
        device="cpu",
        input_scale_by_id={"m1": 1.0},
        env_registry=None,
        ledger_db=ledger_db,
        local_gpu_vram_bytes=None,
    )

    assert skip == set()
