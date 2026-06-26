"""Tests for the package-level menagerie orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from menagerie import __main__ as menagerie_main
from menagerie import generate_menagerie, run_all, trace_summary, validate_menagerie


def test_validate_dispatch_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m menagerie validate`` forwards args to validate_menagerie.main."""

    captured: dict[str, Any] = {}

    def fake_validate_main(argv: list[str]) -> int:
        """Capture forwarded validator arguments.

        Parameters
        ----------
        argv:
            Forwarded CLI arguments.

        Returns
        -------
        int
            Fixture exit status.
        """

        captured["argv"] = argv
        return 17

    monkeypatch.setattr(validate_menagerie, "main", fake_validate_main)

    assert menagerie_main.main(["validate", "--subset", "3", "--device", "cpu"]) == 17
    assert captured["argv"] == ["--subset", "3", "--device", "cpu"]


def test_run_all_orders_steps_and_skips_missing_csv_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run-all calls staged entry points in order and records timing."""

    calls: list[tuple[str, list[str]]] = []
    out_dir = tmp_path / "run"

    def option_value(argv: list[str], flag: str) -> str:
        """Return the value after a CLI option.

        Parameters
        ----------
        argv:
            CLI argument list.
        flag:
            Option flag.

        Returns
        -------
        str
            Option value.
        """

        return argv[argv.index(flag) + 1]

    def fake_validate_main(argv: list[str]) -> int:
        """Write minimal validation outputs.

        Parameters
        ----------
        argv:
            Validator arguments.

        Returns
        -------
        int
            Success exit code.
        """

        calls.append(("validate", list(argv)))
        validation_dir = Path(option_value(argv, "--out-dir"))
        manifest_path = Path(option_value(argv, "--manifest"))
        validation_dir.mkdir(parents=True)
        manifest_path.write_text(
            "name\tmodel_id\tstable_id\trecipe_revision_sha256\tstatus\tn_ops\t"
            "validate_metadata_ok\tscope\telapsed\tdependency_cluster\terror\t"
            "graph_shape_hash\tpeak_rss_mb\tinput_scale\n"
            "fixture\t7\tm_fixture\trecipe\tvalidated\t12\tTrue\tforward\t0.1\tbase\t\t"
            "hash\t10\t1.0\n",
            encoding="utf-8",
        )
        (validation_dir / "validation_summary.json").write_text(
            json.dumps({"totals": {"validated": 1, "failed": 0, "skipped": 0, "total": 1}}) + "\n",
            encoding="utf-8",
        )
        return 0

    def fake_render_main(argv: list[str]) -> int:
        """Write minimal render outputs.

        Parameters
        ----------
        argv:
            Renderer arguments.

        Returns
        -------
        int
            Success exit code.
        """

        calls.append(("render", list(argv)))
        assert argv[argv.index("--model-id") + 1] == "7"
        visuals_dir = Path(option_value(argv, "--out-dir"))
        manifest_path = Path(option_value(argv, "--manifest"))
        visuals_dir.mkdir(parents=True)
        manifest_path.write_text(
            "name\tmodel_id\tstable_id\trecipe_revision_sha256\tstatus\tn_nodes\t"
            "render_path\telapsed\tdependency_cluster\terror\tgraph_shape_hash\n"
            "fixture\t7\tm_fixture\trecipe\trendered\t12\tfixture.svg\t0.1\tbase\t\thash\n",
            encoding="utf-8",
        )
        return 0

    def fake_trace_summary_main(argv: list[str]) -> int:
        """Write a minimal trace-summary database.

        Parameters
        ----------
        argv:
            Trace-summary arguments.

        Returns
        -------
        int
            Success exit code.
        """

        calls.append(("trace_summary", list(argv)))
        assert argv[:2] == ["--stable-ids", "m_fixture"]
        db_path = Path(option_value(argv, "--db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE trace_summaries(stable_id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO trace_summaries VALUES ('m_fixture')")
        return 0

    real_import_module = run_all.importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        """Pretend optional csv_export is not installed.

        Parameters
        ----------
        name:
            Module name.
        package:
            Optional package context.

        Returns
        -------
        Any
            Imported module.
        """

        if name == "menagerie.csv_export":
            raise ModuleNotFoundError(
                "No module named 'menagerie.csv_export'", name="menagerie.csv_export"
            )
        return real_import_module(name, package)

    monkeypatch.setattr(validate_menagerie, "main", fake_validate_main)
    monkeypatch.setattr(generate_menagerie, "main", fake_render_main)
    monkeypatch.setattr(trace_summary, "main", fake_trace_summary_main)
    monkeypatch.setattr(run_all.importlib, "import_module", fake_import_module)

    assert run_all.main(["--out-dir", str(out_dir), "--jobs", "2"]) == 0

    assert [name for name, _argv in calls] == ["validate", "render", "trace_summary"]
    validate_argv = calls[0][1]
    assert validate_argv[validate_argv.index("--runner") + 1] == "auto"
    assert (out_dir / "validation" / "validation_manifest.tsv").exists()
    assert (out_dir / "visuals" / "manifest.tsv").exists()
    assert (out_dir / "metadata" / "trace_summary.db").exists()

    report = json.loads((out_dir / run_all.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["validation"] == {"validated": 1, "failed": 0, "skipped": 0, "total": 1}
    assert report["render_count"] == 1
    assert report["metadata_row_count"] == 1
    assert report["csv_export"] == "skipped: menagerie.csv_export is not available"
    assert [timing["name"] for timing in report["timings"]] == [
        "validation",
        "render",
        "metadata",
    ]
    assert all(timing["elapsed_sec"] >= 0 for timing in report["timings"])
