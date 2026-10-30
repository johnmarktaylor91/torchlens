"""The agent benchmark harness (``tools/agent_evals/bench/``): registry, guards, runner, telemetry.

No model is called. The end-to-end tests drive the runner with the scripted fake agent on the
harness stub task, so the environment, guard, runner, grader and record path all execute.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from agent_evals.bench import guards, registry, runner, scheduler, telemetry  # noqa: E402

BENCH = REPO_ROOT / "tools" / "agent_evals" / "bench"
STUB_ROOT = BENCH / "stub"
FAKE_AGENT = BENCH / "fake_agent.py"


@pytest.fixture
def stub_task() -> registry.BenchTask:
    return registry.load_registry(STUB_ROOT)["T0"]


# --- registry -------------------------------------------------------------------------------


def test_registry_loads_the_stub_with_private_material_kept_out(
    stub_task: registry.BenchTask,
) -> None:
    assert stub_task.task_id == "T0"
    assert stub_task.inputs is not None and stub_task.inputs.name == "inputs"
    assert stub_task.heldout is not None and stub_task.heldout.name == "heldout"
    assert stub_task.solve_timeout_s == 300
    assert stub_task.applies_to("A") and stub_task.applies_to("D")
    prompt = stub_task.prompt()
    assert "Record the output of the submodule" in prompt
    assert "Oracle and pass bars" not in prompt and "grader" not in prompt


def test_prompt_section_falls_back_to_the_whole_spec() -> None:
    assert registry.prompt_section("# T\n\nDo it.\n") == "# T\n\nDo it.\n"
    spec = "# T\n\n## Prompt to the agent\n\nDo it.\n\n## Oracle\n\nSecret.\n"
    assert registry.prompt_section(spec) == "# T\n\nDo it.\n"


def test_registry_refuses_a_missing_heldout_or_grader(tmp_path: Path) -> None:
    task = tmp_path / "t09"
    task.mkdir()
    (task / "spec.md").write_text("spec")
    with pytest.raises(FileNotFoundError, match="grader.py"):
        registry.load_task(task)
    (task / "grader.py").write_text("")
    (task / "task.json").write_text(json.dumps({"heldout": True}))
    with pytest.raises(FileNotFoundError, match="heldout"):
        registry.load_task(task)


def test_registry_skips_shared_folders_and_refuses_duplicate_ids(tmp_path: Path) -> None:
    for name in ("_material", "T1", "T1-copy"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "spec.md").write_text("spec")
        (tmp_path / name / "grader.py").write_text("")
    (tmp_path / "T1-copy" / "task.json").write_text(json.dumps({"id": "T1"}))
    with pytest.raises(ValueError, match="duplicate task id T1"):
        registry.load_registry(tmp_path)
    (tmp_path / "T1-copy" / "task.json").unlink()
    assert sorted(registry.load_registry(tmp_path)) == ["T1", "T1-copy"]


@pytest.mark.parametrize(
    ("stdout", "passed", "error"),
    [
        ('log line\n{"passed": true, "bars": [{"name": "a", "passed": true}]}\n', True, None),
        ("", False, "grader printed nothing"),
        ("Traceback ...\n", False, "grader's last line is not JSON"),
        ("[1, 2]", False, "grader's verdict is not an object"),
    ],
)
def test_verdict_is_the_graders_last_json_line(
    stdout: str, passed: bool, error: str | None
) -> None:
    verdict = registry.parse_verdict(stdout)
    assert verdict["passed"] is passed and verdict.get("error") == error


def test_task_numbers_map_oracle_directories_to_plan_names() -> None:
    assert guards.task_number("t01-jacobian-lens") == "T1"
    assert guards.task_number("t12-control-module-output") == "T12"
    assert guards.task_number("T0") == "T0"
    assert guards.entries_for("t04-attribution-patching", "D") == ["transformer_lens.patching:*"]


@pytest.mark.parametrize(
    ("raw", "passed"),
    [
        ({"passed": True, "bars": [{"name": "a", "passed": True}]}, True),
        ({"passed": True, "bars": [{"name": "a", "passed": False}]}, False),
        ({"passed": "yes", "bars": []}, False),
        ({"bars": [{"name": "a", "passed": True}]}, False),
        ({"passed": True, "checks": [{"name": "a", "passed": 1}]}, False),
    ],
)
def test_verdicts_never_pass_over_a_failing_or_malformed_bar(raw: dict, passed: bool) -> None:
    assert registry.normalize_verdict(raw)["passed"] is passed


# --- guards ---------------------------------------------------------------------------------


@pytest.fixture
def guarded_site(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A site directory with the guard installed and a fake library holding two natives."""

    site = tmp_path / "site"
    pkg = site / "fakelib"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .api import jacobian_lens, keep\n")
    (pkg / "api.py").write_text(
        "def jacobian_lens():\n    return 'native'\n\n"
        "def keep():\n    return 'kept'\n\n"
        "class Layer:\n    def readout(self):\n        return 'native'\n"
    )
    (pkg / "patching.py").write_text("X = 1\n")
    union = ["fakelib.api:jacobian_lens", "fakelib.patching"]
    guards.install(site, union)
    boot = tmp_path / "boot"
    boot.mkdir()
    (boot / "sitecustomize.py").write_text(f"import site\nsite.addsitedir({str(site)!r})\n")
    return tmp_path, {"PYTHONPATH": str(boot), "PATH": "/usr/bin:/bin"}


def _probe(env: dict[str, str], code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_guard_blocks_a_module_and_stubs_an_attribute(guarded_site: tuple[Path, dict]) -> None:
    root, env = guarded_site
    run_file = guards.write_run_config(
        root / "run.json", ["fakelib.api:jacobian_lens", "fakelib.patching"]
    )
    env = {**env, guards.ENV_VAR: str(run_file)}
    results = guards.selftest(
        sys.executable,
        ["fakelib.patching", "fakelib.api:jacobian_lens", "fakelib:jacobian_lens"],
        env,
    )
    assert [r["outcome"] for r in results] == ["blocked", "blocked", "blocked"]
    kept = _probe(env, "import fakelib; print(fakelib.keep())")
    assert kept.returncode == 0 and kept.stdout.strip() == "kept"


def test_guard_wildcard_disables_every_public_callable_of_an_eager_module(
    guarded_site: tuple[Path, dict],
) -> None:
    root, env = guarded_site
    pkg = root / "site" / "fakelib"
    (pkg / "__init__.py").write_text("from . import tools\nfrom .api import jacobian_lens, keep\n")
    (pkg / "tools.py").write_text(
        "def patch_all():\n    return 1\n\ndef patch_one():\n    return 2\n"
    )
    run_file = guards.write_run_config(root / "run.json", ["fakelib.tools:*"])
    env = {**env, guards.ENV_VAR: str(run_file)}
    assert guards.selftest(sys.executable, ["fakelib.tools:*"], env)[0]["outcome"] == "blocked"
    proc = _probe(env, "import fakelib\nprint(fakelib.keep())\nfakelib.tools.patch_one()")
    assert proc.stdout.strip() == "kept" and guards.MARKER in proc.stderr


def test_guard_stubs_a_method_on_a_class(guarded_site: tuple[Path, dict]) -> None:
    root, env = guarded_site
    run_file = guards.write_run_config(root / "run.json", ["fakelib.api:Layer.readout"])
    env = {**env, guards.ENV_VAR: str(run_file)}
    proc = _probe(env, "from fakelib.api import Layer\nLayer().readout()")
    assert proc.returncode != 0 and guards.MARKER in proc.stderr


@pytest.mark.parametrize("variable", [None, "/nonexistent/guard.json"])
def test_guard_fails_closed_to_the_union(
    guarded_site: tuple[Path, dict], variable: str | None
) -> None:
    _, env = guarded_site
    if variable:
        env = {**env, guards.ENV_VAR: variable}
    proc = _probe(env, "import fakelib.patching")
    assert proc.returncode != 0 and guards.MARKER in proc.stderr
    proc = _probe(env, "import fakelib; fakelib.jacobian_lens()")
    assert proc.returncode != 0 and guards.MARKER in proc.stderr


def test_guard_with_an_empty_run_file_blocks_nothing(guarded_site: tuple[Path, dict]) -> None:
    root, env = guarded_site
    # An empty list is a valid run file (a task with no natives): nothing is blocked.
    env = {**env, guards.ENV_VAR: str(guards.write_run_config(root / "none.json", []))}
    assert guards.selftest(sys.executable, ["fakelib.patching"], env)[0]["outcome"] == "not blocked"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import fakelib.patching", ["import fakelib.patching"]),
        ("from fakelib import patching", ["from fakelib import patching"]),
        ("import fakelib as f\nf.api.jacobian_lens()", ["f.api.jacobian_lens"]),
        ("from fakelib.api import jacobian_lens", ["from fakelib.api import jacobian_lens"]),
        (
            "import fakelib\ngetattr(fakelib.api, 'jacobian_lens')()",
            ["getattr(..., 'jacobian_lens')"],
        ),
        ("import fakelib\nfakelib.keep()\nname = 'jacobian_lens.json'", []),
        ("not python (", []),
        ("from fakelib import tools\ntools.patch_all()", ["tools.patch_all"]),
        ("from fakelib.tools import patch_all", ["from fakelib.tools import patch_all"]),
        ("import fakelib.tools", []),
    ],
)
def test_audit_finds_guarded_names_in_agent_code(source: str, expected: list[str]) -> None:
    entries = ["fakelib.api:jacobian_lens", "fakelib.patching", "fakelib.tools:*"]
    assert guards.audit_source(source, entries) == expected


def test_natives_file_names_only_known_arms_and_tasks() -> None:
    natives = guards.load_natives()
    assert sorted(natives, key=lambda t: int(t[1:])) == [f"T{i}" for i in range(1, 13)]
    for arms_map in natives.values():
        assert set(arms_map) <= {"*", *registry.ALL_ARMS}
        assert "A" not in arms_map or not arms_map["A"], "arm A has no library natives"


# --- telemetry ------------------------------------------------------------------------------


def _assistant(mid: str, tools: list[tuple[str, dict]], usage: dict) -> str:
    content = [{"type": "tool_use", "name": n, "input": i} for n, i in tools]
    return json.dumps(
        {"type": "assistant", "message": {"id": mid, "content": content, "usage": usage}}
    )


def test_stream_stats_count_tools_executions_and_tokens_once_per_message() -> None:
    stats = telemetry.StreamStats(non_substantive=frozenset({"TodoWrite"}))
    usage = {"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 5}
    for line in (
        _assistant("m1", [("Bash", {"command": "python run.py"})], usage),
        _assistant("m1", [("Read", {"file_path": "TASK.md"})], usage),
        _assistant("m2", [("TodoWrite", {"todos": []}), ("Bash", {"command": "ls"})], usage),
        "not json",
    ):
        stats.feed(line)
    assert stats.tool_counts == {"Bash": 2, "Read": 1, "TodoWrite": 1}
    assert stats.substantive_calls == 3
    assert stats.executions == 1
    assert stats.tokens == {
        "input_uncached": 20,
        "input_cache_read": 200,
        "input_cache_write": 0,
        "output": 10,
    }
    assert stats.bad_lines == 1
    stats.feed(json.dumps({"type": "result", "usage": {"input_tokens": 1, "output_tokens": 2}}))
    assert stats.total_tokens()["output"] == 2


def test_load_bearing_lines_skip_blanks_comments_and_docstrings() -> None:
    source = '"""Module doc.\n\nMore."""\n\n# comment\nimport torch\n\n\ndef f():\n    """Doc."""\n    return 1\n'
    assert telemetry.load_bearing_lines(source) == 3


@pytest.mark.parametrize(
    ("snippets", "expected"),
    [
        (["import torchlens as tl\nlog = tl.trace(model, x)\nlog.ops()"], "torchlens.trace"),
        (
            ["x = 1", "from torchlens.intervention import steer\nsteer(m)"],
            "torchlens.intervention.steer",
        ),
        (["cat > s.py <<'EOF'\nimport torchlens\ntorchlens.summary(m)\nEOF"], "torchlens.summary"),
        (["import torch\ntorch.zeros(1)"], None),
    ],
)
def test_first_library_call(snippets: list[str], expected: str | None) -> None:
    assert telemetry.first_library_call(snippets, "torchlens") == expected


# --- scheduler ------------------------------------------------------------------------------


def test_pilot_plan_is_42_runs_in_balanced_seeded_blocks() -> None:
    rows = scheduler.plan(scheduler.pilot_cells(), repeats=3, seed=7)
    assert len(rows) == 42
    blocks = {b: [r for r in rows if r.block == b] for b in range(3)}
    cells = {(r.task, r.arm, r.tier) for r in blocks[0]}
    assert all({(r.task, r.arm, r.tier) for r in block} == cells for block in blocks.values())
    assert [r.run_key for r in rows] == [
        r.run_key for r in scheduler.plan(scheduler.pilot_cells(), 3, 7)
    ]
    assert [r.run_key for r in rows] != [
        r.run_key for r in scheduler.plan(scheduler.pilot_cells(), 3, 8)
    ]
    assert len({r.run_key for r in rows}) == 42


def test_plan_keeps_not_applicable_rows(tmp_path: Path) -> None:
    rows = scheduler.plan([("T3", "D", "mid"), ("T9", "D", "mid")], 1, 0, {("T9", "D"): False})
    assert {r.task: r.applicable for r in rows} == {"T3": True, "T9": False}
    path = scheduler.write(rows, tmp_path / "plan.jsonl", 0)
    assert json.loads(path.read_text().splitlines()[0])["seed"] == 0
    assert len(scheduler.pending(rows, tmp_path)) == 2


# --- runner, end to end with the fake agent -------------------------------------------------


def _config(mode: str, *extra: str, caps: runner.Caps | None = None) -> runner.RunConfig:
    reference = str(STUB_ROOT / "T0" / "reference" / "solve.py")
    argv = (sys.executable, str(FAKE_AGENT), "--mode", mode, "--solution", reference, *extra)
    return runner.RunConfig(
        agent_cmd={"small": argv},
        caps={"small": caps or runner.Caps(wall_s=120, checkpoints_s=(60.0,))},
        non_substantive_tools=frozenset({"TodoWrite"}),
    )


def _spec(task: registry.BenchTask, library: str | None = None) -> runner.RunSpec:
    python = Path(sys.executable)
    return runner.RunSpec(task, "A", "small", 0, python, python, library=library)


@pytest.mark.heavy  # three torch subprocesses (agent run, grader, held-out run), about 6 s
def test_fake_agent_solving_the_stub_passes_with_a_complete_record(
    stub_task: registry.BenchTask, tmp_path: Path
) -> None:
    record = runner.run_one(_spec(stub_task), tmp_path / "run", _config("solve"))
    assert telemetry.validate_record(record) == []
    assert record["end_reason"] == "finished" and record["passed"] is True, record
    assert record["tool_calls"] == {
        "total": 3,
        "substantive": 3,
        "by_name": {"Read": 1, "Write": 1, "Bash": 1},
    }
    assert record["executions"] == 1
    assert record["tokens"]["input_cache_write"] == 200
    assert record["load_bearing_lines"] > 5
    verdict = record["checkpoints"]["t1"]
    assert [b["passed"] for b in verdict["bars"]] == [True, True]
    assert verdict["solve"]["visible"]["returncode"] == 0
    assert verdict["solve"]["heldout"]["returncode"] == 0
    work = tmp_path / "run" / "work"
    assert sorted(p.name for p in work.iterdir() if p.name[0] not in "._") == [
        "TASK.md",
        "inputs",
        "out",
        "solve.py",
    ]
    assert "Oracle and pass bars" not in (work / "TASK.md").read_text()
    assert not (work / "inputs" / "grader.py").exists()
    assert json.loads((tmp_path / "run" / "record.json").read_text())["run_id"] == record["run_id"]


@pytest.mark.heavy  # one torch grader subprocess
def test_a_run_that_does_nothing_is_counted_as_a_failure(
    stub_task: registry.BenchTask, tmp_path: Path
) -> None:
    record = runner.run_one(_spec(stub_task), tmp_path / "run", _config("nothing"))
    assert telemetry.validate_record(record) == []
    assert record["end_reason"] == "finished" and record["passed"] is False


@pytest.mark.heavy  # one torch grader subprocess
def test_tool_call_cap_stops_the_agent(stub_task: registry.BenchTask, tmp_path: Path) -> None:
    caps = runner.Caps(wall_s=120, tool_calls=3, checkpoints_s=(60.0,))
    config = _config("loop", "--calls", "50", "--sleep", "0.2", caps=caps)
    record = runner.run_one(_spec(stub_task), tmp_path / "run", config)
    assert record["end_reason"] == "tool_cap"
    assert 3 < record["tool_calls"]["substantive"] < 50


@pytest.mark.heavy  # two torch grader subprocesses
def test_wall_cap_stops_the_agent_and_early_checkpoints_are_snapshots(
    stub_task: registry.BenchTask, tmp_path: Path
) -> None:
    caps = runner.Caps(wall_s=3, checkpoints_s=(1.0, 3.0))
    config = _config("loop", "--calls", "100", "--sleep", "0.2", caps=caps)
    record = runner.run_one(_spec(stub_task), tmp_path / "run", config)
    assert record["end_reason"] == "wall_cap"
    assert 2.5 < record["elapsed_s"] < 30
    assert set(record["checkpoints"]) == {"t0.02", "t0.05"}


@pytest.mark.heavy  # one torch grader subprocess
def test_agent_code_using_a_guarded_name_fails_the_run(
    stub_task: registry.BenchTask, tmp_path: Path
) -> None:
    natives = dict(stub_task.natives) | {"A": ("torch:no_grad",)}
    task = registry.BenchTask(**{**stub_task.__dict__, "natives": natives})
    record = runner.run_one(_spec(task), tmp_path / "run", _config("solve"))
    assert record["guard"]["violations"] == ["torch.no_grad"]
    assert record["passed"] is False


def test_not_applicable_cells_are_recorded(stub_task: registry.BenchTask, tmp_path: Path) -> None:
    task = registry.BenchTask(**{**stub_task.__dict__, "arms": ("B",)})
    record = runner.run_one(_spec(task), tmp_path / "run", _config("solve"))
    assert record["end_reason"] == "not_applicable" and record["passed"] is False
    assert telemetry.validate_record(record) == []


def test_run_config_loads_tiers_and_refuses_unknown_ones(tmp_path: Path) -> None:
    path = tmp_path / "tiers.json"
    path.write_text(
        json.dumps(
            {"tiers": {"small": {"agent_cmd": ["a", "{prompt_file}"], "caps": {"token_cap": 9}}}}
        )
    )
    config = runner.RunConfig.load(path)
    assert config.caps["small"].token_cap == 9 and config.caps["small"].tool_calls == 60
    path.write_text(json.dumps({"tiers": {"huge": {"agent_cmd": []}}}))
    with pytest.raises(ValueError, match="unknown tiers"):
        runner.RunConfig.load(path)


@pytest.mark.heavy  # three candidates, each a solve run, a held-out run and a grader run
def test_check_task_wires_the_stub_grader(tmp_path: Path) -> None:
    from agent_evals.bench.__main__ import main

    out = tmp_path / "check"
    argv = ["check-task", "--oracles", str(STUB_ROOT), "--task", "T0", "--python", sys.executable]
    assert main([*argv, "--out", str(out)]) == 0
    results = json.loads((out / "check.json").read_text())
    assert {k: v["passed"] for k, v in results.items()} == {
        "reference": True,
        "wrong/wrong_module": False,
        "do_nothing": False,
    }


# --- external runner: stage, then ingest what it returns -------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"exit_reason": "completed", "returncode": 0}, "finished"),
        ({"exit_reason": "completed", "returncode": 1}, "agent_error"),
        ({"exit_reason": "wall_cap", "returncode": None}, "wall_cap"),
        ({"exit_reason": "output_token_cap"}, "token_cap"),
        ({"exit_reason": "credential_in_output"}, "harness_error"),
        ({}, "harness_error"),
    ],
)
def test_external_exit_reasons_map_to_record_end_reasons(raw: dict, expected: str) -> None:
    from agent_evals.bench import external

    assert external.end_reason(raw) == expected


def test_stage_writes_only_the_prompt_inputs_and_material(
    stub_task: registry.BenchTask, tmp_path: Path
) -> None:
    from agent_evals.bench import external

    material = tmp_path / "INDEX.md"
    material.write_text("notes")
    spec = runner.RunSpec(
        stub_task, "A", "small", 0, Path(sys.executable), Path(sys.executable), material
    )
    prompt = external.stage(spec, tmp_path / "task", "AGENTS.md")
    assert sorted(p.name for p in (tmp_path / "task").iterdir()) == [
        "AGENTS.md",
        "TASK.md",
        "inputs",
    ]
    assert "solve.py" in prompt and "AGENTS.md" in prompt
    assert not any(
        p.name in {"grader.py", "reference", "heldout"} for p in (tmp_path / "task").rglob("*")
    )


@pytest.mark.heavy  # the fake agent's solve run plus one graded checkpoint (solve, held-out, grader)
def test_ingest_grades_a_returned_run_into_a_valid_record(
    stub_task: registry.BenchTask, tmp_path: Path
) -> None:
    from agent_evals.bench import external

    spec = _spec(stub_task)
    returned = tmp_path / "returned"
    external.stage(spec, returned / "task")
    reference = STUB_ROOT / "T0" / "reference" / "solve.py"
    with (returned / "agent.jsonl").open("w") as stream:
        subprocess.run(
            [sys.executable, str(FAKE_AGENT), "--mode", "solve", "--solution", str(reference)],
            cwd=returned / "task",
            stdout=stream,
            check=True,
            env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
        )
    (returned / "record.json").write_text(
        json.dumps({"exit_reason": "completed", "returncode": 0, "elapsed_s": 12.5})
    )
    record = external.ingest(spec, returned, tmp_path / "graded", _config("solve"), (45,))
    assert telemetry.validate_record(record) == []
    assert record["end_reason"] == "finished" and record["passed"] is True
    assert record["elapsed_s"] == 12.5 and record["tool_calls"]["total"] == 3
    assert record["tokens"]["output"] == 150 and record["load_bearing_lines"] > 5
