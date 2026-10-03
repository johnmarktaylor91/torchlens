"""F29: the determinism contract, enforced as a test (agent memo 3.9).

Same artifact bytes + torchlens version + normalized arguments ->
byte-identical output, across fresh processes, PYTHONHASHSEED values, working
directories, TZ, and locale. Golden files live in tests/agent_surface_goldens
next to the COMMITTED artifact they were generated from (the artifact bytes
are the identity, so goldens never chase re-capture timestamps).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _oracle_env import expect_bundle_minor_version_mismatch

from torchlens.agent import call_tool, canonical_dumps

GOLDEN_DIR = Path(__file__).resolve().parent / "agent_surface_goldens"
ARTIFACT = GOLDEN_DIR / "clean.tlspec"

#: The sweep requests, one per surface class.
SWEEP_REQUESTS: list[tuple[str, dict]] = [
    ("torchlens_overview", {"path": str(ARTIFACT)}),
    ("torchlens_overview", {"path": str(ARTIFACT), "mode": "manifest"}),
    ("torchlens_dump", {"path": str(ARTIFACT), "view": "graph"}),
    ("torchlens_query_sites", {"path": str(ARTIFACT), "query": {"op": "func", "value": "relu"}}),
    ("torchlens_payload_stats", {"path": str(ARTIFACT), "labels": ["relu_1_2:1"]}),
    ("torchlens_schema", {}),
]


def _emit_probe(requests: list[tuple[str, dict]]) -> str:
    """Python source that emits every sweep request as canonical JSON lines."""

    repo_root = str(Path(__file__).resolve().parent.parent)
    return (
        "import json, sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "from torchlens.agent import call_tool, canonical_dumps\n"
        f"requests = json.loads({json.dumps(json.dumps(requests))})\n"
        "for name, args in requests:\n"
        "    sys.stdout.write(canonical_dumps(call_tool(name, args)) + '\\n')\n"
    )


@pytest.mark.heavy
def test_environment_sweep_is_byte_identical(tmp_path: Path) -> None:
    """Two hash seeds x two cwds x TZ x locale -> identical bytes."""

    environments = [
        {"PYTHONHASHSEED": "0", "TZ": "UTC", "LANG": "C", "LC_ALL": "C"},
        {"PYTHONHASHSEED": "1", "TZ": "America/New_York", "LANG": "en_US.UTF-8"},
    ]
    cwds = [str(tmp_path), str(Path(__file__).resolve().parent.parent)]
    outputs: set[str] = set()
    probe = _emit_probe(SWEEP_REQUESTS)
    for env_delta in environments:
        for cwd in cwds:
            env = {**os.environ, **env_delta}
            if "LC_ALL" not in env_delta:
                env.pop("LC_ALL", None)
            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
            )
            assert result.returncode == 0, result.stderr[-800:]
            outputs.add(result.stdout)
    assert len(outputs) == 1, "agent-surface output varied with the environment"


@pytest.mark.smoke
def test_repeated_calls_are_byte_identical() -> None:
    """In-process repetition is byte-stable (no clocks, no id() leakage)."""

    with expect_bundle_minor_version_mismatch():
        for name, args in SWEEP_REQUESTS:
            assert canonical_dumps(call_tool(name, args)) == canonical_dumps(call_tool(name, args))


@pytest.mark.smoke_cells(
    "test_golden_files_pin_the_wire_format[payload_stats-torchlens_payload_stats-args2]",
    "test_golden_files_pin_the_wire_format[query_sites-torchlens_query_sites-args1]",
)
@pytest.mark.parametrize(
    ("golden_name", "tool", "args"),
    [
        ("overview_folded", "torchlens_overview", {"path": str(ARTIFACT)}),
        (
            "query_sites",
            "torchlens_query_sites",
            {"path": str(ARTIFACT), "query": {"op": "func", "value": "relu"}},
        ),
        (
            "payload_stats",
            "torchlens_payload_stats",
            {"path": str(ARTIFACT), "labels": ["relu_1_2:1"]},
        ),
    ],
)
def test_golden_files_pin_the_wire_format(golden_name: str, tool: str, args: dict) -> None:
    """Live output over the committed artifact equals the committed golden.

    ``torchlens_version`` is normalized (it names the producer and changes
    per release); every other byte is pinned. A diff here is a WIRE-FORMAT
    change: bump the schema major or re-baseline consciously, never silently.
    """

    with expect_bundle_minor_version_mismatch():
        envelope = call_tool(tool, args)
    envelope["torchlens_version"] = "GOLDEN"
    envelope["request"]["path"] = "<ARTIFACT>"  # the echo carries the caller's path
    golden = (GOLDEN_DIR / f"{golden_name}.golden.json").read_text()
    assert canonical_dumps(envelope) + "\n" == golden


def test_no_generation_timestamp_anywhere() -> None:
    """Artifact created_at is data; report-generation time is forbidden."""

    import datetime

    with expect_bundle_minor_version_mismatch():
        envelope = call_tool("torchlens_overview", {"path": str(ARTIFACT)})
    text = canonical_dumps(envelope)
    today = datetime.date.today().isoformat()
    saved_day = (envelope.get("artifact") or {}).get("created_at", "")[:10]
    if saved_day != today:  # the committed artifact ages out of "today"
        assert today not in text
