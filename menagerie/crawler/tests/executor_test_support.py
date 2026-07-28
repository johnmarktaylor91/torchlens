"""Shared support for author-executor tests: fake harness, fixtures, envelopes.

The fake ``claude`` binary honors the executor's real invocation contract: it
parses ``-p``/``--session-id``/``--resume``, extracts its output paths from the
rendered brief exactly as a real session would, records every invocation, and
prints a harness-shaped JSON result. Behavior is selected by environment
variables so kill/timeout/limit scenarios are scriptable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESOLVED_SHA = "a" * 39 + "b"
COMMITS_URL = "https://api.github.com/repos/acme/widgets/commits/v1.0.0"
RAW_URL = f"https://raw.githubusercontent.com/acme/widgets/{RESOLVED_SHA}/models/net.py"
IMPL_BODY = "class Net:\n    pass\n"
SUPPLEMENT_URL = "https://example.org/extra.txt"

DEFAULT_DISCOVERY = {
    "discovery_version": "menagerie.crawler.author-discovery.v1",
    "arm": "FOUND",
    "sources": [
        {
            "source_id": "impl-net",
            "kind": "forge-file",
            "repo": "github.com/acme/widgets",
            "path": "models/net.py",
            "ref": "v1.0.0",
            "role": "implementation",
            "media_type": "text/x-python",
        }
    ],
    "basis": "test discovery",
}

DEFAULT_RESULT = {"kind": "PROPOSED", "payload": {"arm": "PROPOSED"}}

_FAKE_CLAUDE_SOURCE = r'''
import json, os, re, sys, time, uuid

argv = sys.argv[1:]


def flag(name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


prompt = flag("-p") or ""
mode = os.environ.get("FAKE_CLAUDE_MODE", "happy")


def extract(label):
    matched = re.search(re.escape(label) + r"[^`]*`([^`]+)`", prompt)
    return matched.group(1) if matched else None


discovery_path = extract("- DISCOVERY output path, exact:")
result_path = extract("- RESULT output path, exact:")
required_path = extract("- REQUIRED output path, exact:")
if discovery_path:
    stage = "stage1"
elif "SUPPLEMENTARY SOURCE ROUND" in prompt:
    stage = "supplement"
elif result_path:
    stage = "stage2"
elif required_path:
    stage = "probe"
else:
    stage = "unknown"

log_dir = os.environ.get("FAKE_CLAUDE_LOG")
if log_dir:
    with open(os.path.join(log_dir, "invocations.jsonl"), "a") as fh:
        fh.write(
            json.dumps(
                {"pid": os.getpid(), "stage": stage, "argv": argv, "prompt": prompt}
            )
            + "\n"
        )
marker_dir = os.environ.get("FAKE_CLAUDE_MARKER")
if marker_dir:
    with open(os.path.join(marker_dir, stage + ".started"), "w") as fh:
        fh.write(str(os.getpid()))


def emit_harness(rc=0):
    sid = flag("--session-id") or str(uuid.uuid4())
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 1234,
                "duration_api_ms": 987,
                "num_turns": 7,
                "session_id": sid,
                "total_cost_usd": 0.0123,
                "usage": {
                    "input_tokens": 111,
                    "output_tokens": 222,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                },
                "result": "done",
            }
        )
    )
    sys.exit(rc)


sleep_s = float(os.environ.get("FAKE_CLAUDE_SLEEP", "120"))

if mode == "sleep-" + stage:
    time.sleep(sleep_s)
    sys.exit(0)
if mode == "late-write-stage2" and stage == "stage2":
    time.sleep(sleep_s)
    if result_path:
        with open(result_path, "w") as fh:
            fh.write(os.environ.get("FAKE_CLAUDE_LATE_RESULT", '{"kind": "STALE"}'))
    sys.exit(0)
if mode == "limit":
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "error_usage_limit",
                "is_error": True,
                "session_id": str(uuid.uuid4()),
            }
        )
    )
    sys.exit(1)
if mode == "crash":
    print("boom, not json")
    print("GitHub API rate limit exceeded for 1.2.3.4", file=sys.stderr)
    sys.exit(3)
if mode == "resume-fail" and "--resume" in argv:
    sys.exit(5)

if stage == "stage1" and discovery_path:
    payload = os.environ.get("FAKE_CLAUDE_DISCOVERY")
    if not payload:
        payload = json.dumps(json.loads(os.environ["FAKE_CLAUDE_DEFAULT_DISCOVERY"]))
    with open(discovery_path, "w") as fh:
        fh.write(payload)
elif stage == "stage2" and result_path:
    if mode == "supplement-request":
        supp = os.path.join(os.path.dirname(result_path), "supplement-request.json")
        with open(supp, "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "supplement_version": (
                            "menagerie.crawler.author-supplement-request.v1"
                        ),
                        "sources": [
                            {
                                "source_id": "supp-1",
                                "kind": "raw-url",
                                "url": os.environ["FAKE_CLAUDE_SUPPLEMENT_URL"],
                                "role": "documentation",
                            }
                        ],
                        "why": "one more file",
                    }
                )
            )
    else:
        with open(result_path, "w") as fh:
            fh.write(os.environ.get("FAKE_CLAUDE_RESULT") or '{"kind": "PROPOSED"}')
elif stage == "supplement" and result_path:
    with open(result_path, "w") as fh:
        fh.write(os.environ.get("FAKE_CLAUDE_RESULT") or '{"kind": "PROPOSED"}')
elif stage == "probe" and required_path:
    with open(required_path, "w") as fh:
        fh.write(json.dumps({"challenge_id": "test", "tools": {}}))
if os.environ.get("FAKE_CLAUDE_STDERR_NOISE"):
    print(os.environ["FAKE_CLAUDE_STDERR_NOISE"], file=sys.stderr)
emit_harness()
'''


def write_fake_claude(directory: Path) -> Path:
    """Write the fake harness script and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_claude.py"
    script.write_text(_FAKE_CLAUDE_SOURCE, encoding="utf-8")
    return script


def write_broker_fixtures(directory: Path) -> Path:
    """Write the hermetic broker transport fixtures and return the root."""

    directory.mkdir(parents=True, exist_ok=True)
    index = {
        COMMITS_URL: {"status": 200, "body_text": json.dumps({"sha": RESOLVED_SHA})},
        RAW_URL: {"status": 200, "body_text": IMPL_BODY},
        SUPPLEMENT_URL: {"status": 200, "body_text": "extra documentation\n"},
    }
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def executor_environment(
    monkeypatch,
    *,
    fake_claude: Path,
    fixtures: Path,
    log_dir: Path,
    campaign: str = "c1-mech",
) -> None:
    """Point the executor at the fake harness and hermetic broker fixtures."""

    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "MENAGERIE_AUTHOR_CLAUDE_BIN", f'"{sys.executable}" "{fake_claude}"'
    )
    monkeypatch.setenv("MENAGERIE_BROKER_FIXTURES", str(fixtures))
    monkeypatch.setenv("MENAGERIE_CAMPAIGN_ID", campaign)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_CLAUDE_DEFAULT_DISCOVERY", json.dumps(DEFAULT_DISCOVERY))
    monkeypatch.setenv("FAKE_CLAUDE_SUPPLEMENT_URL", SUPPLEMENT_URL)
    monkeypatch.delenv("FAKE_CLAUDE_MODE", raising=False)
    monkeypatch.delenv("MENAGERIE_EXECUTOR_PAUSE_AFTER", raising=False)


def subprocess_environment(
    *,
    fake_claude: Path,
    fixtures: Path,
    log_dir: Path,
    campaign: str = "c1-mech",
    **extra: str,
) -> dict:
    """Build the environment for running the executor as a real subprocess."""

    log_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "MENAGERIE_AUTHOR_CLAUDE_BIN": f'"{sys.executable}" "{fake_claude}"',
        "MENAGERIE_BROKER_FIXTURES": str(fixtures),
        "MENAGERIE_CAMPAIGN_ID": campaign,
        "FAKE_CLAUDE_LOG": str(log_dir),
        "FAKE_CLAUDE_DEFAULT_DISCOVERY": json.dumps(DEFAULT_DISCOVERY),
        "FAKE_CLAUDE_SUPPLEMENT_URL": SUPPLEMENT_URL,
    }
    env.pop("MENAGERIE_EXECUTOR_PAUSE_AFTER", None)
    env.pop("FAKE_CLAUDE_MODE", None)
    env.update(extra)
    return env


def write_source_request(root: Path, stable_id: str) -> Path:
    """Write one lane-shaped source-request envelope under the author root."""

    root.mkdir(parents=True, exist_ok=True)
    request = {
        "envelope_version": "menagerie.crawler.author-source-request.v1",
        "work_id": f"work-{stable_id}",
        "stable_id": stable_id,
        "untrusted_hints": {"name": stable_id, "zoo": "test-zoo"},
        "required_output_path": str(root / "source-targets.json"),
        "max_sources": 8,
        "required_fields": [
            "source_id",
            "url",
            "revision",
            "expected_sha256",
            "media_type",
        ],
    }
    path = root / "source-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def write_author_envelope(root: Path, stable_id: str) -> Path:
    """Write one minimal author envelope stub under the author root.

    The executor treats the envelope as opaque authority (the lane built and
    validated it); only the transport fields the executor reads are present.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(exist_ok=True)
    request = {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "c1-mech",
        "required_output_path": str(root / "result.json"),
        "allowed_model_dir": str(root / "model"),
        "prompt": {"path": str(root / "prompts" / "author.txt")},
    }
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "author.txt").write_text("canonical author prompt", encoding="utf-8")
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def read_invocations(log_dir: Path) -> list:
    """Return every recorded fake-harness invocation."""

    path = log_dir / "invocations.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
