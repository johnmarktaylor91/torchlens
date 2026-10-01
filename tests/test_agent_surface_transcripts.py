"""F29: the two gallery transcripts, byte-reproducible (the lane's row gate).

Transcript 1 (flagship, MCP-shaped): gpt2 clean vs middle-block-ablated --
the agent recovers the KNOWN intervention boundary and the direction of
downstream change from two real artifacts without executing code, parsing
prose, or receiving a tensor. Transcript 2 (triage/CI, CLI-shaped): three
artifacts, one NaN-poisoned, ending in the CI wedge
(``torchlens diff base cand --fail-on mismatch``).

Byte-reproducibility: each transcript is the concatenation of canonical
envelopes; generated twice in FRESH subprocesses over the same pinned
artifacts, the two transcripts must be byte-identical.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_agent_surface_helpers import (
    save_ablated_artifact,
    save_clean_artifact,
    save_nan_artifact,
)

REPO_ROOT = str(Path(__file__).resolve().parent.parent)

#: The flagship transcript script: runs the full tool sequence over two
#: artifact paths passed as argv, emitting one canonical envelope per line.
_FLAGSHIP_SCRIPT = """
import json, sys
sys.path.insert(0, {repo_root!r})
from torchlens.agent import call_tool, call_tool_envelope, canonical_dumps

clean, ablated = sys.argv[1], sys.argv[2]
steps = [
    ("torchlens_doctor", {{}}),
    ("torchlens_api_map", {{"name": "trace"}}),
    ("torchlens_overview", {{"path": clean, "mode": "manifest"}}),
    ("torchlens_overview", {{"path": ablated, "mode": "manifest"}}),
    ("torchlens_overview", {{"path": ablated}}),
    ("torchlens_query_sites", {{"path": clean, "query": {{"op": "and", "items": [
        {{"op": "func", "value": "{site_func}"}}, {{"op": "saved", "value": True}}]}},
        "max_rows": 2}}),
    ("torchlens_payload_stats", {{"path": clean, "query": {{"op": "saved", "value": True}},
        "metrics": ["mean", "l2_norm", "nan_count"]}}),
    ("torchlens_compare", {{"reference": clean, "subject": ablated}}),
    ("torchlens_explain", {{"path": ablated, "max_tokens": 800}}),
]
transcript = []
for name, args in steps:
    envelope = call_tool(name, args)
    envelope["torchlens_version"] = "TRANSCRIPT"
    transcript.append(canonical_dumps(envelope))
# Follow the continuation from the paged query (page 2).
page1 = json.loads(transcript[5])
next_struct = page1["data"]["next"]
if next_struct is not None:
    envelope = call_tool("torchlens_query_sites", {{"path": clean, "query": {{"op": "and",
        "items": [{{"op": "func", "value": "{site_func}"}}, {{"op": "saved", "value": True}}]}},
        "max_rows": 2, "continuation": next_struct}})
    envelope["torchlens_version"] = "TRANSCRIPT"
    transcript.append(canonical_dumps(envelope))
# Deliberately request one unsaved site and keep the typed remedy.
error = call_tool_envelope("torchlens_payload_stats", {{"path": clean, "labels": ["{unsaved}"]}})
error["torchlens_version"] = "TRANSCRIPT"
transcript.append(canonical_dumps(error))
sys.stdout.write("\\n".join(transcript))
"""


def _run_transcript(script: str, argv: list[str]) -> str:
    """Run one transcript script in a fresh subprocess and return its bytes."""

    result = subprocess.run(
        [sys.executable, "-c", script, *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return result.stdout


@pytest.mark.slow
def test_small_flagship_transcript_is_byte_reproducible(tmp_path: Path) -> None:
    """The flagship sequence over the deterministic small pair, twice."""

    clean = save_clean_artifact(tmp_path)
    ablated = save_ablated_artifact(tmp_path)
    script = _FLAGSHIP_SCRIPT.format(repo_root=REPO_ROOT, site_func="relu", unsaved="linear_1_1:1")
    first = _run_transcript(script, [str(clean), str(ablated)])
    second = _run_transcript(script, [str(clean), str(ablated)])
    assert first == second, "transcript bytes varied across fresh processes"
    lines = first.splitlines()
    compare = json.loads(lines[7])
    # The acceptance claim, deliberately narrow: the known boundary and the
    # direction of downstream change, from artifacts alone.
    by_label = {row.get("label"): row for row in compare["data"]["rows"]}
    assert by_label["relu_1_2:1"]["allclose"] is True
    assert by_label["relu_2_4:1"]["allclose"] is False
    assert by_label["relu_3_6:1"]["allclose"] is False
    assert compare["data"]["coverage"]["sites_value_compared"] == 3
    # The unsaved-site refusal carries the recapture remedy.
    error_step = json.loads(lines[-1])
    assert error_step["data"]["rows"][0]["status"] == "unsaved"
    assert "save=" in error_step["data"]["rows"][0]["remedy"]


@pytest.mark.slow
def test_gpt2_flagship_transcript_is_byte_reproducible(tmp_path: Path) -> None:
    """The REAL flagship: gpt2 clean vs middle-block-ablated, twice.

    Real-model law (memo section 5): the fold, budgets, and compare claims
    are invisible at toy depth; this is the pinned real-checkpoint scenario.
    """

    transformers = pytest.importorskip("transformers")
    import torch

    import torchlens as tl

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    input_ids = torch.arange(12).unsqueeze(0)
    save_predicate = tl.in_module("transformer.h.5.mlp")
    clean_log = tl.trace(model, (), input_kwargs={"input_ids": input_ids}, save=save_predicate)
    clean_path = tmp_path / "gpt2_clean.tlspec"
    tl.save(clean_log, str(clean_path))
    ablated_log = tl.trace(
        model,
        (),
        input_kwargs={"input_ids": input_ids},
        save=save_predicate,
        intervene=tl.when(tl.func("tanh") & tl.in_module("transformer.h.5.mlp"), tl.zero_ablate()),
    )
    ablated_path = tmp_path / "gpt2_ablated.tlspec"
    tl.save(ablated_log, str(ablated_path))
    del model, clean_log, ablated_log

    script = _FLAGSHIP_SCRIPT.format(repo_root=REPO_ROOT, site_func="tanh", unsaved="ln_f.weight")
    first = _run_transcript(script, [str(clean_path), str(ablated_path)])
    second = _run_transcript(script, [str(clean_path), str(ablated_path)])
    assert first == second, "gpt2 transcript bytes varied across fresh processes"
    compare = json.loads(first.splitlines()[7])
    coverage = compare["data"]["coverage"]
    assert coverage["sites_value_compared"] > 0
    assert coverage["sites_changed"] > 0  # the ablation is visible downstream
    folded = json.loads(first.splitlines()[4])
    assert folded["data"]["n_classes"] < folded["data"]["counts"]["operations"]


@pytest.mark.heavy
def test_cli_triage_transcript(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Transcript 2: ls -> preflight -> overview/audit -> query -> diff gate."""

    from torchlens.agent import cli

    clean = save_clean_artifact(tmp_path)
    ablated = save_ablated_artifact(tmp_path)
    save_nan_artifact(tmp_path)

    assert cli.main(["ls", str(tmp_path), "--json"]) == cli.EXIT_OK
    listing = json.loads(capsys.readouterr().out)
    assert sorted(listing["data"]["artifacts"]) == [
        "ablated.tlspec",
        "clean.tlspec",
        "nonfinite.tlspec",
    ]
    nan_path = str(tmp_path / "nonfinite.tlspec")
    assert cli.main(["info", nan_path, "--json"]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["overview", nan_path, "--json", "--fail-on", "nonfinite"]) == cli.EXIT_GATE
    capsys.readouterr()
    assert (
        cli.main(["query", nan_path, "--json", "--query", '{"op": "contains", "value": "truediv"}'])
        == cli.EXIT_OK
    )
    query_out = json.loads(capsys.readouterr().out)
    assert query_out["data"]["header"]["matches_total"] == 1
    handoff = query_out["data"]["handoff"]
    import torchlens as tl

    scope: dict = {"log": tl.load(nan_path)}
    local = eval(handoff, scope)  # noqa: S307 - handoff contract: runs verbatim
    assert local == [row["label"] for row in query_out["data"]["rows"]]
    # The regression pair: exit 0 on self, exit 1 on drift.
    assert cli.main(["diff", str(clean), str(clean), "--json", "--fail-on", "mismatch"]) == 0
    capsys.readouterr()
    assert cli.main(["diff", str(clean), str(ablated), "--json", "--fail-on", "mismatch"]) == 1
