"""Executed paper replications in ``examples/replications/`` with their stamped verdicts.

Each replication states one published claim and a verdict rule fixed before its first
run; these tests run every script on GPT-2 small (public Hub checkpoint at a pinned
revision, downloaded on first use) and pin the verdict it printed. A verdict test never
skips: a missing model or dependency is a failure. The data tests check the vendored
inputs against their pinned digests and re-check each selection rule. Every test that
loads GPT-2 small is marked slow: a cold cache adds the download to its runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

_FOLDER = Path(__file__).resolve().parents[1] / "examples" / "replications"
_MODULES = ("_common", "induction_heads", "ioi_name_movers", "causal_tracing", "make_data")

#: sha256 of the vendored data files (the IOI rows and known facts are rebuilt by
#: ``make_data.py`` from their pinned sources; the oracle is the induction oracle's
#: scores and protocol).
_DATA_SHA256 = {
    "induction_oracle.json": "90fcfa62802a26d8b78d6046bebc0f50191f6398dfab48c85618a5d855aecfbf",  # pragma: allowlist secret
    "ioi_rows.json": "8b7e3aae8221bd93d080224a8bff3e0c07cf9f5be3c46239c96336a9ca0fc63f",  # pragma: allowlist secret
    "known_facts.json": "a1ddab1d376adb8b7b5fa698626873e33389df8d0e4d25ad1bc09a2c544e4199",  # pragma: allowlist secret
}
#: sha256 of the induction protocol's token tensor (int64 bytes).
_IND_TOKENS_SHA256 = (
    "c77a8839b7f2254b5469748b8d32e25f1553312ce5f7dc1cf77f304869812b18"  # pragma: allowlist secret
)
#: sha256 of the 50 kept IOI rows (canonical JSON).
_IOI_ROWS_SHA256 = (
    "e9892e40f13b117ac3c163ab568ea609c7461e5577ad18397d3b40986ded9359"  # pragma: allowlist secret
)

#: sha256 of the 10 kept known facts (canonical JSON).
_CT_FACTS_SHA256 = (
    "49bbf62bde9b974815a9fdb3d1486d78ae54c0019c1819b928e825cc3207faf1"  # pragma: allowlist secret
)


@pytest.fixture
def scripts(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, ModuleType]]:
    """Import the replication scripts from their folder, then forget them."""

    monkeypatch.syspath_prepend(str(_FOLDER))
    for name in _MODULES:
        sys.modules.pop(name, None)
    yield {name: importlib.import_module(name) for name in _MODULES}
    for name in _MODULES:
        sys.modules.pop(name, None)


def _run(module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict:
    """Run a script's command-line entry and return its summary and stdout."""

    assert module.main(["--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    summary = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert f"CLAIM: {module.CLAIM}" in printed
    assert f"RULE: {module.RULE}" in printed
    assert f"VERDICT: {summary['verdict']}" in printed
    assert len(list(tmp_path.glob("*.tlspec"))) == 1
    return summary


def test_vendored_data_files_are_the_pinned_ones() -> None:
    for name, digest in _DATA_SHA256.items():
        assert hashlib.sha256((_FOLDER / "data" / name).read_bytes()).hexdigest() == digest


def test_r_ind_data_is_the_pinned_protocol(scripts: dict[str, ModuleType]) -> None:
    common, ind = scripts["_common"], scripts["induction_heads"]
    oracle = common.load_data("induction_oracle.json")
    tokens = ind.induction_tokens(oracle["protocol"])
    assert tuple(tokens.shape) == (8, 101)
    assert tokens[0, :8].tolist() == oracle["tokens_sha_preview"]
    assert common.tensor_sha256(tokens) == _IND_TOKENS_SHA256


@pytest.mark.slow
def test_r_ioi_rows_are_the_pinned_subset(scripts: dict[str, ModuleType]) -> None:
    common = scripts["_common"]
    _model, tokenizer = common.load_gpt2()
    data = common.load_data("ioi_rows.json")
    rows = data["rows"]
    assert [row["index"] for row in rows] == list(range(len(rows)))
    for row in rows:
        single = all(
            common.single_token(tokenizer, row[key]) is not None
            for key in ("indirect_object", "subject")
        )
        assert row["kept"] is single
    assert sum(row["kept"] for row in rows) == 50
    assert rows[-1]["kept"]


@pytest.mark.slow
def test_r_ct_facts_are_the_pinned_subset(scripts: dict[str, ModuleType]) -> None:
    common, make_data = scripts["_common"], scripts["make_data"]
    model, tokenizer = common.load_gpt2()
    records = common.load_data("known_facts.json")["records"]
    assert [record["known_id"] for record in records] == list(range(len(records)))
    for record in records:
        known = record["subject"] in record["prompt"] and make_data.is_known(
            model, tokenizer, record
        )
        assert record["kept"] is known
    assert sum(record["kept"] for record in records) == 30
    assert records[-1]["kept"]


def test_r_ct_window_layers_follow_the_paper(scripts: dict[str, ModuleType]) -> None:
    window_layers = scripts["causal_tracing"].window_layers
    assert [window_layers(layer, 1, 12) for layer in (0, 5, 11)] == [[0], [5], [11]]
    assert [window_layers(layer, 3, 12) for layer in (0, 5, 11)] == [[0, 1], [4, 5, 6], [10, 11]]
    assert window_layers(20, 10, 48) == list(range(15, 25))


@pytest.mark.slow
def test_r_ind_induction_heads_replicate(
    scripts: dict[str, ModuleType], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = _run(scripts["induction_heads"], tmp_path, capsys)
    measured = summary["measured"]
    assert measured["top3_oracle"] == ["5.5", "6.9", "7.10"]
    assert summary["data"]["tokens_sha256"] == _IND_TOKENS_SHA256
    passed = (
        measured["top3_torchlens"] == measured["top3_oracle"]
        and measured["max_abs_gap_vs_oracle"] <= 1e-4
    )
    assert summary["verdict"] == ("replicated" if passed else "did not replicate")
    assert summary["verdict"] == "replicated"  # stamped 2026-10-17


@pytest.mark.slow
def test_r_ioi_name_movers_replicate(
    scripts: dict[str, ModuleType], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = _run(scripts["ioi_name_movers"], tmp_path, capsys)
    measured = summary["measured"]
    assert measured["n_heads_ranked"] == 144
    assert summary["data"]["n_prompts"] == 50
    assert summary["data"]["rows_sha256"] == _IOI_ROWS_SHA256
    top5 = [name for name, _value in measured["top5_heads"]]
    passed = all(name in top5 for name in ("9.9", "9.6", "10.0"))
    assert summary["verdict"] == ("replicated" if passed else "did not replicate")
    assert summary["verdict"] == "replicated"  # stamped 2026-10-17


@pytest.mark.slow
def test_r_ct_mlp_causal_trace_replicates(
    scripts: dict[str, ModuleType], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = _run(scripts["causal_tracing"], tmp_path, capsys)
    measured = summary["measured"]
    assert summary["data"]["n_facts"] == 10
    assert summary["data"]["facts_sha256"] == _CT_FACTS_SHA256
    assert measured["patched_modules_are_the_12_mlps"]
    peak = measured["peak"]
    passed = peak["category"] == "last subject token" and peak["layer"] <= 7
    assert summary["verdict"] == ("replicated" if passed else "did not replicate")
    assert summary["verdict"] == "replicated"  # stamped 2026-10-17
