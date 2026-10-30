"""The ACDC driver-loop recipe (``docs/field-guide/acdc_ioi.md``) runs cold on CPU.

The page's fences run top to bottom in one namespace, each ``RESULT``
compared with its ``# recipe-test`` header (the recipe-page convention of
``tests/test_recipe_pages_executed.py``). Then: every edge decision is a
closed ledger entry whose verdict is ``supports`` (kept) or ``refutes``
(pruned); the head-knockout ``site_sweep`` entry carries ``not_assessed``;
and the pruning loop re-run here without the ledger prunes exactly the same
edges, in the same order, to the same KL. Stand-in: the page's two-layer
GPT-2 with random weights, walking its last two receivers (13 edge tests).
"""

from __future__ import annotations

import ast
import linecache
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers")
pytestmark = pytest.mark.heavy

PAGE = Path(__file__).resolve().parents[1] / "docs" / "recipes" / "acdc_ioi.md"
HEADER_RE = re.compile(r"# recipe-test: tool=(?P<tool>[a-z0-9_]+) expected=(?P<expected>.+)")
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> dict[str, Any]:
    import os

    import torch

    workdir = tmp_path_factory.mktemp("acdc")
    previous = os.getcwd()
    os.chdir(workdir)
    namespace: dict[str, Any] = {"__name__": "recipe_page"}
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        for index, match in enumerate(BLOCK_RE.finditer(PAGE.read_text("utf-8")), start=1):
            code = match.group("code")
            header = HEADER_RE.match(code.splitlines()[0])
            assert header is not None and header.group("tool") == "transformers", index
            filename = f"docs/field-guide/acdc_ioi.md:python-block-{index}"
            linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
            namespace["RESULT"] = None
            exec(compile(code, filename, "exec"), namespace)
            assert namespace["RESULT"] == ast.literal_eval(header.group("expected")), index
    finally:
        torch.use_deterministic_algorithms(deterministic)
        os.chdir(previous)
    return namespace


def test_every_decision_is_a_closed_ledger_verdict(page) -> None:
    entries = page["led"].entries()
    first, *decisions = entries.values()
    assert first.verdict["token"] == "not_assessed"
    assert first.steps, "the site_sweep material step lands in the first entry"
    assert len(decisions) == len(page["tested"]) == 13
    assert all(entry.status == "closed" for entry in decisions)
    assert {entry.verdict["token"] for entry in decisions} <= {"supports", "refutes"}
    assert all(entry.observations for entry in decisions)


def test_the_kept_set_reproduces_the_loops_own_pruning(page) -> None:
    graph, circuit_kl, tau = page["graph"], page["circuit_kl"], page["tau"]
    pruned: list[Any] = []
    current = circuit_kl(pruned)
    for receiver in (graph.receivers["logits"], graph.receivers["m1"]):
        for sender in reversed(receiver.senders):
            if sender.kind == "bias":
                continue
            edge = graph.edges[f"{sender.name}->{receiver.name}"]
            trial = circuit_kl(pruned + [edge])
            if trial - current < tau:
                pruned.append(edge)
                current = trial
    assert [e.name for e in pruned] == [e.name for e in page["pruned"]]
    assert current == page["current"]
    kept = [name for name in page["tested"] if name not in {p.name for p in pruned}]
    assert kept == page["kept"]
    assert 0 < len(kept) < 13
