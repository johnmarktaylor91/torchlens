"""``torchlens_dump`` serves every pass of a two-pass input Layer.

``tests/test_trace_boundary_ops_multipass.py`` covers ``Trace.input_ops``,
``build_agent_json``, ``trace.to_agent_json()`` and the folded
``torchlens_overview`` on a model whose two inputs fold into one two-pass input
Layer. This module adds the third agent entry point, ``torchlens_dump``, in each
of its views: the ``full`` view is the agent record and lists both input passes;
the ``overview`` and ``graph`` views must load and serve the same trace without
refusing on the multi-pass Layer.

Torch captures group by call site, where input nodes are singletons (grouping
rule section 2), so only the structural grouper (``grouper_v1``, still the
grouper for captures without call-site evidence) folds inputs. Like the sibling
module, the trace is captured under that grouper; the saved artifact keeps its
grouping on load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from support.structural_grouper import legacy_structural_grouping

from tests.test_trace_boundary_ops_multipass import _loop_trace
from torchlens.agent import call_tool


@pytest.fixture(scope="module")
def saved_loop(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    with legacy_structural_grouping():
        trace, inputs = _loop_trace()
    assert trace["input_1"].num_passes == 2, "the fixture must fold its inputs"
    expected = [f"{trace['input_1'].layer_label}:{index}" for index in (1, 2)]
    for label, value in zip(expected, inputs, strict=True):
        assert torch.equal(trace[label].out, value)
    path = tmp_path_factory.mktemp("findfix_dump") / "loop_inputs.tlspec"
    trace.save(path)
    return path, expected


def test_dump_full_view_lists_both_input_passes(saved_loop: tuple[Path, list[str]]) -> None:
    path, expected = saved_loop

    envelope = call_tool("torchlens_dump", {"path": str(path), "view": "full"})

    assert envelope["status"] == "ok"
    assert envelope["data"]["inputs"] == expected
    row_labels = [row["label"] for row in envelope["data"]["ops"]]
    assert [label for label in row_labels if label in expected] == expected


@pytest.mark.parametrize("view", ["overview", "graph"])
def test_dump_views_serve_a_two_pass_input(saved_loop: tuple[Path, list[str]], view: str) -> None:
    path, _ = saved_loop

    envelope = call_tool("torchlens_dump", {"path": str(path), "view": view})

    assert envelope["status"] == "ok"
    assert envelope["data"]["capture"]["capture_status"] == "complete"
