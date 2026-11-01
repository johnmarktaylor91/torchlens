"""Container drawing regressions found by rendering every ``show_containers`` mode.

The models are the six container shapes of the visuals-guide containers section: a dict
returned by a submodule as the model output, a Hugging Face ``ModelOutput`` with a nested
tuple, dict and tuple inputs, a 16-leaf homogeneous tuple, a mid-graph dict, and a one-layer
GPT-2. The SVG checks need the Graphviz ``dot`` binary: what is in the DOT source is not
always what Graphviz draws.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PayloadUnavailableError

DOT_AVAILABLE = shutil.which("dot") is not None
needs_dot = pytest.mark.skipif(not DOT_AVAILABLE, reason="Graphviz dot binary not installed")
STRUCTURE = tl.options.CaptureOptions(capture_container_structure=True)


class Heads(nn.Module):
    """Two heads returned as a dict."""

    def __init__(self) -> None:
        super().__init__()
        self.cls = nn.Linear(8, 3)
        self.box = nn.Linear(8, 4)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.cls(h), "box": self.box(h)}


class TwoHeads(nn.Module):
    """Returns the dict its ``heads`` submodule built."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 8)
        self.heads = Heads()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(torch.relu(self.body(x)))


class Fuse(nn.Module):
    """Takes a dict input and a tuple input."""

    def __init__(self) -> None:
        super().__init__()
        self.img = nn.Linear(6, 4)
        self.txt = nn.Linear(5, 4)
        self.out = nn.Linear(4, 2)

    def forward(
        self, batch: dict[str, torch.Tensor], pair: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        a = self.img(batch["image"]) * batch["mask"]
        t = self.txt(pair[0]) + pair[1]
        return self.out(a + t)


class WideTuple(nn.Module):
    """Returns 16 same-shape tensors (over the 12-leaf inline ceiling)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.proj(x)
        return tuple(h + float(i) for i in range(16))


class Pair(nn.Module):
    """Returns a dict that its parent takes apart."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"left": self.left(x), "right": self.right(x)}


class MidGraph(nn.Module):
    """A dict made mid-graph by ``pair`` and consumed inside the model."""

    def __init__(self) -> None:
        super().__init__()
        self.pair = Pair()
        self.mix = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pair(x)
        return self.mix(p["left"] * p["right"])


def _tiny_classifier() -> nn.Module:
    """Returns a ``SequenceClassifierOutput`` with a nested tuple of hidden states."""

    modeling_outputs = pytest.importorskip("transformers.modeling_outputs")

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(4, 8)
            self.layer1 = nn.Linear(8, 8)
            self.cls = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> Any:
            h0 = self.embed(x)
            h1 = torch.tanh(self.layer1(h0))
            return modeling_outputs.SequenceClassifierOutput(
                logits=self.cls(h1), hidden_states=(h0, h1)
            )

    return TinyClassifier()


def _tiny_gpt2() -> nn.Module:
    """One-layer GPT-2 with random weights (no download)."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=50, n_positions=16)
    return transformers.GPT2LMHeadModel(config).eval()


def _fuse_inputs() -> list[Any]:
    return [
        {"image": torch.randn(1, 6), "mask": torch.ones(1, 4)},
        (torch.randn(1, 5), torch.randn(1, 4)),
    ]


def _draw(trace: Any, tmp_path: Path, name: str, mode: Any, fmt: str = "dot") -> str:
    """Return the DOT source of a draw (also writes ``<name>.<fmt>``)."""

    return str(trace.draw(path=str(tmp_path / name), file_format=fmt, show_containers=mode))


def _edges_into(source: str, head: str) -> list[str]:
    """Return the attribute text of every DOT edge whose head is ``head``."""

    return re.findall(rf"^\s*\S+ -> {re.escape(head)} \[(.*)\]\s*$", source, re.MULTILINE)


def _keys_kept_in_svg(source: str, svg: str) -> tuple[list[str], list[str]]:
    """Return (keys on non-constraining DOT edges, those whose text the SVG edge draws)."""

    wanted: list[tuple[str, str, str]] = []
    for tail, head, attrs in re.findall(r"^\s*(\S+) -> (\S+) \[(.*)\]\s*$", source, re.MULTILINE):
        key = re.search(r'POINT-SIZE="8">([^<]*)</FONT>', attrs)
        if key and "constraint=false" in attrs:
            wanted.append((tail, head, key.group(1)))
    edge_bodies: dict[str, list[str]] = {}
    for title, body in re.findall(
        r'<g id="edge\d+" class="edge">\s*<title>([^<]*)</title>(.*?)</g>', svg, re.S
    ):
        edge_bodies.setdefault(title, []).append(body)
    kept = [
        key
        for tail, head, key in wanted
        if any(
            re.search(rf">{re.escape(key)}</text>", body)
            for body in edge_bodies.get(f"{tail}&#45;&gt;{head}", [])
        )
    ]
    return [key for _, _, key in wanted], kept


def test_nodes_mode_draws_a_collapsed_wide_tuple_once(tmp_path: Path) -> None:
    """The 16-leaf model output is one summary node carrying the model-output role."""

    trace = tl.trace(WideTuple(), torch.randn(1, 4), capture=STRUCTURE)
    source = _draw(trace, tmp_path, "wide", "nodes")

    assert "container_node_" not in source
    assert source.count("tuple x16") == 1
    assert "tuple x16 (model output)" in source
    into_summary = _edges_into(source, "container_final_output_0_tuple")
    assert len(into_summary) == 16
    assert not any("dashed" in attrs for attrs in into_summary)


def test_collapsed_mode_summary_keeps_its_plain_label(tmp_path: Path) -> None:
    """The role is added in nodes mode only; collapsed mode keeps its picture."""

    trace = tl.trace(WideTuple(), torch.randn(1, 4), capture=STRUCTURE)
    source = _draw(trace, tmp_path, "wide_collapsed", "collapsed")

    assert "tuple x16" in source
    assert "(model output)" not in source


def test_nodes_mode_names_a_submodule_dict_returned_as_model_output(tmp_path: Path) -> None:
    """A dict a submodule returns as the model output is drawn as the model-output sink."""

    trace = tl.trace(TwoHeads(), torch.randn(1, 4), capture=STRUCTURE)
    source = _draw(trace, tmp_path, "two_heads", "nodes")

    assert "dict[2] (model output)" in source
    assert "(call output)" not in source
    into_sink = _edges_into(source, "container_node_0")
    # One solid keyed edge per head; no dashed member-of tie on a boundary container.
    assert len(into_sink) == 2
    assert all("style=solid" in attrs and "arrowhead=none" not in attrs for attrs in into_sink)
    assert {re.search(r'POINT-SIZE="8">([^<]*)<', a).group(1) for a in into_sink} == {  # type: ignore[union-attr]
        "logits",
        "box",
    }


def test_nodes_mode_model_output_record_gets_no_member_ties(tmp_path: Path) -> None:
    """A ``ModelOutput`` the root module returns has keyed sink edges only."""

    trace = tl.trace(_tiny_classifier(), torch.randn(1, 4), capture=STRUCTURE)
    source = _draw(trace, tmp_path, "classifier", "nodes")

    assert "SequenceClassifierOutput (model output)" in source
    into_sink = _edges_into(source, "container_node_0")
    assert len(into_sink) == 3
    assert not any("arrowhead=none" in attrs for attrs in into_sink)


def test_nodes_mode_midgraph_dict_keeps_member_ties(tmp_path: Path) -> None:
    """A mid-graph container still gets dashed member-of ties and the call-output role."""

    trace = tl.trace(MidGraph(), torch.randn(1, 4), capture=STRUCTURE)
    source = _draw(trace, tmp_path, "midgraph", "nodes")

    assert "dict[2] (call output)" in source
    into_node = _edges_into(source, "container_node_0")
    assert len(into_node) == 2
    assert all("arrowhead=none" in attrs and "style=dashed" in attrs for attrs in into_node)


@needs_dot
@pytest.mark.parametrize("row", ["two_heads", "classifier", "fuse"])
def test_nodes_mode_svg_draws_every_container_key(tmp_path: Path, row: str) -> None:
    """Every key on a container edge appears in the drawn SVG, not only in the DOT."""

    if row == "two_heads":
        trace = tl.trace(TwoHeads(), torch.randn(1, 4), capture=STRUCTURE)
    elif row == "classifier":
        trace = tl.trace(_tiny_classifier(), torch.randn(1, 4), capture=STRUCTURE)
    else:
        trace = tl.trace(Fuse(), _fuse_inputs(), capture=STRUCTURE)
    source = _draw(trace, tmp_path, row, "nodes", fmt="svg")
    svg = (tmp_path / f"{row}.svg").read_text()

    wanted, kept = _keys_kept_in_svg(source, svg)

    assert wanted, "the nodes-mode DOT should carry keyed container edges"
    assert kept == wanted


@needs_dot
@pytest.mark.heavy
def test_cluster_mode_draws_the_gpt2_split_box_with_sibling_ordering(tmp_path: Path) -> None:
    """The split tuple's container box is drawn while sibling ordering is on (the default)."""

    trace = tl.trace(_tiny_gpt2(), torch.randint(0, 50, (1, 5)), capture=STRUCTURE)
    source = _draw(trace, tmp_path, "gpt2_cluster", "cluster", fmt="svg")
    svg = (tmp_path / "gpt2_cluster.svg").read_text()

    declared = set(re.findall(r"subgraph (cluster_container_\w+) \{", source))
    drawn = set(re.findall(r'class="cluster">\s*<title>(cluster_container_\w+)</title>', svg))
    assert declared, "the cluster-mode DOT should declare the split tuple's container cluster"
    assert drawn == declared
    assert "tl:sibling-order" in source


@pytest.mark.heavy
def test_trace_output_on_an_opaque_cache_names_the_real_cause() -> None:
    """With container structure captured, the refusal names the opaque member, not the option."""

    trace = tl.trace(_tiny_gpt2(), torch.randint(0, 50, (1, 5)), capture=STRUCTURE)

    with pytest.raises(PayloadUnavailableError) as excinfo:
        trace.output

    message = str(excinfo.value)
    assert excinfo.value.fields["code"] == "container_not_reconstructable"
    assert "opaque" in message
    assert "capture_container_structure=True" not in message
    assert "use_cache=False" in message
