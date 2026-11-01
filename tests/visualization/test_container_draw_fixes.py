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
