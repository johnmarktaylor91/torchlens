"""Rolled-renderer SVG honesty regressions for smart collapse (round 27).

The UNROLLED op-segment defect class was structurally closed in rounds 21-25.
These pins exercise its ROLLED-path siblings, a distinct code path that no
prior pin ever rendered (every earlier "rolled" pin stopped at the selector):

- HIGH: in rolled mode every cluster-OWNED segment (op segment AND child
  segment) lost its labeled node at render. ``_op_segment_owner_key`` and
  ``_segment_owner_key`` returned pass-QUALIFIED owners (``"mid:1"``,
  ``"p:1"``) while rolled clusters drain pass-FREE buckets, so
  ``_queue_segment_node`` posted the labeled node into a bucket the rolled
  cluster flush never drains. Graphviz then materialized a default ellipse
  whose visible label was the raw internal node name
  (``p_c0__segment__p_c2pass1``): the "-- N ops" honesty label, the dashed
  styling, and ALL disclosure of the hidden ops vanished.
- MED: the rolled branch of ``_build_collapsed_module_node`` never invoked
  the round-25 surfaced-exit remainder belt, while the atomic-exit drop in
  ``_collapse_address_for_node`` is vis_mode-independent -- so a rolled box
  kept counting its surfaced atomic-exit op although the exit rendered as
  its own separate node (one op double-represented per surfaced exit).

Every pin here renders a rolled SVG and asserts on the actual output: label
conservation (no hidden op unclaimed, no op double-represented), segment
node styling (labeled dashed box, never a default ellipse named by the raw
internal identifier), plan/SVG node-count parity on the rolled path, and
geometric containment of cluster-owned segments inside their owning rolled
cluster.
"""

import html
import re

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.visualization._segment_descriptors import (
    _op_segment_owner_key,
    _segment_owner_key,
)
from torchlens.visualization.collapse_optimizer import select_collapse_plan
from torchlens.visualization.collapse_plan import RenderContext


class RolledLeaf(nn.Module):
    """Three-op leaf block: linear child plus two own functional ops."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.l = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(self.l(x)))


class ChildSegParent(nn.Module):
    """Parent whose three leaf children condense into one child segment."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.c0 = RolledLeaf(width)
        self.c1 = RolledLeaf(width)
        self.c2 = RolledLeaf(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(self.c1(self.c0(x)))


class ChildSegNet(nn.Module):
    """Top chain + parent module + tail; rolled max makes a cluster-owned child segment."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.p = ChildSegParent(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        x = torch.sigmoid(x)
        x = x * 2
        x = self.p(x)
        return torch.tanh(torch.relu(x))


class OwnOpsMid(nn.Module):
    """Module whose own-op run sits strictly between child calls.

    The four functional own ops are bounded by module calls on both sides, so
    a max plan condenses them into an op segment whose members all share the
    ``mid`` module cluster -- the cluster-OWNED op-segment case.
    """

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.b0 = RolledLeaf(width)
        self.b1 = RolledLeaf(width)
        self.b2 = RolledLeaf(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b0(x)
        x = x * 2
        x = x + 1
        x = x - 0.5
        x = x / 2
        x = self.b1(x)
        return self.b2(x)


class OpSegNet(nn.Module):
    """Looped top/tail chains around ``OwnOpsMid`` for band pressure."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.mid = OwnOpsMid(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(10):
            x = torch.tanh(x)
            x = torch.sigmoid(x)
            x = x * 3
        x = self.mid(x)
        for _ in range(10):
            x = torch.relu(x)
            x = torch.tanh(x)
        return x


class AtomicExitInner(nn.Module):
    """Five boxed linear children plus one surfaced atomic-exit own op."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.ls = nn.ModuleList([nn.Linear(width, width) for _ in range(5)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.ls:
            x = layer(x)
        return torch.relu(x)


class AtomicExitNet(nn.Module):
    """Boxing ``inner`` must not double-count its separately drawn exit op."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.inner = AtomicExitInner(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        x = self.inner(x)
        return torch.sigmoid(x)


class MultiPassInner(nn.Module):
    """Three linear children plus a surfaced atomic-exit own op."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.ls = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.ls:
            x = layer(x)
        return torch.relu(x)


class MultiPassBoxNet(nn.Module):
    """Calls ``inner`` three times: a rolled box whose layers are multi-pass.

    Rolled remainder accounting runs on bare layer labels; a multi-pass layer
    base is an AMBIGUOUS op-accessor key, so the surfaced-exit scan must
    qualify it explicitly (round-27 follow-up: the first rolled remainder cut
    raised ``AmbiguousOpLookupError`` on every rolled box with reused
    layers). The box must also subtract its exit once per layer BASE, not
    once per pass.
    """

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.inner = MultiPassInner(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        for _ in range(3):
            x = self.inner(x)
        return torch.sigmoid(x)


FIXTURES = {
    "child_seg": ChildSegNet,
    "op_seg": OpSegNet,
    "atomic_exit": AtomicExitNet,
    "multipass_box": MultiPassBoxNet,
}

LEVELS = ["max", "auto", 1.0, 0.5]


def _svg_nodes(path):
    """Return (title, unescaped text lines, has_default_ellipse, body) per SVG node."""

    with open(path, encoding="utf-8") as handle:
        svg = handle.read()
    nodes = []
    for match in re.finditer(r'<g id="[^"]*" class="node">(.*?)</g>', svg, re.S):
        body = match.group(1)
        title = re.search(r"<title>(.*?)</title>", body, re.S)
        texts = [html.unescape(text) for text in re.findall(r"<text[^>]*>(.*?)</text>", body, re.S)]
        nodes.append((title.group(1) if title else "", texts, "<ellipse" in body, body))
    return nodes


def _svg_cluster_bboxes(path):
    """Return {cluster address: (xmin, ymin, xmax, ymax)} from cluster polygons."""

    with open(path, encoding="utf-8") as handle:
        svg = handle.read()
    boxes = {}
    for match in re.finditer(r'<g id="[^"]*" class="cluster">(.*?)</g>', svg, re.S):
        body = match.group(1)
        title = re.search(r"<title>cluster_(.*?)</title>", body, re.S)
        points = re.search(r'points="([^"]+)"', body)
        if title is None or points is None:
            continue
        coords = [tuple(map(float, pair.split(","))) for pair in points.group(1).split()]
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        boxes[title.group(1)] = (min(xs), min(ys), max(xs), max(ys))
    return boxes


def _node_coords(body):
    """Return all path/polygon/ellipse coordinates of one SVG node body."""

    coords = []
    for shape_match in re.finditer(r'(?:d|points)="([^"]+)"', body):
        coords.extend(
            (float(x), float(y))
            for x, y in re.findall(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", shape_match.group(1))
        )
    for ellipse in re.finditer(r'<ellipse[^>]*cx="([^"]+)"[^>]*cy="([^"]+)"', body):
        coords.append((float(ellipse.group(1)), float(ellipse.group(2))))
    return coords


def _is_boundary(title, texts):
    joined = " ".join(texts)
    return (
        re.match(r"^(input|output)_\d", title) is not None
        or "@input" in joined
        or "@output" in joined
    )


def _claimed_ops(texts):
    """Return the op count one rendered node claims (label conservation currency)."""

    joined = " ".join(texts)
    match = re.search(r"(\d+) ops?\b", joined)
    if match:
        return int(match.group(1))
    return 1


def _real_rolled_op_count(trace):
    """Return the number of distinct pass-free op layer bases, minus boundaries."""

    bases = {str(op.label).rsplit(":", 1)[0] for op in trace.ops}
    return len([base for base in bases if not re.match(r"^(input|output)_\d", base)])


def _rolled_draw(trace, tmp_path, tag, level):
    out = tmp_path / f"{tag}_{level}"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse=level,
        vis_mode="rolled",
        vis_outpath=str(out),
    )
    return str(out) + ".svg"


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
@pytest.mark.parametrize("level", LEVELS)
def test_rolled_svg_conservation_and_parity(fixture, level, tmp_path):
    """Rolled SVGs must conserve op claims exactly and match the rolled plan.

    - plan.total (rolled context) == rendered SVG node-group count;
    - sum of box/segment "N ops" claims + standalone nodes == distinct layer
      bases (nothing hidden without disclosure, nothing double-represented);
    - no segment node may render as a default ellipse or show its raw
      internal ``__segment__`` identifier as the visible label.
    """

    trace = tl.trace(FIXTURES[fixture]().eval(), torch.randn(2, 8))
    plan = trace.collapse_plan(mode=level, context=RenderContext(vis_mode="rolled"))
    svg_path = _rolled_draw(trace, tmp_path, fixture, level)
    nodes = _svg_nodes(svg_path)

    assert len(nodes) == plan.total

    claims = sum(
        _claimed_ops(texts) for title, texts, _, _ in nodes if not _is_boundary(title, texts)
    )
    assert claims == _real_rolled_op_count(trace)

    for title, texts, has_ellipse, _ in nodes:
        if "__segment__" not in title:
            continue
        assert not has_ellipse, f"segment {title} rendered as a default ellipse"
        joined = " ".join(texts)
        assert title not in joined, f"segment {title} shows its raw internal name"
        assert re.search(r"\d+ ops\b", joined), f"segment {title} lost its op-count label"


def test_rolled_cluster_owned_child_segment_renders_labeled(tmp_path):
    """A rolled child segment owned by an expanded parent must render labeled.

    Round-27 HIGH regression (child-segment variant): the descriptor owner was
    pass-qualified (``p:1``) while rolled clusters drain pass-free buckets, so
    the labeled node was silently dropped and Graphviz materialized a default
    ellipse labeled ``p_c0__segment__p_c2pass1``; the 9 hidden ops were
    disclosed nowhere.
    """

    trace = tl.trace(ChildSegNet().eval(), torch.randn(2, 8))
    result = select_collapse_plan(trace, RenderContext(vis_mode="rolled"), mode="max")
    owned = {
        name: descriptor
        for name, descriptor in (result.segments or {}).items()
        if descriptor.owner is not None
    }
    assert owned, "fixture must produce a cluster-owned segment under rolled max"
    for descriptor in owned.values():
        assert ":" not in descriptor.owner, (
            f"rolled segment owner {descriptor.owner!r} is pass-qualified; rolled "
            "clusters are keyed pass-free"
        )

    svg_path = _rolled_draw(trace, tmp_path, "childseg", "max")
    nodes = _svg_nodes(svg_path)
    segment_nodes = [entry for entry in nodes if "__segment__" in entry[0]]
    assert len(segment_nodes) == len(result.segments or {})
    title, texts, has_ellipse, body = segment_nodes[0]
    joined = " ".join(texts)
    assert not has_ellipse
    assert "stroke-dasharray" in body, "segment must keep its dashed styling"
    assert re.search(r"3 blocks, 9 ops", joined), joined

    bboxes = _svg_cluster_bboxes(svg_path)
    assert "p" in bboxes, "rolled cluster p must render"
    xmin, ymin, xmax, ymax = bboxes["p"]
    coords = _node_coords(body)
    assert coords
    assert all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in coords), (
        "cluster-owned segment node must render inside its owning rolled cluster"
    )


def test_rolled_cluster_owned_op_segment_renders_labeled(tmp_path):
    """A rolled op segment of module-own ops must render inside its cluster.

    Round-27 HIGH regression (op-segment variant): ``_op_segment_owner_key``
    compared stack levels pass-free in rolled mode but returned the
    pass-qualified first entry (``mid:1``), orphaning the labeled node.
    """

    trace = tl.trace(OpSegNet().eval(), torch.randn(2, 8))
    result = select_collapse_plan(trace, RenderContext(vis_mode="rolled"), mode="max")
    descriptors = result.segments or {}
    owned_op_segments = {
        name: descriptor
        for name, descriptor in descriptors.items()
        if descriptor.kind == "op" and descriptor.owner is not None
    }
    assert owned_op_segments, "fixture must produce a cluster-owned op segment"
    assert all(":" not in descriptor.owner for descriptor in owned_op_segments.values())
    assert all(descriptor.owner == "mid" for descriptor in owned_op_segments.values())

    svg_path = _rolled_draw(trace, tmp_path, "opseg", "max")
    nodes = _svg_nodes(svg_path)
    bboxes = _svg_cluster_bboxes(svg_path)
    assert "mid" in bboxes
    xmin, ymin, xmax, ymax = bboxes["mid"]

    rendered = {title: (texts, has_ellipse, body) for title, texts, has_ellipse, body in nodes}
    for name, descriptor in owned_op_segments.items():
        assert name in rendered, f"declared segment node {name} missing from the SVG"
        texts, has_ellipse, body = rendered[name]
        joined = " ".join(texts)
        assert not has_ellipse, f"segment {name} rendered as a default ellipse"
        assert name not in joined
        assert f"{descriptor.num_ops} ops" in joined
        assert "stroke-dasharray" in body
        coords = _node_coords(body)
        assert coords
        assert all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in coords), (
            f"segment {name} must render inside rolled cluster mid"
        )


def test_rolled_box_subtracts_surfaced_exit(tmp_path):
    """A rolled collapsed box must not count its separately drawn exit op.

    Round-27 MED regression: the rolled branch of
    ``_build_collapsed_module_node`` had no remainder call, so the box said
    "6 ops" while ``relu`` (``@inner``) also rendered standalone -- one op
    double-represented, exactly the round-25 class on the rolled path.
    """

    trace = tl.trace(AtomicExitNet().eval(), torch.randn(2, 8))
    svg_path = _rolled_draw(trace, tmp_path, "atomic_exit", "max")
    nodes = _svg_nodes(svg_path)

    box_texts = next(texts for title, texts, _, _ in nodes if title == "inner")
    exit_nodes = [
        (title, texts)
        for title, texts, _, _ in nodes
        if title.startswith("relu") and "@inner" in " ".join(texts)
    ]
    assert exit_nodes, "surfaced atomic exit must render as its own node"
    assert "5 ops" in " ".join(box_texts), box_texts
    assert "6 ops" not in " ".join(box_texts)


def test_rolled_multipass_box_subtracts_exit_in_layer_currency(tmp_path):
    """A multi-pass rolled box must subtract its exit once per layer BASE.

    Also guards the op-accessor ambiguity: rolled remainder accounting feeds
    bare layer labels to the surfaced-exit scan, and a reused layer base is
    an ambiguous accessor key unless qualified to its first pass.
    """

    trace = tl.trace(MultiPassBoxNet().eval(), torch.randn(2, 8))
    svg_path = _rolled_draw(trace, tmp_path, "multipass_box", "max")
    nodes = _svg_nodes(svg_path)

    box_texts = next(texts for title, texts, _, _ in nodes if title == "inner")
    joined = " ".join(box_texts)
    assert "(x3)" in joined, joined
    assert "3 ops" in joined, joined
    assert "4 ops" not in joined
    exit_nodes = [
        (title, texts)
        for title, texts, _, _ in nodes
        if title.startswith("relu") and "@inner" in " ".join(texts)
    ]
    assert exit_nodes, "surfaced multi-pass exit must render as its own rolled node"


def test_rolled_owner_keys_are_pass_free():
    """Both segment owner-key helpers must emit pass-free owners in rolled mode.

    Rolled clusters are keyed by pass-free module address; a pass-qualified
    owner names a bucket the rolled cluster flush never drains.
    """

    trace = tl.trace(OpSegNet().eval(), torch.randn(2, 8))
    # Resolve the mid-owned own-op labels directly from the module log.
    mid_own = [
        label
        for label in trace.modules["mid"].layer_labels
        if getattr(trace.ops[label], "modules", None)
        and [str(m) for m in trace.ops[label].modules][-1].rsplit(":", 1)[0] == "mid"
        and not getattr(trace.ops[label], "is_atomic_module", False)
    ]
    assert len(mid_own) >= 3, mid_own
    rolled_owner = _op_segment_owner_key(trace, tuple(mid_own), "rolled")
    assert rolled_owner == "mid"
    unrolled_owner = _op_segment_owner_key(trace, tuple(f"{label}:1" for label in mid_own))
    assert unrolled_owner == "mid:1"

    addresses = ("mid.b1", "mid.b2")
    assert _segment_owner_key(trace, addresses, "rolled") == "mid"
    assert _segment_owner_key(trace, addresses) == "mid:1"
    assert _segment_owner_key(trace, addresses, "unrolled") == "mid:1"


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_unrolled_conservation_unchanged(fixture, tmp_path):
    """The unrolled path must keep exact claim conservation on these fixtures.

    Guards the round-27 fix surface itself: the owner-key rolled branches and
    the render-boundary owner projection must be no-ops for unrolled draws
    (rounds 21-25 sealed behavior).
    """

    trace = tl.trace(FIXTURES[fixture]().eval(), torch.randn(2, 8))
    plan = trace.collapse_plan(mode="max")
    out = tmp_path / f"{fixture}_unrolled"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    nodes = _svg_nodes(str(out) + ".svg")
    assert len(nodes) == plan.total

    claims = sum(
        _claimed_ops(texts) for title, texts, _, _ in nodes if not _is_boundary(title, texts)
    )
    real = len(
        [
            op
            for op in trace.ops
            if not re.match(r"^(input|output)_\d", str(op.label).rsplit(":", 1)[0])
        ]
    )
    assert claims == real

    for title, texts, has_ellipse, _ in nodes:
        if "__segment__" in title:
            assert not has_ellipse
            assert title not in " ".join(texts)
