"""S5 escaping test pack (E1-E6): the precondition for any HTML-label ship.

The S5 contract (L5 design memo sec 1.4) requires this pack to land BEFORE any
encoding channel, suppression, or summary-row work ships. Scope: (1) node
labels through the ONE HTML choke point ``render_lines_to_html`` and (2)
graph-proper labels through ``_render_utils.html_escape`` (cluster titles,
edge labels, graph titles). ``code_panel.py`` and ``fastlog_live.py`` carry
their own local escapes and are OUT of S5 scope.
"""

from __future__ import annotations

import re
from html import escape as html_stdlib_escape
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.layer import Layer
from torchlens.visualization._render_edges import (
    _html_combined_recurrence_label,
    _html_container_edge_label,
    _html_edge_label,
)
from torchlens.visualization._render_utils import html_escape
from torchlens.visualization.node_spec import NodeSpec, render_lines_to_html

# --------------------------------------------------------------------------
# E1 corpus. TRANSFORMABLE members carry at least one of &/</> and therefore
# have a well-defined escaped-vs-raw distinction; ESCAPE_INVARIANT members are
# unchanged by html.escape(quote=False) BY DESIGN (quotes, newlines, ordinary
# unicode), so only exact preservation is assertable for them (sol r2 M4).
# --------------------------------------------------------------------------
TRANSFORMABLE_CORPUS = (
    "<B>x</B>",
    "a & b",
    "</TD></TR>",
    "<<TABLE",
    "5 > 4 < 6",
)
ESCAPE_INVARIANT_CORPUS = (
    'quo"te',
    "new\nline",
    "unicöde ↦ ok",
)
#: The documented ModuleDict hazard (_render_dot.py module-title path): a
#: ModuleDict key carrying ``&`` must escape in cluster titles.
MODULEDICT_HAZARD_KEY = "amp&key"


def _escaped(text: str) -> str:
    """Return the choke-point escaping of ``text`` (html.escape, quote=False)."""

    return html_stdlib_escape(text, quote=False)


def _sentineled(index: int, member: str) -> str:
    """Wrap a corpus member in a unique sentinel so raw-absence is decidable.

    ``<<TABLE`` is a substring of every HTML table label the composer itself
    emits, so raw-absence of the bare member is structurally unsatisfiable;
    the sentinel makes each injected occurrence unique.
    """

    return f"S5E1c{index}[{member}]"


def _render_dot(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render a Trace to DOT source, running Graphviz (parse validation)."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    )


def _assert_member_escaped(dot: str, injected: str) -> None:
    """Assert one injected corpus string only appears escaped in ``dot``."""

    escaped = _escaped(injected)
    if escaped != injected:
        assert escaped in dot, f"escaped form missing for {injected!r}"
        assert injected not in dot, f"raw form leaked for {injected!r}"
    else:
        # Escape-invariant member: exact preservation is the contract.
        assert injected in dot, f"escape-invariant member missing: {injected!r}"


def _detect_raw_markup(dot: str, marker: str) -> bool:
    """E5 detector: return True when ``marker`` appears UNESCAPED in ``dot``."""

    return marker in dot


class _CorpusLinesModel(nn.Module):
    """Tiny model for node-line injection tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _corpus_module_dict_model() -> nn.Module:
    """Model whose ModuleDict keys carry the E1 corpus (module-name placement).

    ``add_module`` rejects names containing ``"."`` and empty names only; the
    corpus contains neither.
    """

    class CorpusDictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            keys = [
                _sentineled(i, member)
                for i, member in enumerate(TRANSFORMABLE_CORPUS + ESCAPE_INVARIANT_CORPUS)
                if "\n" not in member  # Graphviz cluster ids cannot carry newlines.
            ]
            keys.append(MODULEDICT_HAZARD_KEY)
            self.blocks = nn.ModuleDict({key: nn.Linear(4, 4) for key in keys})

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks.values():
                x = torch.relu(block(x))
            return x

    return CorpusDictModel()


# --------------------------------------------------------------------------
# E1: adversarial corpus round-trips at every declared placement.
# --------------------------------------------------------------------------


def test_e1_corpus_via_node_spec_lines(tmp_path: Path) -> None:
    """(b) user node_spec_fn lines: escaped present, raw absent, dot parses."""

    model = _CorpusLinesModel()
    log = tl.trace(model, torch.randn(1, 4))
    injected = [
        _sentineled(i, member)
        for i, member in enumerate(TRANSFORMABLE_CORPUS + ESCAPE_INVARIANT_CORPUS)
    ]

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
        if layer_log.layer_type == "linear":
            return default_spec.replace(lines=[*default_spec.lines, *injected])
        return None

    dot = _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)
    for member in injected:
        _assert_member_escaped(dot, member)


def test_e1_corpus_via_module_names(tmp_path: Path) -> None:
    """(a) module names (ModuleDict keys incl. the documented ``&`` hazard)."""

    log = tl.trace(_corpus_module_dict_model(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path)
    for i, member in enumerate(TRANSFORMABLE_CORPUS + ESCAPE_INVARIANT_CORPUS):
        if "\n" in member:
            continue
        _assert_member_escaped(dot, _sentineled(i, member))
    assert "amp&amp;key" in dot
    # The raw key may legally appear only inside quoted plain-DOT identifiers
    # (node names / tooltips), never inside an HTML-like label.
    for html_label in re.findall(r"label=<(.*?)>\s*[\],]", dot, flags=re.DOTALL):
        assert MODULEDICT_HAZARD_KEY not in html_label


def test_e1_corpus_via_func_config_values(tmp_path: Path) -> None:
    """(c) func_config values: adversarial strings injected at the record layer.

    No shipped torch op accepts arbitrary user strings into ``func_config``,
    so the injection is a record-layer mutation on a finished trace — the
    render path downstream is identical.
    """

    model = _CorpusLinesModel()
    log = tl.trace(model, torch.randn(1, 4))
    injected = {
        f"s5_e1_key_{i}": _sentineled(i, member)
        for i, member in enumerate(TRANSFORMABLE_CORPUS + ESCAPE_INVARIANT_CORPUS)
    }
    target = next(layer for layer in log._layer_list if layer.type == "linear")
    target.func_config = {**(target.func_config or {}), **injected}

    dot = _render_dot(log, tmp_path)
    for member in injected.values():
        _assert_member_escaped(dot, member)


# --------------------------------------------------------------------------
# E2: escaper equivalence pin — two implementations, one invariant.
# --------------------------------------------------------------------------


def test_e2_escaper_equivalence() -> None:
    """node_spec's html.escape(quote=False) and _render_utils.html_escape agree."""

    corpus = (
        *TRANSFORMABLE_CORPUS,
        *ESCAPE_INVARIANT_CORPUS,
        MODULEDICT_HAZARD_KEY,
        "",
        "&amp; already-escaped",
    )
    for member in corpus:
        assert html_escape(member) == html_stdlib_escape(member, quote=False), member


# --------------------------------------------------------------------------
# E3: extra_attrs discipline — the power valve stays a power valve.
# --------------------------------------------------------------------------

_VIZ_ROOT = Path(__file__).parent.parent / "torchlens" / "visualization"

#: Internal write sites allowed to place a ``label`` key through extra_attrs.
#: Today: NONE. Intervention-hook helpers emit literal constants through
#: direct node-args dicts, not extra_attrs["label"].
_EXTRA_ATTRS_LABEL_ALLOWLIST: frozenset[str] = frozenset()


def test_e3_no_internal_extra_attrs_label_writes() -> None:
    """No TorchLens-internal call site sets extra_attrs["label"]."""

    offenders: list[str] = []
    label_write = re.compile(r"extra_attrs\[\s*['\"]label['\"]\s*\]\s*=")
    label_literal = re.compile(r"extra_attrs\s*=\s*\{[^}]*['\"]label['\"]\s*:", re.DOTALL)
    for path in sorted(_VIZ_ROOT.rglob("*.py")):
        text = path.read_text()
        if label_write.search(text) or label_literal.search(text):
            rel = str(path.relative_to(_VIZ_ROOT))
            if rel not in _EXTRA_ATTRS_LABEL_ALLOWLIST:
                offenders.append(rel)
    assert offenders == [], f"internal extra_attrs label writes: {offenders}"


def test_e3_extra_attrs_last_wins_and_unescaped() -> None:
    """Power-valve semantics: merged last (beats NodeSpec fields), unescaped."""

    from torchlens.visualization._render_leaf import _node_spec_to_graphviz_args

    spec = NodeSpec(
        lines=["title"],
        fillcolor="#111111",
        extra_attrs={"fillcolor": "#EEEEEE", "customattr": "a & b < c"},
    )
    args = _node_spec_to_graphviz_args(spec)
    assert args["fillcolor"] == "#EEEEEE"  # extra_attrs WINS by merge order.
    assert args["customattr"] == "a & b < c"  # raw graphviz attr: unescaped.


# --------------------------------------------------------------------------
# E4: tooltip robustness — quotes/newlines/& never break emitted DOT.
# --------------------------------------------------------------------------


def test_e4_tooltip_robustness(tmp_path: Path) -> None:
    """Adversarial tooltips render to valid DOT (graphviz quoting mechanism)."""

    model = _CorpusLinesModel()
    log = tl.trace(model, torch.randn(1, 4))
    hostile = 'tool"tip & <B>\nline'

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
        if layer_log.layer_type == "linear":
            return default_spec.replace(tooltip=hostile)
        return None

    # draw() runs the dot binary (svg) — an unquotable tooltip would fail here.
    dot = _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)
    assert "tooltip=" in dot


# --------------------------------------------------------------------------
# E5: negative control — the detector catches a deliberately smuggled raw tag.
# --------------------------------------------------------------------------


def test_e5_negative_control_detector_catches_raw_markup(tmp_path: Path) -> None:
    """A raw transformable tag smuggled past the choke point IS detected."""

    model = _CorpusLinesModel()
    log = tl.trace(model, torch.randn(1, 4))
    smuggled = "<B>S5E5RAW</B>"

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
        if layer_log.layer_type == "linear":
            # extra_attrs is the documented unescaped power valve — the one
            # sanctioned way to place raw markup — used here to calibrate the
            # detector in the failing direction. The marker is embedded in a
            # longer string so graphviz quotes the attr as a plain string
            # (a bare <...> value would be emitted as an HTML label).
            return default_spec.replace(
                extra_attrs={**default_spec.extra_attrs, "comment": f"raw {smuggled} raw"}
            )
        return None

    dot = _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)
    assert _detect_raw_markup(dot, smuggled), "detector failed to flag raw markup"
    # And the same detector stays quiet on a clean render (both directions).
    clean = _render_dot(log, tmp_path / "clean")
    assert not _detect_raw_markup(clean, smuggled)


# --------------------------------------------------------------------------
# E6: graph-proper end-to-end — one adversarial case per call-site family.
# --------------------------------------------------------------------------


def test_e6_graph_title_escapes_model_class_name(tmp_path: Path) -> None:
    """Graph title family: adversarial model class name (dynamic type name)."""

    adversarial_cls = type(
        "S5E6<B>&Model",
        (nn.Module,),
        {
            "__init__": lambda self: (
                nn.Module.__init__(self),
                setattr(self, "fc", nn.Linear(4, 4)),
            )[0],
            "forward": lambda self, x: torch.relu(self.fc(x)),
        },
    )
    log = tl.trace(adversarial_cls(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path)
    assert "S5E6&lt;B&gt;&amp;Model" in dot
    # The raw name legally appears once as the QUOTED plain-DOT digraph id
    # (graphviz quoting is the mechanism there); it must never appear inside
    # an HTML-like label.
    for html_label in re.findall(r"label=<(.*?)>\s*[\],]", dot, flags=re.DOTALL):
        assert "S5E6<B>&Model" not in html_label


def test_e6_cluster_title_escapes_module_name(tmp_path: Path) -> None:
    """Cluster title family: ModuleDict key with corpus content in a cluster."""

    class ClusterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleDict(
                {"e6clu <B>&title": nn.Sequential(nn.Linear(4, 4), nn.ReLU())}
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.blocks["e6clu <B>&title"](x)

    log = tl.trace(ClusterModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path)
    assert "e6clu &lt;B&gt;&amp;title" in dot
    for html_label in re.findall(r"label=<(.*?)>\s*[\],]", dot, flags=re.DOTALL):
        assert "e6clu <B>&title" not in html_label


def test_e6_edge_label_helpers_escape_user_text() -> None:
    """Edge label family: the two HTML edge-label composers escape user text.

    User-controlled strings reach edge labels through container labels (e.g. a
    DictKey pulled from a model's output dict); the composers are the choke
    points and are pinned directly with the transformable corpus.
    """

    for i, member in enumerate(TRANSFORMABLE_CORPUS):
        # Sentinel-wrapped: "</TD></TR>" is structural markup in the
        # composers' own tables, so bare raw-absence is undecidable (E1 note).
        injected = _sentineled(i, member)
        for label in (
            _html_edge_label(injected),
            _html_container_edge_label(injected),
            _html_combined_recurrence_label(injected, injected),
        ):
            assert _escaped(injected) in label, member
            assert injected not in label, member


def test_e6_dict_output_key_edge_label(tmp_path: Path) -> None:
    """End-to-end: an adversarial dict-output key renders escaped, dot parses."""

    class DictOutModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"e6edge<B>&key": torch.relu(self.fc(x))}

    log = tl.trace(DictOutModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path)
    # Wherever the key surfaces inside an HTML-like label it must be escaped.
    for html_label in re.findall(r"label=<(.*?)>\s*[\],]", dot, flags=re.DOTALL):
        assert "e6edge<B>&key" not in html_label
    if "e6edge" in dot:
        assert ("e6edge&lt;B&gt;&amp;key" in dot) or ("e6edge<B>&key" not in dot), (
            "dict key surfaced raw inside DOT"
        )


# --------------------------------------------------------------------------
# Choke-point unit pins (extends test_node_spec_api.py:196's single case).
# --------------------------------------------------------------------------


def test_render_lines_to_html_unit_corpus() -> None:
    """The composer escapes every transformable member and bolds row one."""

    lines = ["title & co", *TRANSFORMABLE_CORPUS, *ESCAPE_INVARIANT_CORPUS]
    html = render_lines_to_html(lines)
    assert html.startswith('<<TABLE BORDER="0"')
    assert html.endswith("</TABLE>>")
    assert "<B>title &amp; co</B>" in html
    for member in TRANSFORMABLE_CORPUS:
        assert _escaped(member) in html
        # Raw absence inside cell contents: strip the composer's own tags.
        cells = re.findall(r'<TD ALIGN="CENTER">(.*?)</TD>', html, flags=re.DOTALL)
        assert all(member not in cell for cell in cells), member
    for member in ESCAPE_INVARIANT_CORPUS:
        assert _escaped(member) == member
        assert member in html


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
