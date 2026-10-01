"""Draw-option combination TOTALITY test (viz-correctness lane, 2026-08-19).

The hardening campaign was category-driven and per-file, so nobody exercised
the CROSS-PRODUCT of user-facing ``draw()`` options: ``node_mode="profiling"``
worked, ``vis_call_depth=1`` worked, and together they crashed on any
collapsed module. This suite closes that class the same way
``test_marker_combination_totality.py`` closes the marker table:

1. The full ``Trace.draw`` signature is transcribed into ``DECLARED`` — a
   closed inventory in which every parameter is classified. A new draw option
   landing without a declaration here fails ``test_draw_signature_is_fully_
   declared`` loudly, forcing its combination behavior to be decided (and
   exercised) in the same change.
2. Every PAIRED option is exercised alone and in every unordered pair, on a
   module-bearing multi-pass trace. Each combination must either succeed or
   raise exactly the typed refusal the pinned ``REFUSING_COMBOS`` table
   declares — any other exception is a V1-class combination defect.

The refusal table is EXACT and pinned as a literal: adding a refusal (or
legalizing one) is a reviewed diff here, never a silent legality change.

This suite owns crash/refusal totality; warning hygiene is owned by the
dedicated deprecation and advisory suites, so draws run with warnings
visible-but-not-fatal.

Tiering (the smoke family-aggregate budget rules the full sweep out of the
5s tier): the signature tripwire and the pinned refusal rows run at smoke on
every commit; the CORE pair family moved to ``heavy`` when the merged tree's
per-cell draw cost outgrew the cell-scaled smoke budget (T67d), joining the
exhaustive singles + all-pairs sweep in the ``not slow`` tiers.
"""

from __future__ import annotations

import inspect
import itertools
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError

# ---------------------------------------------------------------------------
# The declared option inventory. Classification vocabulary:
#
# PAIRED       exercised alone and pairwise with the probe value given.
# HARNESS      supplied on EVERY draw by this suite (output sink plumbing);
#              exercising it pairwise would be self-pairing.
# ALIAS        deprecated or legacy alias sentinel for a canonical option;
#              pairing two spellings of one canonical is incoherent, and the
#              alias hop itself is asserted by the dedicated deprecation
#              suites (test_api_renames / test_deprecation_inventory).
# DEFAULT_ONLY declared but exercised at its default: the non-default value
#              needs an optional runtime this suite must not depend on.
# ---------------------------------------------------------------------------

PAIRED = "paired"
HARNESS = "harness"
ALIAS = "alias"
DEFAULT_ONLY = "default_only"


def _noop_node_spec_fn(layer_log: Any, spec: Any) -> None:
    """No-op user node callback (returning None keeps the mode spec)."""

    del layer_log, spec
    return None


def _noop_collapsed_spec_fn(module_log: Any, spec: Any) -> None:
    """No-op collapsed-module callback."""

    del module_log, spec
    return None


def _never_collapse(module_log: Any) -> bool:
    """collapse_fn probe: never force-collapse a module."""

    del module_log
    return False


def _never_skip(layer_log: Any) -> bool:
    """skip_fn probe: never skip a layer."""

    del layer_log
    return False


def _unit_overlay(layer_log: Any) -> float:
    """node_overlay probe: constant overlay score."""

    del layer_log
    return 1.0


#: option -> (classification, probe value or reason string)
DECLARED: dict[str, tuple[str, Any]] = {
    # -- harness plumbing (supplied on every draw below) --------------------
    "vis_outpath": (HARNESS, "output sink; a fresh tmp path every draw"),
    "vis_save_only": (HARNESS, "True everywhere: no display side effects"),
    "vis_fileformat": (HARNESS, '"dot" everywhere: no graphviz binary cost'),
    # -- alias sentinels (dedicated deprecation suites own the hop) ---------
    "view": (ALIAS, "vis_mode"),
    "depth": (ALIAS, "vis_call_depth"),
    "renderer": (ALIAS, "vis_renderer"),
    "layout": (ALIAS, "vis_node_placement"),
    "node_style": (ALIAS, "node_mode"),
    # -- default-only ---------------------------------------------------------
    "vis_renderer": (DEFAULT_ONLY, 'non-default "dagua" needs the optional dagua runtime'),
    # -- paired semantic options -------------------------------------------
    "vis_mode": (PAIRED, "rolled"),
    "vis_call_depth": (PAIRED, 1),
    "vis_graph_overrides": (PAIRED, {"bgcolor": "white"}),
    "module": (PAIRED, "block"),
    "node_mode": (PAIRED, "profiling"),
    "node_spec_fn": (PAIRED, _noop_node_spec_fn),
    "collapsed_node_spec_fn": (PAIRED, _noop_collapsed_spec_fn),
    "collapse_fn": (PAIRED, _never_collapse),
    "collapse": (PAIRED, "max"),
    "fold_repeats": (PAIRED, True),
    # F11 pattern folding (memo D11): pattern-only view is the v1 contract;
    # the collapse pairing refuses typed (REFUSING_COMBOS row below).
    "fold_patterns": (PAIRED, {"LinearOnly": "linear"}),
    "skip_fn": (PAIRED, _never_skip),
    "vis_edge_overrides": (PAIRED, {"color": "black"}),
    "vis_grad_edge_overrides": (PAIRED, {"color": "black"}),
    "vis_module_overrides": (PAIRED, {"penwidth": "2"}),
    "show_buffer_layers": (PAIRED, "always"),
    "direction": (PAIRED, "topdown"),
    "vis_node_placement": (PAIRED, "rank"),
    "vis_theme": (PAIRED, "paper"),
    "vis_intervention_mode": (PAIRED, "as_node"),
    "vis_show_cone": (PAIRED, False),
    "code_panel": (PAIRED, True),
    "node_overlay": (PAIRED, _unit_overlay),
    "node_label_fields": (PAIRED, ["label", "time"]),
    "show_legend": (PAIRED, True),
    "font_size": (PAIRED, 10),
    "dpi": (PAIRED, 100),
    "for_paper": (PAIRED, True),
    "return_graph": (PAIRED, True),
    "order_siblings": (PAIRED, False),
    "show_containers": (PAIRED, "labels"),
    "container_max_inline": (PAIRED, 2),
    "show_input_transform_summary": (PAIRED, True),
    "show_orphans": (PAIRED, True),
    # Added by the small-builds lane (saved-for-backward viz annotation): marks which
    # tensors autograd retained, so memory behaviour is visible rather than guessed.
    # Caught landing UNDECLARED by this suite's own signature tripwire -- the first
    # cross-lane catch by a totality test built in the same sprint.
    "show_saved_for_backward": (PAIRED, True),
    "color_by": (PAIRED, "time"),
    "size_by": (PAIRED, "dims"),
    "scale": (PAIRED, "linear"),
    "stack_by": (PAIRED, True),
    "show_redundant_args": (PAIRED, True),
}

PAIRED_OPTIONS: tuple[str, ...] = tuple(
    name for name, (kind, _) in DECLARED.items() if kind == PAIRED
)

#: The exact refusing combinations among the probes above, pinned as
#: (frozen option set) -> expected typed code. ``"scale"`` refuses with EVERY
#: partner except ``size_by`` (its own documented dependency), so it is
#: expressed as a rule in ``expected_refusal`` rather than 37 literal rows.
REFUSING_COMBOS: dict[frozenset[str], str] = {
    frozenset({"collapse", "fold_patterns"}): "pattern_collapse_combination_unsupported",
    frozenset({"fold_patterns", "vis_node_placement"}): "pattern_rank_layout_unsupported",
    frozenset({"vis_node_placement", "color_by"}): "encoding_requires_dot_layout",
    frozenset({"vis_node_placement", "size_by"}): "encoding_requires_dot_layout",
    frozenset({"vis_node_placement", "stack_by"}): "encoding_requires_dot_layout",
    frozenset({"vis_mode", "stack_by"}): "stack_by_requires_unrolled",
}


def expected_refusal(combo: frozenset[str]) -> str | None:
    """Return the typed refusal code for a probe combination, else ``None``.

    ``scale`` without ``size_by`` refuses first (the option is meaningless
    alone), matching shipped precedence; the pinned pair rows cover the rest.
    """

    if "scale" in combo and "size_by" not in combo:
        return "scale_requires_size_by"
    return REFUSING_COMBOS.get(combo)


# ---------------------------------------------------------------------------
# Signature totality: a new draw() option must be declared here.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_draw_signature_is_fully_declared() -> None:
    """Every ``Trace.draw`` parameter is classified in ``DECLARED``.

    A new option landing without a declaration is exactly how the
    profiling x vis_call_depth crash shipped: each option tested alone,
    the combination never exercised. Declare the newcomer (probe +
    refusal rows if any) in the same change that adds it.
    """

    signature = inspect.signature(tl.Trace.draw)
    actual = {name for name in signature.parameters if name != "self"}
    declared = set(DECLARED)
    undeclared = actual - declared
    stale = declared - actual
    assert not undeclared, (
        f"draw() options landed undeclared: {sorted(undeclared)}. Add each to "
        "DECLARED in this file (classification + probe) so its combination "
        "behavior is exercised, and add REFUSING_COMBOS rows for any typed "
        "refusals it introduces."
    )
    assert not stale, (
        f"DECLARED lists options draw() no longer accepts: {sorted(stale)}. "
        "Remove them (and their refusal rows) in the same change."
    )


@pytest.mark.smoke
def test_refusal_table_only_names_declared_paired_options() -> None:
    """Every refusal row references PAIRED options actually being exercised."""

    paired = set(PAIRED_OPTIONS)
    for combo in REFUSING_COMBOS:
        assert combo <= paired, f"refusal row names non-paired options: {sorted(combo)}"


# ---------------------------------------------------------------------------
# Behavioral sweep: singles + every unordered pair.
# ---------------------------------------------------------------------------


class _RecurrentWrapper(nn.Module):
    """Module-bearing multi-pass model: exercises collapse, folds, and passes."""

    def __init__(self) -> None:
        """Initialize the repeated block."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block twice (multi-pass layers, monotone lockstep)."""

        for _ in range(2):
            x = self.block(x)
        return x


@pytest.fixture(scope="module")
def sweep_trace() -> Iterator[tl.Trace]:
    """One shared trace for the whole sweep, released at module teardown."""

    trace = tl.trace(_RecurrentWrapper(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _draw(trace: Any, tmp_path: Path, kwargs: dict[str, Any]) -> None:
    """Draw with the harness base; warnings visible but never fatal here."""

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        trace.draw(
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(tmp_path / "graph"),
            **kwargs,
        )


_SINGLES = [(name,) for name in PAIRED_OPTIONS]
_PAIRS = list(itertools.combinations(PAIRED_OPTIONS, 2))

#: Structural axes that reshape the rendered graph — the axes whose
#: combinations have historically crashed (profiling x vis_call_depth,
#: rolled x show_containers). Their pairs run at SMOKE on every commit;
#: the exhaustive sweep below covers everything at the heavy tier (the
#: smoke family-aggregate budget rules 741 draw cells out of the 5s tier).
CORE_OPTIONS: tuple[str, ...] = (
    "vis_mode",
    "vis_call_depth",
    "node_mode",
    "collapse",
    "fold_repeats",
    "show_containers",
    "module",
    "color_by",
)
_CORE_PAIRS = list(itertools.combinations(CORE_OPTIONS, 2))

#: Every pinned-refusal combination, provoked at smoke (each raises before
#: rendering, so the family is cheap).
_REFUSAL_COMBOS: list[tuple[str, ...]] = [("scale",)] + [
    tuple(sorted(combo)) for combo in sorted(REFUSING_COMBOS, key=sorted)
]


def _exercise_combo(combo: tuple[str, ...], trace: Any, tmp_path: Path) -> None:
    """Draw one option combination; assert its pinned verdict."""

    kwargs = {name: DECLARED[name][1] for name in combo}
    code = expected_refusal(frozenset(combo))
    if code is None:
        _draw(trace, tmp_path, kwargs)
        return
    with pytest.raises(InvalidArgumentError) as excinfo:
        _draw(trace, tmp_path, kwargs)
    assert excinfo.value.fields["code"] == code


@pytest.mark.heavy
@pytest.mark.parametrize("combo", _CORE_PAIRS, ids=["+".join(c) for c in _CORE_PAIRS])
def test_draw_core_option_pairs(combo: tuple[str, ...], sweep_trace: Any, tmp_path: Path) -> None:
    """High-risk structural option pairs stay legal (or refuse typed) at commit time."""

    _exercise_combo(combo, sweep_trace, tmp_path)


@pytest.mark.smoke
@pytest.mark.parametrize("combo", _REFUSAL_COMBOS, ids=["+".join(c) for c in _REFUSAL_COMBOS])
def test_draw_pinned_refusals(combo: tuple[str, ...], sweep_trace: Any, tmp_path: Path) -> None:
    """Every pinned refusing combination raises exactly its declared code."""

    assert expected_refusal(frozenset(combo)) is not None
    _exercise_combo(combo, sweep_trace, tmp_path)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "combo", _SINGLES + _PAIRS, ids=["+".join(combo) for combo in _SINGLES + _PAIRS]
)
def test_draw_option_combination(combo: tuple[str, ...], sweep_trace: Any, tmp_path: Path) -> None:
    """Exhaustive sweep: each declared combination succeeds or refuses typed."""

    _exercise_combo(combo, sweep_trace, tmp_path)
