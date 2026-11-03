"""Draw-option combination TOTALITY test (viz-correctness lane, 2026-08-19).

The hardening campaign was category-driven and per-file, so nobody exercised
the CROSS-PRODUCT of user-facing ``draw()`` options: ``node_style="profiling"``
worked, ``depth=1`` worked, and together they crashed on any
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

Sampling (test-shape ruling tests.4): the parametrized cells are the
documented combinations: every pinned refusal, every option alone, the CORE
structural pairs whose combinations have historically crashed, and the pairs
that alone reach a library branch (``_COVERAGE_PAIRS``). One property test
walks the whole generator (every single and every unordered pair, about 260
seconds of draws, so ``slow``) and names every combination that neither
draws nor refuses with its pinned code.
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
#              suites (scripts/checks/deprecation_inventory.py).
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
    "path": (HARNESS, "output sink; a fresh tmp path every draw"),
    "open_viewer": (HARNESS, "False everywhere (the default): no display side effects"),
    "file_format": (HARNESS, '"dot" everywhere: no graphviz binary cost'),
    # -- default-only ---------------------------------------------------------
    "renderer": (DEFAULT_ONLY, 'non-default "dagua" needs the optional dagua runtime'),
    "graph": (DEFAULT_ONLY, "backward and combined graphs need a captured backward pass"),
    "bwd": (DEFAULT_ONLY, "backward and combined graphs only"),
    "show_module_paths": (DEFAULT_ONLY, "backward graph only"),
    "show_grad_values": (DEFAULT_ONLY, "backward graph only"),
    "backward_node_spec_fn": (DEFAULT_ONLY, "combined graph only"),
    "intervening_cluster": (DEFAULT_ONLY, "combined graph only"),
    # -- paired semantic options -------------------------------------------
    "rolled": (PAIRED, True),
    "depth": (PAIRED, 1),
    "graph_overrides": (PAIRED, {"bgcolor": "white"}),
    "module": (PAIRED, "block"),
    "node_style": (PAIRED, "profiling"),
    "node_spec_fn": (PAIRED, _noop_node_spec_fn),
    "collapsed_node_spec_fn": (PAIRED, _noop_collapsed_spec_fn),
    "collapse_fn": (PAIRED, _never_collapse),
    "collapse": (PAIRED, "max"),
    "fold_repeats": (PAIRED, True),
    # F11 pattern folding (memo D11): pattern-only view is the v1 contract;
    # the collapse pairing refuses typed (REFUSING_COMBOS row below).
    "fold_patterns": (PAIRED, {"LinearOnly": "linear"}),
    "skip_fn": (PAIRED, _never_skip),
    "edge_overrides": (PAIRED, {"color": "black"}),
    "grad_edge_overrides": (PAIRED, {"color": "black"}),
    "module_overrides": (PAIRED, {"penwidth": "2"}),
    "show_buffers": (PAIRED, "always"),
    "direction": (PAIRED, "topdown"),
    "layout": (PAIRED, "rank"),
    "theme": (PAIRED, "paper"),
    "intervention_mode": (PAIRED, "as_node"),
    "show_cone": (PAIRED, True),
    "code_panel": (PAIRED, True),
    "node_overlay": (PAIRED, _unit_overlay),
    "node_label_fields": (PAIRED, ["label", "time"]),
    "show_legend": (PAIRED, True),
    "font_size": (PAIRED, 10),
    "dpi": (PAIRED, 100),
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
    "size_scale": (PAIRED, "linear"),
    "stack_by": (PAIRED, True),
    "show_redundant_args": (PAIRED, True),
}

PAIRED_OPTIONS: tuple[str, ...] = tuple(
    name for name, (kind, _) in DECLARED.items() if kind == PAIRED
)

#: The exact refusing combinations among the probes above, pinned as
#: (frozen option set) -> expected typed code. ``"size_scale"`` refuses with EVERY
#: partner except ``size_by`` (its own documented dependency), so it is
#: expressed as a rule in ``expected_refusal`` rather than 37 literal rows.
REFUSING_COMBOS: dict[frozenset[str], str] = {
    frozenset({"collapse", "fold_patterns"}): "pattern_collapse_combination_unsupported",
    frozenset({"fold_patterns", "layout"}): "pattern_rank_layout_unsupported",
    frozenset({"layout", "color_by"}): "encoding_requires_dot_layout",
    frozenset({"layout", "size_by"}): "encoding_requires_dot_layout",
    frozenset({"layout", "stack_by"}): "encoding_requires_dot_layout",
    frozenset({"rolled", "stack_by"}): "stack_by_requires_unrolled",
}


def expected_refusal(combo: frozenset[str]) -> str | None:
    """Return the typed refusal code for a probe combination, else ``None``.

    ``scale`` without ``size_by`` refuses first (the option is meaningless
    alone), matching shipped precedence; the pinned pair rows cover the rest.
    """

    if "size_scale" in combo and "size_by" not in combo:
        return "scale_requires_size_by"
    return REFUSING_COMBOS.get(combo)


# ---------------------------------------------------------------------------
# Signature totality: a new draw() option must be declared here.
# ---------------------------------------------------------------------------


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
            file_format="dot",
            path=str(tmp_path / "graph"),
            **kwargs,
        )


_SINGLES = [(name,) for name in PAIRED_OPTIONS]
_PAIRS = list(itertools.combinations(PAIRED_OPTIONS, 2))

#: Structural axes that reshape the rendered graph: the axes whose
#: combinations have historically crashed (profiling x vis_call_depth,
#: rolled x show_containers). Every pair among them is a documented cell; the
#: slow property test below walks every other pair.
CORE_OPTIONS: tuple[str, ...] = (
    "rolled",
    "depth",
    "node_style",
    "collapse",
    "fold_repeats",
    "show_containers",
    "module",
    "color_by",
)
_CORE_PAIRS = list(itertools.combinations(CORE_OPTIONS, 2))

#: Every pinned-refusal combination, provoked at smoke (each raises before
#: rendering, so the family is cheap).
_REFUSAL_COMBOS: list[tuple[str, ...]] = [("size_scale",)] + [
    tuple(sorted(combo)) for combo in sorted(REFUSING_COMBOS, key=sorted)
]


#: Pairs outside the CORE family whose cells alone reached a library branch
#: in the full sweep (per-test branch coverage, test-shape lane evidence).
_COVERAGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("vis_node_placement", "code_panel"),
    ("vis_mode", "fold_patterns"),
    ("vis_call_depth", "show_legend"),
    ("collapse_fn", "fold_repeats"),
    ("vis_mode", "size_by"),
    ("collapse_fn", "collapse"),
    ("module", "vis_node_placement"),
    ("vis_mode", "node_label_fields"),
    ("vis_graph_overrides", "vis_node_placement"),
    ("size_by", "scale"),
    ("vis_node_placement", "show_orphans"),
    ("vis_mode", "collapse"),
    ("vis_call_depth", "fold_patterns"),
    ("vis_call_depth", "collapsed_node_spec_fn"),
    ("node_spec_fn", "color_by"),
    ("vis_node_placement", "show_legend"),
    ("vis_mode", "vis_node_placement"),
    ("vis_mode", "show_saved_for_backward"),
    ("vis_graph_overrides", "vis_intervention_mode"),
    ("vis_call_depth", "stack_by"),
    ("module", "skip_fn"),
    ("module", "fold_patterns"),
    ("vis_mode", "show_buffers"),
)


def _combo_problem(combo: tuple[str, ...], trace: Any, tmp_path: Path) -> str | None:
    """How one option combination breaks its pinned verdict, else ``None``."""

    kwargs = {name: DECLARED[name][1] for name in combo}
    code = expected_refusal(frozenset(combo))
    try:
        _draw(trace, tmp_path, kwargs)
    except InvalidArgumentError as error:
        got = error.fields.get("code")
        if got == code:
            return None
        return f"refused with {got!r}, expected {code or 'a successful draw'}"
    except Exception as error:  # noqa: BLE001 -- any other exception IS the defect
        return f"{type(error).__name__}: {error}"
    if code is not None:
        return f"drew, expected the typed refusal {code!r}"
    return None


def _exercise_combo(combo: tuple[str, ...], trace: Any, tmp_path: Path) -> None:
    """Draw one option combination; assert its pinned verdict."""

    problem = _combo_problem(combo, trace, tmp_path)
    assert problem is None, ("+".join(combo), problem)


@pytest.mark.parametrize("combo", _REFUSAL_COMBOS, ids=["+".join(c) for c in _REFUSAL_COMBOS])
def test_draw_pinned_refusals(combo: tuple[str, ...], sweep_trace: Any, tmp_path: Path) -> None:
    """Every pinned refusing combination raises exactly its declared code."""

    assert expected_refusal(frozenset(combo)) is not None
    _exercise_combo(combo, sweep_trace, tmp_path)


_SAMPLED = _SINGLES + _CORE_PAIRS + [pair for pair in _COVERAGE_PAIRS if pair not in _CORE_PAIRS]


@pytest.mark.heavy
@pytest.mark.parametrize("combo", _SAMPLED, ids=["+".join(combo) for combo in _SAMPLED])
def test_draw_option_combination(combo: tuple[str, ...], sweep_trace: Any, tmp_path: Path) -> None:
    """Documented combination: each option alone and each structural pair draws or refuses typed."""

    _exercise_combo(combo, sweep_trace, tmp_path)


@pytest.mark.slow
def test_every_declared_option_combination_draws_or_refuses_typed(
    sweep_trace: Any, tmp_path: Path
) -> None:
    """Property: every single and every unordered pair draws, or refuses with its pinned code."""

    broken = {
        "+".join(combo): problem
        for combo in _SINGLES + _PAIRS
        if (problem := _combo_problem(combo, sweep_trace, tmp_path)) is not None
    }
    assert not broken, broken
