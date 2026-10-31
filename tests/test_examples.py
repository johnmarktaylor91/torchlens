"""Executable documentation tests for intervention examples."""

from __future__ import annotations

import runpy
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "intervention"
EXAMPLE_FILES = tuple(sorted(EXAMPLE_DIR.glob("[0-9][0-9]_*.py")))

# The glob above is the sole discovery mechanism, so a deleted or renamed
# example would silently drop out of the parameterized run. Pin the known
# roster: shrinking it must be a deliberate edit here, never an accident.
EXPECTED_EXAMPLE_STEMS = frozenset(
    {
        "01_first_five_minutes",
        "02_exact_site_after_discovery",
        "03_activation_patching_paired_prompt",
        "04_sticky_hooks_multiple_engines",
        "05_set_vs_attach_hooks",
        "06_chunked_batching",
        "07_bundle_comparison",
        "08_live_post_hooks_during_capture",
        "09_submodule_discover_first",
        "10_post_hoc_replay_generation_trace",
        "11_sae_attachment",
        "12_linear_probe_attachment",
        "13_paired_prompt_3plus",
        "14_per_position_steering",
        "15_publishing_for_reproducibility",
        "17_raw_forward_hook_replacement",
    }
)


def _example_ids() -> Iterator[str]:
    """Yield stable pytest IDs for intervention example scripts.

    Yields
    ------
    str
        Example file stem.
    """

    for path in EXAMPLE_FILES:
        yield path.stem


def test_intervention_examples_discovered() -> None:
    """Fail if a known intervention example is no longer collected.

    Returns
    -------
    None
        Asserts every pinned example stem is discovered by the glob.
    """

    stems = {path.stem for path in EXAMPLE_FILES}
    missing = EXPECTED_EXAMPLE_STEMS - stems
    assert not missing, f"intervention examples missing from discovery: {sorted(missing)}"


@pytest.mark.slow
@pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=tuple(_example_ids()))
def test_intervention_example_runs(example_path: Path) -> None:
    """Import and run one intervention worked example.

    Parameters
    ----------
    example_path:
        Path to the example script.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        namespace = runpy.run_path(str(example_path))
        namespace["main"]()
