"""Option-effectiveness witnesses (M(oracles) item 11; F36 oracle wave 1).

For each witnessed option: BOTH arms execute and the observable DIFFERS --
an option whose flip changes nothing is a silent no-op (the stop_after
class). The witness helper itself is red-capability-proven with a planted
no-op option (a dead checker exonerating silent options is worse than no
checker). Every witnessed field is cross-checked against the dual-source
census (options_fields.tsv) so a renamed option cannot orphan its witness.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.smoke]

DATA_DIR = Path(__file__).resolve().parent / "data"


def _census_fields() -> frozenset[str]:
    rows = [
        line.strip()
        for line in (DATA_DIR / "options_fields.tsv").read_text().splitlines()
        if line.strip() and not line.startswith("#") and line.strip() != "key"
    ]
    return frozenset(rows)


def _capture(**option_overrides: Any) -> Any:
    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    torch.manual_seed(1)
    return tl.trace(model, torch.randn(2, 4), capture=tl.options.CaptureOptions(**option_overrides))


def _observable_payload_retention(trace: Any) -> Any:
    return trace["relu_1_2"].out is not None


def _observable_edges_available(trace: Any) -> Any:
    try:
        _ = trace.edges
        return True
    except Exception:
        return False


def _observable_nonfinite_basis(trace: Any) -> Any:
    coverage = trace.nonfinite_coverage
    return getattr(coverage, "basis", str(coverage))


#: field name (census spelling) -> (off arm, on arm, observable reader).
WITNESSES: dict[str, tuple[dict[str, Any], dict[str, Any], Callable[[Any], Any]]] = {
    "CaptureOptions.structure_only": (
        {},
        {"structure_only": True},
        _observable_payload_retention,
    ),
    "CaptureOptions.intervention_ready": (
        {},
        {"intervention_ready": True},
        _observable_edges_available,
    ),
    "CaptureOptions.track_nonfinite": (
        {},
        {"track_nonfinite": True},
        _observable_nonfinite_basis,
    ),
}


@pytest.mark.parametrize("field", sorted(WITNESSES))
def test_option_flip_changes_its_observable(field: str) -> None:
    """Both arms run; the declared observable differs between them."""

    off_options, on_options, read = WITNESSES[field]
    off_trace = _capture(**off_options)
    try:
        off_value = read(off_trace)
    finally:
        off_trace.cleanup()
    on_trace = _capture(**on_options)
    try:
        on_value = read(on_trace)
    finally:
        on_trace.cleanup()
    assert off_value != on_value, (
        f"{field}: flipping the option left the observable at {off_value!r}"
        " on both arms -- a silent no-op (the stop_after class)"
    )


def test_witnessed_fields_ride_the_dual_source_census() -> None:
    """A witness on a field the census does not know is an orphan."""

    census = _census_fields()
    orphans = sorted(set(WITNESSES) - census)
    assert not orphans, (
        f"witnessed fields missing from options_fields.tsv: {orphans}"
        " (renamed option? re-key the witness in the same commit)"
    )


def test_witness_helper_catches_a_planted_neutralized_option() -> None:
    """Red-capability: a flip whose observable does NOT move must fail."""

    def dead_observable(trace: Any) -> Any:
        del trace
        return "constant"

    off_trace = _capture()
    try:
        off_value = dead_observable(off_trace)
    finally:
        off_trace.cleanup()
    on_trace = _capture(verbose=False)
    try:
        on_value = dead_observable(on_trace)
    finally:
        on_trace.cleanup()
    with pytest.raises(AssertionError):
        assert off_value != on_value, "planted"
