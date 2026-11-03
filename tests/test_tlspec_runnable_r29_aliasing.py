"""H3 input-aliasing honesty, both directions (r29-C3).

Torch capture de-aliases model inputs, so an aliased-input runtime call that mutates a model
input is unreproducible against the recorded de-aliased path. Two r29 gaps:

* F2-hon (GAP): a VIEW-mediated input mutation (``a[0].add_(100)`` with ``a is b``) targets the
  getitem view slot, whose ``version_of`` never links back to the input, so the fail-closed
  gate missed it -> false VERIFIED. The gate now follows view-op lineage from the input slots.
* codex-F4 (OVER-TRIGGER): two DISJOINT views of one base (``base[:2]`` / ``base[2:]``) share a
  base storage pointer but touch non-overlapping bytes, so a mutation of one cannot reach the
  other; keying aliasing on the base pointer falsely diverged them. Aliasing is now the OVERLAP
  of storage byte spans.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions


def _save(model: nn.Module, args: object, path: Path) -> Path:
    """Capture and save a runnable artifact."""

    trace = tl.trace(
        model,
        args,
        capture=CaptureOptions(replayable=True, capture_container_structure=True, cache=False),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


class ViewMutateModel(nn.Module):
    """Mutate a VIEW of one input in place, then read another input site."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """In-place add on the ``a[0]`` view; a fresh aliased model sees it through ``b``."""

        a[0].add_(100.0)
        return a + b


class DirectMutateModel(nn.Module):
    """Mutate one whole input in place, then read another input site."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """In-place add on ``a``."""

        a.add_(1.0)
        return a + b


# Kept: sole branch coverage of _runnable/path_faithfulness.py, _runnable/verification.py.
def test_r29_disjoint_views_of_one_base_do_not_over_trigger(tmp_path: Path) -> None:
    """Two disjoint views of the same base storage must NOT falsely diverge (codex-F4)."""

    base = torch.randn(8)
    path = _save(DirectMutateModel(), (base[:4].clone(), base[4:].clone()), tmp_path / "dm.tlspec")

    big = torch.randn(8)
    va, vb = big[:4], big[4:]  # share base storage, disjoint byte spans
    assert va.untyped_storage().data_ptr() == vb.untyped_storage().data_ptr()
    result = tl.load(path).run(inputs=(va, vb))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ---------------------------------------------------------------------------
# r35 decision D (corr2_1): three-valued touched-byte alias engine.
# ---------------------------------------------------------------------------


class AddInputsModel(nn.Module):
    """Pure read-only combination of two input sites."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def _run_pair(tmp_path: Path, name: str, a: torch.Tensor, b: torch.Tensor):
    """Capture on clones, then run on the given (possibly aliased) pair."""

    path = _save(AddInputsModel(), (a.detach().clone(), b.detach().clone()), tmp_path / name)
    return tl.load(path).run(inputs=(a, b))


# Kept: sole branch coverage of _runnable/input_aliases.py, _utils/alias_footprint.py.
def test_r35_broadcast_zero_stride_overlap_fails_closed(tmp_path: Path) -> None:
    """A zero-stride broadcast view over a consumed element -> fail closed."""

    base = torch.randn(4)
    expanded = base[:1].expand(4)  # touches exactly element 0, stride 0
    with pytest.raises(PathDivergenceError):
        _run_pair(tmp_path, "broadcast.tlspec", expanded, base)


# Kept: sole branch coverage of _runnable/input_aliases.py.
def test_r35_same_object_identity_fails_closed(tmp_path: Path) -> None:
    """``forward(a, b)`` with ``a is b`` stays an observed contradiction."""

    shared = torch.randn(4)
    with pytest.raises(PathDivergenceError):
        _run_pair(tmp_path, "identity.tlspec", shared, shared)


# Kept: sole branch coverage of _runnable/path_faithfulness.py.
def test_r35_unprovable_topology_is_unverifiable_never_attested(tmp_path: Path) -> None:
    """Above the cap with no residue proof: unknown -> UNVERIFIABLE, never DIVERGED."""

    # Same storage, both > 65536 elements, congruent starts and gcd-compatible
    # strides -> the residue proof cannot separate them, enumeration is over the
    # cap, and the bounding intervals overlap: verdict must be UNKNOWN.
    base = torch.randn(3 * 70000 + 4)
    a = base[0::3][:69000]
    b = base[3::3][:69000]
    result = _run_pair(tmp_path, "unknown.tlspec", a, b)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# Kept: sole branch coverage of _runnable/input_metadata.py, _utils/alias_footprint.py.
def test_r35_engine_unit_matrix() -> None:
    """Unit rows for the three-valued engine (proof, not assumption)."""

    from torchlens._runnable.input_metadata import _touched_bytes_relation

    base = torch.randn(16)
    # Identity geometry.
    assert _touched_bytes_relation(base[2:6], base[2:6]) == "overlap"
    # Distinct storages.
    assert _touched_bytes_relation(torch.randn(4), torch.randn(4)) == "disjoint"
    # Interleaves.
    assert _touched_bytes_relation(base[::2], base[1::2]) == "disjoint"
    # Genuine overlap via enumeration.
    assert _touched_bytes_relation(base[:3], base[2:]) == "overlap"
    # Transposed 2-D tiles, disjoint.
    grid = torch.randn(4, 4)
    assert _touched_bytes_relation(grid[0:2, 0:2].t(), grid[2:4, 2:4]) == "disjoint"
    # Mixed element sizes over one storage: byte-level enumeration decides.
    stor = torch.zeros(8, dtype=torch.float32)
    as_half = stor.view(torch.float16)  # 16 half elements over the same bytes
    assert _touched_bytes_relation(stor[:2], as_half[0:4]) == "overlap"
    assert _touched_bytes_relation(stor[2:4], as_half[0:4]) == "disjoint"
