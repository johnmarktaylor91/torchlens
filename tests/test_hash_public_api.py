"""Tests for the provisional public structural-hash API."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _ResidualModel(nn.Module):
    """Small model with an explicit residual connection."""

    def __init__(self) -> None:
        """Initialize the residual layers."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the residual topology.

        Parameters
        ----------
        value:
            Input activation.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        return self.right(torch.relu(self.left(value))) + value


class _SequentialModel(nn.Module):
    """Similar model without the residual connection."""

    def __init__(self) -> None:
        """Initialize the sequential layers."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the non-residual topology.

        Parameters
        ----------
        value:
            Input activation.

        Returns
        -------
        torch.Tensor
            Sequential output.
        """

        return self.right(torch.relu(self.left(value)))


class _ExpandedModel(nn.Module):
    """Sequential model with one additional nonlinear layer."""

    def __init__(self) -> None:
        """Initialize the expanded layers."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.middle = nn.Sigmoid()
        self.right = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the expanded topology.

        Parameters
        ----------
        value:
            Input activation.

        Returns
        -------
        torch.Tensor
            Expanded sequential output.
        """

        return self.right(self.middle(torch.relu(self.left(value))))


class _BufferedModel(nn.Module):
    """Model whose buffer event guards metadata-only capture equivalence."""

    def __init__(self) -> None:
        """Initialize the buffer and affine layer."""

        super().__init__()
        self.register_buffer("offset", torch.arange(4, dtype=torch.float32))
        self.proj = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add the buffer before projection.

        Parameters
        ----------
        value:
            Input activation.

        Returns
        -------
        torch.Tensor
            Projected output.
        """

        return self.proj(value + self.offset)


def _input() -> torch.Tensor:
    """Return a stable example input.

    Returns
    -------
    torch.Tensor
        Example activation.
    """

    return torch.ones(2, 4)


def test_structural_hash_is_deterministic_across_initializations_and_processes() -> None:
    """Public model hashes ignore random parameter initialization across processes."""

    first = tl.hash.model(_ResidualModel(), _input())
    torch.manual_seed(100)
    second = tl.hash.model(_ResidualModel(), _input())
    script = textwrap.dedent(
        """
        import torch
        from torch import nn
        import torchlens as tl

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(4, 4)
                self.right = nn.Linear(4, 4)

            def forward(self, value):
                return self.right(torch.relu(self.left(value))) + value

        torch.manual_seed(999)
        print(tl.hash.model(Model(), torch.ones(2, 4)))
        """
    )
    third = subprocess.check_output([sys.executable, "-c", script], text=True).strip()

    assert first == second == third


def test_structural_hash_changes_with_graph_topology() -> None:
    """Adding a layer or dropping a structural edge changes the public hash."""

    residual_hash = tl.hash.model(_ResidualModel(), _input())
    sequential_hash = tl.hash.model(_SequentialModel(), _input())
    expanded_hash = tl.hash.model(_ExpandedModel(), _input())

    assert residual_hash != sequential_hash
    assert sequential_hash != expanded_hash


def test_structural_hash_is_capture_option_invariant_for_metadata_options() -> None:
    """Address-free hashes match across equivalent payload-retention choices."""

    model = _BufferedModel()
    all_layers = tl.trace(model, _input(), capture=CaptureOptions(layers_to_save="all"))
    metadata_only = tl.trace(model, _input(), capture=CaptureOptions(layers_to_save=None))
    code_context = tl.trace(
        model,
        _input(),
        capture=CaptureOptions(layers_to_save=None, save_code_context=True),
    )

    assert tl.hash.trace(all_layers) == tl.hash.trace(metadata_only) == tl.hash.trace(code_context)
    assert tl.hash.model(model, _input()) == tl.hash.trace(metadata_only)


def test_assert_unchanged_returns_hash_and_bootstraps(capsys: pytest.CaptureFixture[str]) -> None:
    """Matching and bootstrap tripwire calls return the current hash."""

    expected = tl.hash.model(_ResidualModel(), _input())
    assert tl.assert_unchanged(_ResidualModel(), _input(), expected) == expected

    bootstrapped = tl.assert_unchanged(_ResidualModel(), _input(), None)
    assert bootstrapped == expected
    assert bootstrapped in capsys.readouterr().out


def test_assert_unchanged_reports_both_hashes_on_mismatch() -> None:
    """Mismatch errors name both the pin and the actual structural hash."""

    expected = tl.hash.model(_ResidualModel(), _input())
    actual = tl.hash.model(_SequentialModel(), _input())

    with pytest.raises(tl.hash.StructuralHashMismatchError) as error:
        tl.assert_unchanged(_SequentialModel(), _input(), expected)

    assert expected in str(error.value)
    assert actual in str(error.value)
