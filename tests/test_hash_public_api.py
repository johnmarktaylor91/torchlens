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
        capture=CaptureOptions(layers_to_save=None, save_source_context=True),
    )

    assert tl.hash.trace(all_layers) == tl.hash.trace(metadata_only) == tl.hash.trace(code_context)
    assert tl.hash.model(model, _input()) == tl.hash.trace(metadata_only)


def test_assert_unchanged_returns_hash_and_bootstraps() -> None:
    """Matching and bootstrap tripwire calls return the current hash.

    The bootstrap path discloses the fresh pin through the warning machinery
    (not an unconditional stdout write), so the hash is capturable and
    silenceable like any other TorchLens diagnostic.
    """

    from torchlens.errors import TorchLensWarning

    expected = tl.hash.model(_ResidualModel(), _input())
    assert tl.assert_unchanged(_ResidualModel(), _input(), expected) == expected

    with pytest.warns(TorchLensWarning, match="TorchLens structural hash") as record:
        bootstrapped = tl.assert_unchanged(_ResidualModel(), _input(), None)
    assert bootstrapped == expected
    assert any(bootstrapped in str(warning.message) for warning in record)


def test_assert_unchanged_reports_both_hashes_on_mismatch() -> None:
    """Mismatch errors name both the pin and the actual structural hash."""

    expected = tl.hash.model(_ResidualModel(), _input())
    actual = tl.hash.model(_SequentialModel(), _input())

    with pytest.raises(tl.hash.StructuralHashMismatchError) as error:
        tl.assert_unchanged(_SequentialModel(), _input(), expected)

    assert expected in str(error.value)
    assert actual in str(error.value)
    assert error.value.fields["code"] == "structural_hash_mismatch"
    assert error.value.fields["expected"] == expected
    assert error.value.fields["actual"] == actual
    assert isinstance(error.value.fields["remedy"], str) and error.value.fields["remedy"]


def test_structural_hash_mismatch_error_joins_the_taxonomy() -> None:
    """The mismatch class is registered and keeps AssertionError in the MRO."""

    from torchlens import errors

    assert errors.StructuralHashMismatchError is tl.hash.StructuralHashMismatchError
    assert issubclass(tl.hash.StructuralHashMismatchError, errors.ValidationError)
    assert issubclass(tl.hash.StructuralHashMismatchError, AssertionError)

    expected = tl.hash.model(_ResidualModel(), _input())
    with pytest.raises(AssertionError):
        tl.assert_unchanged(_SequentialModel(), _input(), expected)


def test_assert_unchanged_expected_type_door_is_typed() -> None:
    """A non-string, non-None pin refuses typed while staying a TypeError."""

    from torchlens import errors

    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        tl.assert_unchanged(_ResidualModel(), _input(), 123)  # type: ignore[arg-type]

    assert isinstance(exc_info.value, TypeError)
    assert exc_info.value.fields["code"] == "hash_expected_type_invalid"
    assert exc_info.value.fields["remedy"]


def test_content_hash_type_door_is_typed() -> None:
    """An unencodable input refuses typed while staying a TypeError."""

    from torchlens import errors

    class _Slotted:
        """Slots-only object with no inspectable ``__dict__``."""

        __slots__ = ("value",)

    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        tl.hash.content(_Slotted())

    assert isinstance(exc_info.value, TypeError)
    assert exc_info.value.fields["code"] == "hash_content_type_unsupported"
    assert "_Slotted" in exc_info.value.fields["value_type"]
    assert exc_info.value.fields["remedy"]
