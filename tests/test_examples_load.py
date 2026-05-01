"""Tests for the minimal TorchLens examples loader."""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens.intervention.types import InterventionSpec


def test_examples_load_returns_model_log() -> None:
    """torchlens.examples.load should return a supported TorchLens artifact."""

    artifact = tl.examples.load("name")

    assert isinstance(artifact, (tl.ModelLog, tl.Bundle, InterventionSpec))


def test_examples_load_unknown_name_errors() -> None:
    """Unknown example names should fail clearly."""

    with pytest.raises(KeyError, match="Unknown TorchLens example"):
        tl.examples.load("missing-example")
