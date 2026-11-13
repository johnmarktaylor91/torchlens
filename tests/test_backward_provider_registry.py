"""BackwardPass reads its torch-only providers through a leaf registry.

``_data_classes/backward_pass.py`` must not import the torch backward census or
the report attribution module (that tied every backward module into the largest
import cycle); the providers register themselves when their modules import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torchlens
from torchlens._data_classes import _backward_providers

_BACKWARD_PASS = Path(torchlens.__file__).parent / "_data_classes" / "backward_pass.py"


def test_backward_pass_imports_no_provider_module() -> None:
    """No import of the census or attribution module, at module top or in a function."""

    tree = ast.parse(_BACKWARD_PASS.read_text(encoding="utf-8"))
    imported = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.endswith("_backward_dispatch") for name in imported)
    assert not any(name.endswith("_backward_attribution") for name in imported)


def test_the_torch_backward_capture_fills_every_provider() -> None:
    """Importing the torch backward capture registers all four providers."""

    import torchlens.backends.torch.backward  # noqa: F401

    for name in (
        "recorded_ops",
        "attribute_census",
        "executed_flops_record",
        "attribution_from_record",
    ):
        assert callable(_backward_providers.backward_provider(name))


def test_the_loader_record_check_fills_the_attribution_providers() -> None:
    """The load-time executed-FLOPs record check imports the attribution module."""

    from torchlens._io import _load_validators
    from torchlens.report import _backward_attribution

    source = Path(_load_validators.__file__).read_text(encoding="utf-8")
    assert "from ..report._backward_attribution import" in source
    assert _backward_providers.backward_provider("attribution_from_record") is (
        _backward_attribution.attribution_from_record
    )
