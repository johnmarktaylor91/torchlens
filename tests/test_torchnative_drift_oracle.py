"""``torch.overrides`` drift oracle (torchnative 8b / W2.8).

TorchLens is BUILT on the ``__torch_function__`` protocol, so torch's own
overridable-function registry is a free drift oracle: a torch upgrade
cannot silently add protocol-visible functions TorchLens does not see. The
frozen baseline (``tests/data/torchnative/overrides_drift_baseline.json``)
names every overridable the wrap inventory deliberately does not cover on
the frozen torch minor; a NEW unseen name fails red with a teaching message
until it is consciously classified (wrap it, exclude it with a reason, or
extend the baseline in this lane's owner review).

The registry count is a DOCS NUMBER, never a coverage claim -- coverage
language requires executed probes, row by row (memo 8b).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.overrides import get_overridable_functions

from torchlens.constants import get_orig_torch_funcs

_BASELINE_PATH = Path(__file__).parent / "data" / "torchnative" / "overrides_drift_baseline.json"


def _overridable_names() -> set[str]:
    """Return normalized dotted names for torch's overridable functions.

    Property-object namespaces (Tensor attribute accessors) are EXCLUDED:
    their reprs carry memory addresses (unstable across runs) and they are
    attribute plumbing, not callable interposition surface. The exclusion
    is part of the oracle's declared scope.
    """

    names: set[str] = set()
    for namespace, functions in get_overridable_functions().items():
        if namespace is torch.Tensor:
            prefix = "torch.Tensor"
        else:
            module_name = getattr(namespace, "__name__", None)
            if not isinstance(module_name, str):
                continue  # property-object namespaces: declared exclusion
            prefix = module_name if module_name.startswith("torch") else f"torch.{module_name}"
        for function in functions:
            function_name = getattr(function, "__name__", None)
            if isinstance(function_name, str):
                names.add(f"{prefix}.{function_name}")
    return names


def _wrap_inventory() -> set[str]:
    """Return TorchLens's decorated-callable inventory as dotted names."""

    return {f"{namespace}.{attr}" for namespace, attr in get_orig_torch_funcs()}


def test_no_silently_unseen_overridables() -> None:
    """A torch upgrade cannot add overridable functions we do not see."""

    baseline = set(json.loads(_BASELINE_PATH.read_text())["unseen"])
    unseen = _overridable_names() - _wrap_inventory()
    new_names = unseen - baseline
    assert not new_names, (
        f"torch {torch.__version__} exposes {len(new_names)} overridable "
        f"function(s) TorchLens's wrap inventory does not cover and the "
        f"drift baseline does not name: {sorted(new_names)[:20]}... "
        "Classify each consciously: wrap it, exclude it with a reason, or "
        "extend tests/data/torchnative/overrides_drift_baseline.json in the "
        "torchnative owner review. Never silently regenerate the baseline."
    )


def test_interposition_surface_docs_number() -> None:
    """The docs number stays truthful: derived, never typed.

    ``docs/native-torch.md`` quotes the interposition-surface count from
    this derivation; the tolerance band keeps the docs sentence honest
    across patch releases without demanding lockstep edits for tiny drift.
    """

    overridable = _overridable_names()
    covered = overridable & _wrap_inventory()
    fraction = len(covered) / len(overridable)
    # On the frozen minor the measured point was ~0.85 covered; the docs
    # sentence says "the great majority", which this band keeps true.
    assert fraction > 0.75, (
        f"only {fraction:.0%} of torch's overridable registry intersects the "
        "wrap inventory; the docs wording and the baseline need a conscious "
        "re-derivation"
    )


def test_baseline_is_selfdescribing() -> None:
    """The frozen baseline carries its scope and version stamp."""

    payload = json.loads(_BASELINE_PATH.read_text())
    assert payload["schema"] == "torchlens.overrides_drift_baseline.v1"
    assert payload["torch_version"]
    assert isinstance(payload["unseen"], list)
    assert payload["scope_note"]
