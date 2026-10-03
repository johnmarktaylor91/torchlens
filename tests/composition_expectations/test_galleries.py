"""Gallery gates (row 0.6): fingerprints, fork isolation, consume-not-construct.

The galleries are the substrate every matrix shares (memo 0.6); these gates
prove the economics rules actually enforce -- the fingerprint detects
mutation, forks isolate, and no cell constructs its own products.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from tests.composition_expectations.conftest import (
    GALLERY_MODEL_TRAITS,
    build_gallery_model,
    trace_fingerprint,
)

pytestmark = pytest.mark.compo

TREE_ROOT = Path(__file__).resolve().parent

#: Modules licensed to CONSTRUCT products (the gallery itself + future
#: gallery extensions). Everything else consumes fixtures.
_CONSTRUCTION_LICENSED = {"conftest.py"}

_PRODUCT_CONSTRUCTORS = {"trace", "record", "merge_ranks", "extract_dataset", "load", "save"}


def test_model_gallery_roster_and_traits(model_gallery) -> None:
    """Every declared model builds, runs, and matches its trait row."""

    assert set(model_gallery) == set(GALLERY_MODEL_TRAITS)
    for name, (model, inputs) in model_gallery.items():
        output = model(*inputs)
        assert isinstance(output, torch.Tensor), name
        assert GALLERY_MODEL_TRAITS[name], f"{name} has no declared traits"


def test_product_gallery_serves_shared_products(product_gallery) -> None:
    """The session products exist and read as their declared kinds."""

    assert type(product_gallery["trace"]).__name__ == "Trace"
    assert type(product_gallery["recording"]).__name__ == "Recording"
    assert len(product_gallery["trace"].layer_labels) >= 4


def test_fingerprint_detects_mutation() -> None:
    """Red-capability: an in-place payload edit changes the fingerprint.

    Runs on a PRIVATE capture (never the shared session product -- proving
    the tripwire by tripping it on the shared artifact would poison every
    later cell).
    """

    import torchlens as tl

    model, inputs = build_gallery_model("mlp")
    trace = tl.trace(model, *inputs)
    try:
        before = trace_fingerprint(trace)
        assert trace_fingerprint(trace) == before, "fingerprint is not deterministic"
        label = trace.layer_labels[1]
        with torch.no_grad():
            trace[label].out.add_(1.0)
        assert trace_fingerprint(trace) != before, (
            "an in-place payload mutation left the fingerprint unchanged -- "
            "the teardown tripwire is disarmed"
        )
    finally:
        trace.cleanup()


def test_fork_for_mutation_isolates_the_shared_trace(product_gallery, fork_for_mutation) -> None:
    """A sanctioned edit verb on the fork leaves the shared Trace byte-identical.

    Mutating cells edit through ``do()`` (the engine substitutes at
    consumption); the parent fingerprint must not move and the fork must
    carry the edit.
    """

    import torchlens as tl

    before = trace_fingerprint(product_gallery["trace"])
    fork_for_mutation.do("relu_1_2", tl.zero_ablate())
    assert float(fork_for_mutation["relu_1_2"].out.detach().sum()) == 0.0
    assert trace_fingerprint(product_gallery["trace"]) == before


def test_raw_fork_writes_reach_the_shared_storage() -> None:
    """DISCLOSURE: forks share payload STORAGE; raw in-place writes poison the parent.

    The 75 ms fork cost (memo D12) comes exactly from not copying payloads:
    ``fork[label].out`` aliases the parent's tensor, so a raw ``mul_``
    through a fork corrupts the shared session product -- which is why the
    mutation rule says EDIT VERBS ONLY and why the gallery teardown
    fingerprints. Runs on a PRIVATE capture. If torchlens ever flips fork
    payloads to copy-on-write, this goes red and the rule can relax.
    """

    import torchlens as tl

    model, inputs = build_gallery_model("mlp")
    trace = tl.trace(model, *inputs, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = trace.fork()
    try:
        before = trace_fingerprint(trace)
        assert fork["relu_1_2"].out.data_ptr() == trace["relu_1_2"].out.data_ptr()
        with torch.no_grad():
            fork["relu_1_2"].out.mul_(0.0)
        assert trace_fingerprint(trace) != before, (
            "fork payloads stopped sharing storage (copy-on-write?) -- "
            "revisit the raw-write rule and the gallery teardown tripwire"
        )
    finally:
        fork.cleanup()
        trace.cleanup()


def _construction_violations(tree: ast.Module, label: str) -> list[str]:
    """Product-construction calls in one parsed composition test module."""

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name in _PRODUCT_CONSTRUCTORS:
            violations.append(f"{label}:{node.lineno} calls {name}()")
    return violations


def test_cells_consume_the_gallery_and_do_not_construct(request: pytest.FixtureRequest) -> None:
    """No composition TEST module constructs products (memo D12).

    Cells consume the session gallery; construction lives in the licensed
    gallery modules only. Red-capability plants that legitimately construct
    a private product declare themselves with the module-level marker
    ``COMPO_CONSTRUCTION_LICENSE = "<reason>"``.
    """

    del request
    violations: list[str] = []
    for path in sorted(TREE_ROOT.glob("test_*.py")):
        if path.name in _CONSTRUCTION_LICENSED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        licensed = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "COMPO_CONSTRUCTION_LICENSE"
                for target in node.targets
            )
            for node in tree.body
        )
        if licensed:
            continue
        violations.extend(_construction_violations(tree, path.name))
    assert not violations, (
        "composition cells must CONSUME the session gallery, never construct "
        "products (fork for mutation; declare a COMPO_CONSTRUCTION_LICENSE "
        "with a reason for sanctioned exceptions):\n  " + "\n  ".join(violations)
    )


def test_construction_lint_is_red_capable() -> None:
    """A planted trace() call in a cell module is flagged."""

    planted = ast.parse("import torchlens as tl\n\ndef test_x():\n    tl.trace(m, x)\n")
    assert _construction_violations(planted, "planted.py") == ["planted.py:4 calls trace()"]


# This module's own red-capability plant constructs a PRIVATE product
# (test_fingerprint_detects_mutation) -- the sanctioned, declared exception.
COMPO_CONSTRUCTION_LICENSE = (
    "fingerprint red-capability must trip the tripwire on a private capture, "
    "never the shared session product"
)
