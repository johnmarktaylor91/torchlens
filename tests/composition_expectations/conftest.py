"""Composition-tree conftest: census prewarm + product/model galleries (row 0.6).

Fixture economics are measured law (compo memo D12): a save+load hop is
~4.7x a capture, so cells CONSUME the session gallery and may not construct
products -- read-only cells share, mutating cells fork (~75 ms even on a
567-op trace), persisting cells are declared and counted, and teardown
fingerprints prove the shared artifacts left the session unchanged.

The census prewarm keeps the ~10s S-17/S-18 source walks outside every
charged test window (the root conftest's ``_source_corpus`` idiom).
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

import pytest
import torch
from torch import nn

#: Model gallery: structural fixtures with DECLARED traits (the model-trait
#: axis consumes these; R0 real-class fixtures are P04's roster and join at
#: wave A -- a toy row can never discharge a learned-semantics obligation).
GALLERY_MODEL_TRAITS: dict[str, tuple[str, ...]] = {
    "mlp": ("feedforward",),
    "tied": ("tied_weights",),
    "multi_input": ("multi_input",),
}


class _TiedLinear(nn.Module):
    """Two consumptions of ONE weight (the identity-collapse substrate)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(x @ self.weight)
        return hidden @ self.weight


class _MultiInput(nn.Module):
    """Two heterogeneous inputs (the positional-guess substrate)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * scale


def build_gallery_model(name: str) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    """Construct one gallery model with a fixed-seed example input."""

    generator = torch.Generator().manual_seed(0)
    if name == "mlp":
        torch.manual_seed(0)
        model: nn.Module = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        inputs: tuple[torch.Tensor, ...] = (torch.randn(2, 8, generator=generator),)
    elif name == "tied":
        torch.manual_seed(0)
        model = _TiedLinear()
        inputs = (torch.randn(2, 8, generator=generator),)
    elif name == "multi_input":
        torch.manual_seed(0)
        model = _MultiInput()
        inputs = (torch.randn(2, 8, generator=generator), torch.ones(2, 1))
    else:  # pragma: no cover - registry drift guard
        raise KeyError(f"unknown gallery model {name!r}")
    return model.eval(), inputs


def trace_fingerprint(trace: Any) -> str:
    """Content fingerprint of a shared Trace: labels + saved payload bytes.

    Teardown compares fingerprints to PROVE read-only cells did not mutate
    the shared artifact (memo D12). Payloads hash detached CPU bytes, so any
    in-place edit, payload swap, or structural drift changes the digest.
    """

    digest = hashlib.sha256()
    for label in trace.layer_labels:
        digest.update(label.encode())
        out = trace[label].out
        if isinstance(out, torch.Tensor):
            digest.update(out.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="session")
def model_gallery() -> dict[str, tuple[nn.Module, tuple[torch.Tensor, ...]]]:
    """The declared structural model roster (consume, never rebuild)."""

    return {name: build_gallery_model(name) for name in GALLERY_MODEL_TRAITS}


@pytest.fixture(scope="session")
def product_gallery(
    model_gallery: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...]]],
) -> Any:
    """Session products every composition cell CONSUMES (never constructs).

    Wave 0 ships the two cheapest products (a live intervention-ready Trace
    and a sparse Recording); M1's full 14-state roster joins at wave A
    through this one fixture. Teardown re-fingerprints the shared Trace and
    fails the session if any cell mutated it in place.
    """

    import torchlens as tl

    model, inputs = model_gallery["mlp"]
    trace = tl.trace(model, *inputs, capture=tl.options.CaptureOptions(intervention_ready=True))
    recording = tl.record(model, *inputs, save=tl.func("relu"))
    products = {"trace": trace, "recording": recording}
    baseline = trace_fingerprint(trace)
    try:
        yield products
        assert trace_fingerprint(trace) == baseline, (
            "a composition cell MUTATED the shared session Trace in place; "
            "mutating cells must fork (fork_for_mutation) -- memo D12"
        )
    finally:
        trace.cleanup()


@pytest.fixture()
def fork_for_mutation(product_gallery: dict[str, Any]) -> Any:
    """A fresh fork of the shared Trace for ONE mutating cell (~75 ms)."""

    fork = product_gallery["trace"].fork()
    yield fork
    fork.cleanup()


def pytest_collection_finish(session: pytest.Session) -> None:
    """Prewarm the census caches iff a consumer module was collected."""

    del session
    censuses = sys.modules.get("tests.composition_expectations._censuses")
    if censuses is not None:
        censuses.raise_sites()
        censuses.warn_sites()
        censuses.env_var_reads()
