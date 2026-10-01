"""Provider adapters for the conformance runner (MEMO 4.1/4.2, B4).

The runner exercises providers through ONE neutral adapter protocol so a
buggy provider path can never serve as its own oracle. The torch reference
adapter wraps the public capture surface; the reserved ``"fake"`` adapter
ships SWITCHABLE PLANTS -- deliberate defects, each of which must fail the
intended check for the intended code (the anti-vacuity mechanism all three
labs derived independently). Toy models are the smoke pack and mutation
substrate ONLY; they emit no claim (see ``torchlens.conformance``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

__tl_layer__ = "L9"

#: Closed plant vocabulary for the reserved fake provider (MEMO 4.2): each
#: plant breaks exactly one checked property; ``shared_adapter_assumption``
#: breaks an assumption of the ADAPTER LAYER itself (load returning the very
#: object save received), so an oracle comparing an object to itself is
#: caught rather than trusted.
FAKE_PLANTS: tuple[str, ...] = (
    "structural_failure",
    "capability_undispatched",
    "output_corruption",
    "roundtrip_field_loss",
    "tamper_tripwire",
    "shared_adapter_assumption",
)


@dataclass(frozen=True)
class RosterModel:
    """One conformance roster row (model family + realism evidence).

    Parameters
    ----------
    family:
        Architecture family label (``"resnet"``, ``"gpt2"``, ...).
    realism:
        ``"pretrained"`` (pinned checkpoint with digest evidence) or
        ``"config_built"`` (random weights). Only pretrained rows can earn
        a claim -- toys earn nothing, not even suffixed.
    build:
        Zero-arg factory returning ``(model, example_input)``.
    checkpoint_evidence:
        Pinned checkpoint revision/digest for pretrained rows, else ``""``.
    """

    family: str
    realism: str
    build: Callable[[], tuple[Any, Any]]
    checkpoint_evidence: str = ""


class TorchReferenceAdapter:
    """The reference provider adapter over the public torch capture surface."""

    name = "torch"
    #: The reference provider may earn claims (the plant substrate never does).
    claim_eligible = True

    def capabilities(self) -> dict[str, bool]:
        """Declared capability flags for the torch reference backend."""

        return {"capture": True, "durable_save": True, "validation_replay": True}

    def trace(self, model: Any, example_input: Any) -> Any:
        """Capture one forward pass through the public surface."""

        import torchlens as tl

        return tl.trace(model, example_input)

    def reference_output(self, model: Any, example_input: Any) -> Any:
        """Run the model directly (the parity oracle's independent arm)."""

        return model(example_input)

    def save(self, trace: Any, path: Path) -> None:
        """Persist one captured trace as a portable artifact."""

        import torchlens as tl

        tl.save(trace, path)

    def load(self, path: Path) -> Any:
        """Reload one persisted artifact."""

        import torchlens as tl

        return tl.load(str(path))


class FakeProviderAdapter:
    """The reserved plant-bearing fake provider (never claim-eligible).

    Parameters
    ----------
    plant:
        One of :data:`FAKE_PLANTS`, or ``None`` for the well-behaved
        reference-delegating shape (used to prove the harness passes a
        correct provider before trusting any plant to fail it).
    """

    name = "fake"
    #: The reserved plant substrate never earns a claim, planted or not.
    claim_eligible = False

    def __init__(self, plant: str | None = None) -> None:
        """Arm zero or one plant."""

        if plant is not None and plant not in FAKE_PLANTS:
            raise ConfigurationError(
                f"Unknown conformance plant {plant!r}; the closed plant set is "
                f"{list(FAKE_PLANTS)}. Remedy: use a plant from the closed set.",
                code="conformance_plant_unknown",
                remedy="use a plant from the closed set",
            )
        self.plant = plant
        self._delegate = TorchReferenceAdapter()
        self._last_saved: Any = None

    def capabilities(self) -> dict[str, bool]:
        """Capability flags; one plant claims a capability it cannot serve."""

        flags = dict(self._delegate.capabilities())
        if self.plant == "capability_undispatched":
            flags["validation_replay"] = True
        return flags

    def trace(self, model: Any, example_input: Any) -> Any:
        """Capture, with the structural plant corrupting the graph."""

        trace = self._delegate.trace(model, example_input)
        if self.plant == "structural_failure":
            # Simulate a provider that silently drops captured ops.
            object.__setattr__(trace, "_conformance_dropped_ops", True)
        return trace

    def reference_output(self, model: Any, example_input: Any) -> Any:
        """Direct-run oracle arm; the corruption plant skews it."""

        output = self._delegate.reference_output(model, example_input)
        if self.plant == "output_corruption":
            import torch

            return output + torch.ones_like(output)
        return output

    def replay(self, trace: Any) -> Any:
        """Replay door consumed by the capability-consumption check."""

        if self.plant == "capability_undispatched":
            raise AttributeError("fake provider declared validation_replay but has no engine")
        return trace

    def save(self, trace: Any, path: Path) -> None:
        """Persist; the field-loss, identity, and tamper plants sabotage it."""

        self._delegate.save(trace, path)
        self._last_saved = trace
        if self.plant == "tamper_tripwire":
            # Forge the writer identity to an UNGOVERNED release after the
            # bytes land: the compat-ledger pair gate must refuse the reload
            # (c2_roundtrip_completes goes red), proving load-side tamper
            # detection rather than trusting the adapter's own report.
            import json

            from torchlens._io._json import read_bounded

            manifest_path = Path(path) / "manifest.json"
            manifest = read_bounded(manifest_path)
            manifest["torchlens_version"] = "2.30.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def load(self, path: Path) -> Any:
        """Reload; two plants corrupt this arm in distinct ways."""

        if self.plant == "shared_adapter_assumption":
            # Return the very object save received: an oracle that compares
            # live vs reloaded would compare an object to itself.
            return self._last_saved
        loaded = self._delegate.load(path)
        if self.plant == "roundtrip_field_loss":
            object.__setattr__(loaded, "_conformance_field_lost", True)
        return loaded


#: The closed v1 adapter vocabulary, dispatched as data (never a literal
#: comparison on the backend name -- the backend-registry lint owns those).
_ADAPTER_FACTORIES: dict[str, Callable[[str | None], Any]] = {
    TorchReferenceAdapter.name: lambda plant: TorchReferenceAdapter(),
    FakeProviderAdapter.name: lambda plant: FakeProviderAdapter(plant),
}


def resolve_adapter(backend: str, *, plant: str | None = None) -> Any:
    """Resolve one provider adapter by backend name.

    Parameters
    ----------
    backend:
        ``"torch"`` (the reference) or ``"fake"`` (the reserved plant
        substrate).
    plant:
        Optional plant for the fake provider.

    Returns
    -------
    Any
        The adapter instance.

    Raises
    ------
    ConfigurationError
        ``conformance_backend_unknown`` for any other name in v1 (external
        providers enter through the C0 provider-API pack once activated).
    """

    factory = _ADAPTER_FACTORIES.get(backend)
    if factory is not None:
        return factory(plant)
    raise ConfigurationError(
        f"Unknown conformance backend {backend!r}; v1 resolves 'torch' (the "
        "reference) and 'fake' (the reserved plant substrate). External "
        "providers are exercised after explicit activation through "
        "torchlens.ecosystem.plugins. Remedy: pass 'torch' or 'fake'.",
        code="conformance_backend_unknown",
        remedy="pass 'torch' or 'fake'",
        requested_backend=backend,
    )


def default_roster() -> tuple[RosterModel, ...]:
    """The zero-network default roster (config-built; NEVER claim-eligible).

    Returns
    -------
    tuple[RosterModel, ...]
        Config-built rows for the smoke pack. Pretrained packs with pinned
        revisions live behind extras and network access (MEMO section 7);
        their rows carry ``realism="pretrained"`` plus checkpoint evidence.
    """

    def _build_resnet() -> tuple[Any, Any]:
        """Config-built resnet18 (random weights) with a tiny input."""

        import torch
        from torchvision.models import resnet18

        return resnet18(weights=None).eval(), torch.randn(1, 3, 32, 32)

    def _build_mlp() -> tuple[Any, Any]:
        """Tiny random-weight MLP with a matching input."""

        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)
        ).eval()
        return model, torch.randn(2, 16)

    return (
        RosterModel(family="resnet", realism="config_built", build=_build_resnet),
        RosterModel(family="mlp", realism="config_built", build=_build_mlp),
    )
