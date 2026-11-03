"""Provider adapters for the conformance runner (MEMO 4.1/4.2, B4).

The runner exercises providers through ONE neutral adapter protocol so a
buggy provider path can never serve as its own oracle. The torch reference
adapter wraps the public capture surface; the reserved ``"fake"`` adapter
ships SWITCHABLE PLANTS -- deliberate defects, each of which must fail the
intended check for the intended code (the anti-vacuity mechanism all three
labs derived independently). Toy models are the smoke pack and mutation
substrate ONLY; they emit no claim (see ``tests.conformance``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torchlens as tl
from torchlens._io._json import read_bounded
from torchlens.backends.registry import BackendUnsupportedError, get_backend_spec
from torchlens.backends.torch import _torch_ops_calls
from torchlens.errors import ConfigurationError

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
    # protocol-era plants (SPEC 4.2): each fails exactly the cell named in
    # ``tests.conformance._protocol_cases.PLANT_INTENDED_CELL``.
    "untyped_refusal",
    "bare_bool_validation",
    "missing_constant_payload",
    "shape_drift_after_load",
    "flagged_trace",
    "shape_metadata_drift",
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
    builder:
        Zero-arg factory returning ``(model, example_input)``.
    checkpoint_evidence:
        Pinned checkpoint revision/digest for pretrained rows, else ``""``.
    expected_op_kinds:
        The independent op inventory: every op kind (``Op.func_name``) one
        forward of this model performs, in execution order, derived from the
        model definition. When declared, c1 runs ``c1_op_inventory_complete``
        so output parity can never pass with an op missing or extra.
    expected_intermediates:
        Independent oracle for intermediate values: op kind (``Op.func_name``)
        to the NumPy array (or the list of arrays, in capture order) that kind's
        ops must produce, computed from the row's own definition with NumPy.
        When declared, c1 runs ``c1_intermediates_match``.
    has_constants:
        The forward uses a tensor that is neither an input, a parameter nor
        the output of a recorded op (a closure constant or a literal
        producer); c1 then runs ``c1_constant_parents_recorded``.
    random_init:
        The builder seeds the framework RNG and draws random weights; on a
        lazy backend c1 then runs ``c1_rng_consumed_once``.
    group:
        ``"config_built"`` (the per-change rows; they earn no claim) or
        ``"ecosystem"`` (pinned pretrained families from the backend's own
        ecosystem, graduation criterion 3).
    """

    family: str
    realism: str
    builder: Callable[[], tuple[Any, Any]]
    checkpoint_evidence: str = ""
    expected_op_kinds: tuple[str, ...] | None = None
    expected_intermediates: dict[str, Any] | None = None
    has_constants: bool = False
    random_init: bool = False
    group: str = "config_built"


class TorchReferenceAdapter:
    """The reference provider adapter over the public torch capture surface."""

    name = "torch"
    #: The reference provider may earn claims (the plant substrate never does).
    claim_eligible = True

    def capabilities(self) -> dict[str, bool]:
        """Declared capability flags for the torch reference backend."""

        return {"capture": True, "durable_save": True, "validation_replay": True}

    def protocol(self) -> Any:
        """The registered torch ``BackendProtocol`` object (the suite's view of the backend)."""

        factory = get_backend_spec("torch").protocol
        return factory() if factory is not None else None

    def default_roster(self) -> tuple[RosterModel, ...]:
        """Torch's own roster (``_rosters/torch.py``): the reference rows."""

        from ._rosters.torch import roster

        return roster()

    def trace(self, model: Any, example_input: Any) -> Any:
        """Capture one forward pass through the torch protocol object."""

        return self.trace_with(model, example_input)

    def trace_with(self, model: Any, example_input: Any, **options: Any) -> Any:
        """Capture through ``protocol().capture`` (the public ``tl.trace`` entry).

        The protocol's capture records what replay validation needs unless
        ``options`` carries its own ``capture=``; refusal cells pass their
        trace options here.
        """

        return self.protocol().capture(model, example_input, None, **options)

    def reference_output(self, model: Any, example_input: Any) -> Any:
        """Run the model directly through the protocol's pristine oracle arm."""

        protocol = self.protocol()
        if protocol is None:
            return model(example_input)
        return protocol.reference_output(model, example_input, None)

    def side_effect_snapshot(self) -> Any:
        """Framework state a refusal must leave untouched: RNG bytes and wrapper count."""

        import torch

        return (torch.get_rng_state().numpy().tobytes(), len(_torch_ops_calls._WRAPPED_FUNC_CODE))

    #: Save level whose bundle keeps what torch replay validation re-executes
    #: from (saved arguments and RNG states); ``portable`` drops them by design.
    #: Only ``c2_validation_after_load`` saves at it (SPEC 4.2: validation of
    #: the loaded trace passes); every other c2 cell keeps the default level.
    replay_save_level = "executable_with_callables"

    def save(self, trace: Any, path: Path, level: str | None = None) -> None:
        """Persist one captured trace (default level unless ``level`` is given)."""

        if level is None:
            trace.save(path)
        else:
            trace.save(path, level=level)

    def load(self, path: Path) -> Any:
        """Reload one persisted artifact."""

        return tl.load(str(path))


class PlantedProtocol:
    """The fake provider's protocol view: torch's object with one plant armed.

    Parameters
    ----------
    inner:
        The torch protocol object every member delegates to.
    plant:
        The armed plant, or ``None``.
    """

    def __init__(self, inner: Any, plant: str | None) -> None:
        self._inner = inner
        self.plant = plant

    def __getattr__(self, name: str) -> Any:
        """Delegate every member the plants do not touch."""

        return getattr(self._inner, name)

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay validation; the bare-bool plant collapses a failure to ``False``."""

        status = self._inner.validate(trace, ground_truth)
        if self.plant == "bare_bool_validation" and getattr(status, "state", None) != "passed":
            return False
        return status


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

    def protocol(self) -> Any:
        """Torch's protocol object behind the plant layer."""

        return PlantedProtocol(self._delegate.protocol(), self.plant)

    def trace(self, model: Any, example_input: Any) -> Any:
        """Capture, with the structural plant corrupting the graph."""

        return self.trace_with(model, example_input)

    def trace_with(self, model: Any, example_input: Any, **options: Any) -> Any:
        """Capture through the public entry; the capture-side plants act here.

        The fake's capability table is torch's, whose one False flag owes a
        typed refusal when its option is requested; the untyped plant raises a
        plain ``RuntimeError`` instead (the refusal cell must read failed).
        """

        from ._protocol_cases import false_capability_flags
        from ._refusal_cases import request_option_for_flag

        for flag in false_capability_flags(self._delegate.protocol()):
            option = request_option_for_flag(flag)
            if option is not None and options.get(option) is not None:
                if self.plant == "untyped_refusal":
                    raise RuntimeError(f"fake provider cannot serve {option}")
                raise BackendUnsupportedError(
                    f"fake provider declares {flag}=False; {option}= is refused",
                    code=f"{flag}_unsupported",
                    remedy="omit the option",
                )
        trace = self._delegate.trace_with(model, example_input, **options)
        if self.plant == "structural_failure":
            # Simulate a provider that silently drops captured ops.
            object.__setattr__(trace, "_conformance_dropped_ops", True)
        if self.plant == "flagged_trace":
            # A capture that recorded a gap and flagged itself instead of refusing.
            trace._provenance_verified = False
            trace._provenance_verification_reason = "unwrapped_op_in_capture"
        if self.plant == "missing_constant_payload":
            _drop_first_constant_payload(trace)
        if self.plant == "shape_metadata_drift":
            _drift_first_shape_metadata(trace)
        return trace

    def side_effect_snapshot(self) -> Any:
        """Torch's framework state (the fake runs on torch)."""

        return self._delegate.side_effect_snapshot()

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

    @property
    def replay_save_level(self) -> str:
        """The torch delegate's replay save level (the fake validates through torch)."""

        return self._delegate.replay_save_level

    def save(self, trace: Any, path: Path, level: str | None = None) -> None:
        """Persist; the field-loss, identity, and tamper plants sabotage it."""

        self._delegate.save(trace, path, level=level)
        self._last_saved = trace
        if self.plant == "tamper_tripwire":
            # Forge the writer identity to an earlier MAJOR release after the
            # bytes land: the load door must refuse the reload
            # (c2_roundtrip_completes goes red), proving load-side tamper
            # detection rather than trusting the adapter's own report.
            import json

            manifest_path = Path(path) / "manifest.json"
            manifest = read_bounded(manifest_path)
            manifest["torchlens_version"] = "1.0.0"
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
        if self.plant == "shape_drift_after_load":
            _drift_first_payload_shape(loaded)
        return loaded


def _drop_first_constant_payload(trace: Any) -> None:
    """Strip the saved payload of the first constant-like op (a recorded node without data)."""

    from ._protocol_cases import constant_like_ops

    for op in constant_like_ops(trace):
        if getattr(op, "out", None) is not None:
            op.out = None
            return


def _drift_first_shape_metadata(trace: Any) -> None:
    """Give the first saved payload's recorded ``shape`` a leading unit axis (payload kept)."""

    import warnings

    for op in trace._layer_list:
        shape = getattr(op, "shape", None)
        if getattr(op, "out", None) is None or shape is None:
            continue
        with warnings.catch_warnings():
            # The metadata write is the plant itself, not a recipe edit.
            warnings.simplefilter("ignore")
            op.shape = (1, *tuple(shape))
        return


def _drift_first_payload_shape(trace: Any) -> None:
    """Give the first saved payload a leading unit axis (values kept, shape drifted)."""

    for op in trace._layer_list:
        payload = getattr(op, "out", None)
        reshape = getattr(payload, "reshape", None)
        if payload is None or not callable(reshape):
            continue
        op.out = reshape(1, *tuple(payload.shape))
        return


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
        ``"torch"`` (the reference), ``"fake"`` (the reserved plant
        substrate), or a preview backend (``"jax"``, ``"tf"``, ``"mlx"``,
        ``"tinygrad"``, ``"paddle"``).
    plant:
        Optional plant for the fake provider.

    Returns
    -------
    Any
        The adapter instance.

    Raises
    ------
    ConfigurationError
        ``conformance_backend_unknown`` for any other name (external
        providers enter through the c0 provider-API pack once activated).
    """

    from ._preview_adapters import PREVIEW_ADAPTER_FACTORIES

    factory = _ADAPTER_FACTORIES.get(backend) or PREVIEW_ADAPTER_FACTORIES.get(backend)
    if factory is not None:
        return factory(plant)
    raise ConfigurationError(
        f"Unknown conformance backend {backend!r}; the runner resolves 'torch' "
        "(the reference), 'fake' (the reserved plant substrate) and the preview "
        f"backends {sorted(PREVIEW_ADAPTER_FACTORIES)}. External providers are "
        "exercised after explicit activation through torchlens._ecosystem.plugins. "
        "Remedy: pass 'torch', 'fake' or a preview backend name.",
        code="conformance_backend_unknown",
        remedy="pass 'torch', 'fake' or a preview backend name",
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
        RosterModel(family="resnet", realism="config_built", builder=_build_resnet),
        RosterModel(family="mlp", realism="config_built", builder=_build_mlp),
    )
