"""Lazy capability-implementation resolvers for the default backend specs.

Each resolver imports its implementing surface on first use, so registering the
default specs imports no framework. ``capability_bindings(name)`` returns the
``BackendSpec.capability_implementations`` mapping for one default backend.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import MappingProxyType


def _torch_interventions_implementation() -> object:
    """Resolve the torch live-intervention implementing surface.

    Returns
    -------
    object
        The live hook applier the torch capture hot path invokes for
        ``trace(intervene=...)`` sites.
    """

    from .torch._ops_emission import apply_live_hooks_to_outputs

    return apply_live_hooks_to_outputs


def _tf_interventions_implementation() -> object:
    """Resolve the TensorFlow static-label intervention implementing surface.

    Returns
    -------
    object
        The writable-layer normalizer the tf capture path dispatches for
        ``trace(intervene=...)`` sites.
    """

    from .tf.interventions import normalize_tf_interventions

    return normalize_tf_interventions


def _torch_fastlog_implementation() -> object:
    """Resolve the torch sparse-recording implementing surface.

    Returns
    -------
    object
        The ``Recorder`` class ``tl.record()`` actually runs.
    """

    from ..recording._recorder import Recorder

    return Recorder


def _torch_structure_only_implementation() -> object:
    """Resolve the torch structure-only capture implementing surface.

    Returns
    -------
    object
        The escalated escape-belt installer the torch capture path enters
        for ``trace(structure_only=True)`` sessions.
    """

    from .torch.structure_only_belt import structure_only_escape_belt

    return structure_only_escape_belt


def _torch_streaming_implementation() -> object:
    """Resolve the torch streaming-save implementing surface.

    Returns
    -------
    object
        The disk storage backend behind ``storage=`` / ``streaming=`` saves.
    """

    from ..recording._storage_disk import DiskStorageBackend

    return DiskStorageBackend


def _torch_backward_capture_implementation() -> object:
    """Resolve the torch backward-capture implementing surface.

    Returns
    -------
    object
        The torch backward capture module.
    """

    from .torch import backward

    return backward


def _torch_rng_replay_implementation() -> object:
    """Resolve the torch RNG snapshot/replay implementing surface.

    Returns
    -------
    object
        The RNG snapshot entry on the torch capture adapter.
    """

    from .torch.backend import TorchBackend

    return TorchBackend.snapshot_rng


def _tinygrad_interventions_implementation() -> object:
    """Resolve the tinygrad edit-plan resolver behind ``can_intervene`` (lazy import)."""

    from .tinygrad._edit_plan import resolve_tinygrad_edit_plan

    return resolve_tinygrad_edit_plan


def _paddle_interventions_implementation() -> object:
    """Resolve the Paddle live-intervention implementing surface.

    Returns
    -------
    object
        Runtime class the Paddle capture wrapper dispatches for
        ``trace(intervene=...)`` / ``trace(halt=...)`` sites. Import-light:
        the module defers the paddle import to apply time.
    """

    from .paddle.interventions import PaddleInterventionRuntime

    return PaddleInterventionRuntime


def _mlx_interventions_implementation() -> object:
    """Resolve the MLX static-label intervention implementing surface.

    Returns
    -------
    object
        The plan resolver the MLX capture path invokes for
        ``trace(intervene=...)`` / ``trace(halt=...)`` sites.
    """

    from .mlx.interventions import resolve_mlx_intervention_plan

    return resolve_mlx_intervention_plan


def _jax_interventions_implementation() -> object:
    """Resolve the JAX capture-time edit runtime behind ``can_intervene`` (lazy import)."""

    from .jax._edits import JaxEditRuntime

    return JaxEditRuntime


def _lazy_binding(
    module: str, attr: str | None = None, *, call: bool = False
) -> Callable[[], object]:
    """Return a resolver for a backend's record, retention or stream surface.

    The surface is named as data and imported on first use, so binding a
    backend's ``tl.record`` producer here imports neither that framework nor its
    capture modules (and adds no import edge to this table).

    Parameters
    ----------
    module
        Absolute module path of the implementing surface.
    attr
        Attribute of ``module`` to return; ``None`` returns the module itself.
    call
        Call the attribute and return its result (a fresh producer per resolve).

    Returns
    -------
    Callable[[], object]
        Zero-argument resolver.
    """

    def _resolve() -> object:
        """Import ``module`` and return the bound object (called when ``call`` is set)."""

        target = importlib.import_module(module)
        if attr is None:
            return target
        found = getattr(target, attr)
        return found() if call else found

    return _resolve


_BINDINGS: Mapping[str, Mapping[str, Callable[[], object]]] = MappingProxyType(
    {
        "torch": {
            "backward_capture": _torch_backward_capture_implementation,
            "fastlog": _torch_fastlog_implementation,
            "can_intervene": _torch_interventions_implementation,
            "rng_replay": _torch_rng_replay_implementation,
            "has_streaming": _torch_streaming_implementation,
            "structure_only_capture": _torch_structure_only_implementation,
        },
        "jax": {
            "can_intervene": _jax_interventions_implementation,
            "fastlog": _lazy_binding(
                "torchlens.backends.jax._record", "jax_record_producer", call=True
            ),
        },
        "mlx": {
            "can_intervene": _mlx_interventions_implementation,
            "fastlog": _lazy_binding(
                "torchlens.backends.mlx._record", "mlx_record_producer", call=True
            ),
            "structure_only_capture": _lazy_binding(
                "torchlens.backends.mlx._retention", "retention_scope"
            ),
        },
        "tinygrad": {
            "can_intervene": _tinygrad_interventions_implementation,
        },
        "paddle": {
            "can_intervene": _paddle_interventions_implementation,
            "fastlog": _lazy_binding(
                "torchlens.backends.paddle._record", "paddle_record_producer", call=True
            ),
            "structure_only_capture": _lazy_binding("torchlens.backends.paddle._stream_capture"),
            "has_streaming": _lazy_binding("torchlens.backends.paddle._stream_capture"),
        },
        "tf": {
            "can_intervene": _tf_interventions_implementation,
            "fastlog": _lazy_binding(
                "torchlens.backends.tf._record", "tf_record_producer", call=True
            ),
        },
    }
)


def capability_bindings(name: str) -> dict[str, Callable[[], object]]:
    """Return a fresh copy of the capability-implementation mapping for ``name``.

    Parameters
    ----------
    name
        Default backend spec name.

    Returns
    -------
    dict[str, Callable[[], object]]
        Capability flag name to lazy resolver; empty for a backend with none.
    """

    return dict(_BINDINGS.get(name, {}))
