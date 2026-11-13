"""Lazy capability-implementation resolvers for the default backend specs.

Each resolver imports its implementing surface on first use, so registering the
default specs imports no framework. ``capability_bindings(name)`` returns the
``BackendSpec.capability_implementations`` mapping for one default backend.
"""

from __future__ import annotations

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


def _native_backward_binding(name: str) -> Callable[[], object]:
    """Return a resolver for ``name``'s native backward surface (imports no framework)."""

    def _resolve() -> object:
        from ._backward_native import native_backward_surface

        return native_backward_surface(name)

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
            "backward_capture": _native_backward_binding("jax"),
        },
        "mlx": {
            "can_intervene": _mlx_interventions_implementation,
        },
        "tinygrad": {
            "can_intervene": _tinygrad_interventions_implementation,
        },
        "paddle": {
            "can_intervene": _paddle_interventions_implementation,
        },
        "tf": {
            "can_intervene": _tf_interventions_implementation,
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
