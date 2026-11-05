"""Typed refusals raised from the torch function wrapper's logged call path."""

from __future__ import annotations

from ..._utils._dispatch_modes import TensorSubclassUnsupportedError
from ..._utils._torch_compat import HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE


def _raise_if_subclass_ctor_under_dispatch_mode(
    exc: RuntimeError, func_name: str, pauses_owned_modes: bool
) -> None:
    """Translate torch's subclass-under-dispatch-mode crash into the typed refusal."""
    if (
        pauses_owned_modes
        and not HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
        and "already associated to a python object" in str(exc)
    ):
        raise TensorSubclassUnsupportedError(
            f"torchlens could not construct the Tensor subclass via "
            f"{func_name!r} even after pausing its own dispatch mode on this "
            f"torch build (original torch error: {exc}). "
            "Remedy: upgrade torch (HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE reports "
            "False here), or avoid this construction while other TorchLens "
            "instrumentation observes the same forward on this build.",
            code="subclass_ctor_under_dispatch_mode_unsupported",
            func_name=func_name,
        ) from exc
