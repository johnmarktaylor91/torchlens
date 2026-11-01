"""Import the modules a process's first capture would otherwise load in-window.

``warm_lazy_torch_imports`` (``_torch_compat``) runs this before the RNG monitor
arms; the list names every module a first wrapped op imports lazily.
"""

from __future__ import annotations

import importlib

#: Modules a process's first wrapped op imports lazily. Torch's own
#: ``torch._compile``/``torch._dynamo`` cascade draws host entropy at module-exec
#: time, ``torch.backends.opt_einsum`` is imported inside ``torch.einsum``, and
#: TorchLens's producer-version parser is imported inside
#: ``resolve_runnable_torch_alias``, which the first wrapped op reaches in-window.
_FIRST_CAPTURE_LAZY_MODULES: tuple[str, ...] = (
    "torch._compile",
    "torch._dynamo",
    "torch.backends.opt_einsum",
    "torchlens._utils._producer_version",
)


def import_first_capture_lazy_modules() -> bool:
    """Import every first-capture lazy module under paused logging.

    Returns
    -------
    bool
        ``True`` when every import succeeded. Python evicts a partially executed
        failed import from ``sys.modules``, so the caller leaves its latch unset
        and the in-window retry's draws are honestly marked.
    """

    # Running this BEFORE the monitor's own patches install only protects
    # the RNG-channel census. The broader capture logging/dispatch window is
    # already armed by the time this runs (it fires from inside
    # ``run_and_log_inputs_through_model``), so the cascade's own tensor ops
    # -- torch._dynamo.variables.torch_function's module-level
    # ``populate_builtin_to_tensor_fn_map()`` calls ``torch.ones`` plus
    # several builtin unary ops (``abs``, ...) as an import side effect --
    # went through TorchLens's OWN wrapped functions and were recorded as
    # unattributable "unmodeled_tensor_return" / "caught_exception_control"
    # dispatch-census facts, permanently ceilinging the capture's witness
    # completeness even though nothing in the user's model caused them.
    # ``pause_logging()`` is the general mechanism for exactly this (critical
    # invariant 2): it hides the cascade from every capture-time witness, not
    # just the RNG monitor.
    from .._state import pause_logging

    warmed = True
    with pause_logging():
        for module_name in _FIRST_CAPTURE_LAZY_MODULES:
            try:
                importlib.import_module(module_name)
            except Exception:
                warmed = False
    return warmed
