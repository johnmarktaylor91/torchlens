"""Capability-derived guards for backward-only public accessors.

True-backward accessors must refuse typed on every backend whose registered
capability table declares ``backward_capture=False`` instead of returning
silently-empty results. The guarded surface is wider than the three headline
accessors: ``Op.grads`` / ``Module.grads``, ``Trace.autograd.backward_passes``,
``Trace.saved_grad_ops``, ``Trace.autograd.last_backward_pass``,
``len(Trace.saved_grad_ops)``, ``Trace.saved_grad_module_calls``, and
``Trace.saved_grad_modules`` all route through this guard, so on
``backward_capture=False`` backends they raise where they previously returned
``None``/``0``/empty. The registered table, not a hardcoded backend-name set,
is the authority; the error text redirects to ``trace.autograd.derived_grads`` only
when the backend actually declares a derived-gradient surface, and an
UNREGISTERED backend name refuses typed too — fail-closed, never silently
empty.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError
from ..backends.registry import BackendRegistryError, get_backend_spec


def registered_backend_named(name: str) -> bool:
    """Return whether ``name`` is the name of a backend registered in this runtime now.

    Parameters
    ----------
    name:
        Backend name to look up; an alias is not a backend name.

    Returns
    -------
    bool
        ``True`` when a registered spec carries exactly this name.
    """

    try:
        return get_backend_spec(name).name == name
    except BackendRegistryError:
        return False


def backend_materializes_payloads(name: str) -> bool:
    """Return whether the registered backend ``name`` rebuilds its saved payloads.

    Parameters
    ----------
    name:
        Backend name to look up; an alias or an unregistered name answers ``False``.

    Returns
    -------
    bool
        The registered capability ``payload_materialization``.
    """

    if not registered_backend_named(name):
        return False
    return get_backend_spec(name).capabilities.payload_materialization


def live_grad_slot(param_log: Any) -> bool:
    """Return whether the live parameter behind ``param_log`` carries a framework gradient slot.

    Only a true-backward backend (``backward_capture=True``) fills a gradient slot on
    the live parameter (``nn.Parameter.grad``). Every other backend reaches a Param's
    gradient only as a derived payload (``mirror_param_derived_grads``), so its live
    arrays and variables are never read for one.

    Parameters
    ----------
    param_log:
        Param record. With no owning trace, or an owning trace that ``cleanup()``
        husked (its backend fact deleted), the answer is ``True``: the live read then
        finds no parameter (the source-model reference is gone too) and returns ``None``.

    Returns
    -------
    bool
        The owning trace backend's registered ``backward_capture``; ``False`` for an
        unregistered backend name.
    """

    trace = param_log.trace
    backend = None if trace is None else trace.__dict__.get("_provenance_backend")
    if backend is None:
        return True
    name = str(backend)
    return registered_backend_named(name) and get_backend_spec(name).capabilities.backward_capture


def raise_if_no_backward_capture(trace: Any, *, plural_subject: str) -> None:
    """Raise when ``trace``'s backend cannot expose a true-backward accessor.

    Parameters
    ----------
    trace:
        Trace (or object carrying ``backend``) owning the accessor.
    plural_subject:
        Sentence subject describing what is unavailable, e.g.
        ``"op.autograd.grads or saved_grad_ops"``.

    Returns
    -------
    None
        Returns when the backend declares ``backward_capture=True``.

    Raises
    ------
    ValueError
        When the backend declares ``backward_capture=False``, or when the
        backend name is not registered at all — capability support cannot be
        checked for an unknown backend, so the guard fails closed instead of
        readmitting the silently-empty accessor behavior it exists to remove.
    """

    backend = str(trace._provenance_backend)
    try:
        capabilities = get_backend_spec(backend).capabilities
    except BackendRegistryError as exc:
        raise InvalidArgumentError(
            f"trace.provenance.backend={backend!r} is not a registered backend, so "
            f"{plural_subject} availability cannot be capability-checked; "
            "refusing instead of returning silently-empty results",
            code="unknown_backend",
            remedy="use a trace whose backend is registered in this session",
            backend=backend,
        ) from exc
    if capabilities.backward_capture:
        return
    if capabilities.intermediate_derived_grads:
        raise InvalidArgumentError(
            f"{backend} traces do not expose {plural_subject} because they do not "
            "capture true backward graphs",
            code="backend_unsupported",
            remedy=(
                "use trace.autograd.derived_grads for leaf-level derived gradients and "
                "op.autograd.derived_grad for exact op-level derived gradients"
            ),
            backend=backend,
        )
    raise InvalidArgumentError(
        f"{backend} traces do not expose {plural_subject} because they do not "
        f"capture true backward graphs, and the {backend} preview declares no "
        "derived-gradient surface",
        code="backend_unsupported",
        remedy="use the PyTorch backend for backward capture",
        backend=backend,
    )
