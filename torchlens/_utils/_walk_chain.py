"""The call-site chain of one full code-context walk (``introspection._get_code_context``).

Split out of ``introspection`` so that module stays under its size cap. The
chain itself is :func:`torchlens._utils._call_site.cached_chain` of the walk's
first ``boundary`` survivors, with a module token at each hook or crossed
stateless module's place.
"""

from types import CodeType, FrameType
from typing import Any

from . import _call_site

__all__ = ("call_site_chain",)


def call_site_chain(
    survivors: list[tuple[FrameType, CodeType, int, int]],
    boundary: int,
    context_cache: dict[Any, Any] | None,
    hook_scopes: list[tuple[int, Any]] | tuple[()] = (),
) -> tuple[Any, ...]:
    """Return the call-site chain of the first ``boundary`` survivors.

    ``survivors`` runs innermost first (see :func:`_call_site.chain_boundary`). Each
    ``hook_scopes`` entry (a module whose hook the op runs in, or a stateless
    module the op runs inside) becomes a module token at its place in the chain.
    """

    if boundary <= 0:
        return ()
    raw_frames: list[tuple[Any, int]] = []
    scopes = iter(hook_scopes)
    scope = next(scopes, None)
    for position, (_frame, code, _lineno, lasti) in enumerate(survivors[:boundary]):
        while scope is not None and scope[0] <= position:
            raw_frames.append(_call_site.module_scope_entry(scope[1]))
            scope = next(scopes, None)
        raw_frames.append((code, lasti))
    # A module crossed just inside the boundary (no survivor between it and the
    # scope's frames, e.g. under a root whose entry is not named ``forward``).
    while scope is not None and scope[0] <= boundary:
        raw_frames.append(_call_site.module_scope_entry(scope[1]))
        scope = next(scopes, None)
    return _call_site.cached_chain(tuple(raw_frames), context_cache)
