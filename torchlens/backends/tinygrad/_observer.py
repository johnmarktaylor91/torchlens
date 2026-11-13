"""Tensor-API observation for the tinygrad capture.

Moved out of ``backend`` so that module stays under its size ceiling. The
observer replaces ``Tensor._apply_uop`` for one forward. Every Tensor-API call
reports its result UOp here, with the callable's name, the active module stack,
the user source lines (``_source_lines``) and, for an edited capture, the
static-label edit that replaces the call's result (``_edits``). Nothing here
realizes a value: the replacement an edit returns is lazy, so the capture
still realizes once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._module_tree import _ACTIVE_TINYGRAD_MODULE_STACK, TinygradModuleFrame
from ._uop_graph import _uop_name

if TYPE_CHECKING:
    from ._edits import TinygradEditSession
    from ._source_lines import SourceLineRecorder

__tl_layer__ = "L2"


class _observe_tensor_ops:
    """Context manager observing tinygrad Tensor API UOp results."""

    def __init__(
        self,
        observed_ops: dict[int, list[str]],
        observed_module_stacks: dict[int, tuple[TinygradModuleFrame, ...]] | None = None,
        observed_tensors: dict[int, list[Any]] | None = None,
        pinned_uops: list[Any] | None = None,
    ) -> None:
        """Initialize the observation context.

        Parameters
        ----------
        observed_ops
            Mutable mapping receiving Tensor API names by returned UOp id.
        observed_module_stacks
            Optional mutable mapping receiving first-observed module stacks by
            returned UOp id.
        observed_tensors
            Optional mutable mapping receiving live result tensors by returned
            UOp id.
        pinned_uops
            Optional list receiving every returned UOp, in call order. The
            mappings above are keyed by ``id()``; a UOp tinygrad drops
            mid-forward frees its id for a later UOp, which would then inherit
            the dropped one's names. Holding the returned UOps for the capture
            keeps every key unique, and their order numbers the calls.
        The optional ``source_lines`` recorder (the user frames each call
        ran from) and ``edits`` session (static-label edits applied to call
        results) are attributes set after construction.
        """

        self.observed_ops = observed_ops
        self.observed_module_stacks = observed_module_stacks
        self.observed_tensors = observed_tensors
        self.pinned_uops = pinned_uops
        self.source_lines: SourceLineRecorder | None = None
        self.edits: TinygradEditSession | None = None
        self.paused = False
        self.original: Any = None

    def note(self, result: Any, name: str) -> None:
        """Record one call result under its UOp id.

        Parameters
        ----------
        result
            Tensor the call returned.
        name
            Name of the callable tinygrad applied.
        """

        key = id(result.uop)
        if self.pinned_uops is not None:
            self.pinned_uops.append(result.uop)
        self.observed_ops.setdefault(key, []).append(name)
        if self.observed_module_stacks is not None:
            self.observed_module_stacks.setdefault(key, tuple(_ACTIVE_TINYGRAD_MODULE_STACK))
        if self.observed_tensors is not None:
            self.observed_tensors.setdefault(key, []).append(result)
        if self.source_lines is not None:
            self.source_lines.record(result.uop)

    def alias(self, site: Any, replacement: Any) -> None:
        """Record an edit's replacement UOp under everything its site was observed with.

        Parameters
        ----------
        site
            UOp the call returned.
        replacement
            Lazy UOp the edit returned in its place.
        """

        if self.pinned_uops is not None:
            self.pinned_uops.append(replacement)
        self.observed_ops[id(replacement)] = list(self.observed_ops.get(id(site), ()))
        if self.observed_module_stacks is not None and id(site) in self.observed_module_stacks:
            self.observed_module_stacks[id(replacement)] = self.observed_module_stacks[id(site)]
        if self.source_lines is not None:
            self.source_lines.alias(site, replacement)

    def __enter__(self) -> _observe_tensor_ops:
        """Install the Tensor._apply_uop observer.

        Returns
        -------
        _observe_tensor_ops
            This context manager.
        """

        from tinygrad import Tensor

        self.original = Tensor._apply_uop

        def wrapped(tensor: Any, fxn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            """Observe one Tensor API operation returning a UOp-backed Tensor."""

            result = self.original(tensor, fxn, *args, **kwargs)
            if self.paused:
                return result
            name = str(getattr(fxn, "__name__", _uop_name(result.uop).lower()))
            self.note(result, name)
            if self.edits is None:
                return result
            return self.edits.apply(self, result, name)

        # tinygrad deliberately exposes this as a method; capture replaces it temporarily.
        setattr(Tensor, "_apply_uop", wrapped)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore the original Tensor._apply_uop implementation.

        Parameters
        ----------
        exc_type
            Exception type, if any.
        exc
            Exception value, if any.
        tb
            Traceback, if any.

        Returns
        -------
        None
            The monkeypatch is removed.
        """

        from tinygrad import Tensor

        # Restore the deliberately replaced tinygrad method after capture.
        setattr(Tensor, "_apply_uop", self.original)
