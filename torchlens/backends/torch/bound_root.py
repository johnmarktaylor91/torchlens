"""TL-authored synthetic root for bound-method capture (lane F41).

The ruled root contract (foldA MEMO s5 item 9 + D11, a direct maintainer ruling):
``tl.trace`` accepts an ``nn.Module`` OR a bound method of one. The owner
is resolved via ``method.__self__`` and registered as a submodule of this
TorchLens-authored wrapper root, whose ``forward`` calls the bound method
EXACTLY ONCE (no probing, no retry). The synthetic root discloses
TL-authorship -- the ``bound_method:`` root entry-point fact on the Trace
and the ``Op.tl_authored_root`` marker on the root op record (the
output-boundary op records, the terminal records of the wrapper's DAG) --
and its model identity reads ``type(owner).__name__``, never the string
``"method"``.

Interim posture (foldA D11, never widened past the ruling): bound-method
captures refuse rerun/append at the ONE rerun identity gate
(``Trace._validate_supplied_model_matches_capture``), because re-executing
a capture of ``owner.generate`` by supplying ``owner`` would pass the
class+weights checks and run ``__call__`` under a report that can say
verified -- a silent-wrongness door the entry-point fact closes.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import torch.nn as nn

from ..._errors import InvalidArgumentError

#: Attribute name under which the owner registers as a submodule of the
#: wrapper root; captured module addresses read ``owner.<child>``.
OWNER_ATTRIBUTE = "owner"


def is_bound_method_of_module(candidate: Any) -> bool:
    """True when ``candidate`` is a bound method whose owner is an nn.Module."""

    return inspect.ismethod(candidate) and isinstance(
        getattr(candidate, "__self__", None), nn.Module
    )


class TLBoundMethodRoot(nn.Module):
    """TL-authored wrapper root for a bound method of an ``nn.Module``.

    The owner registers as the ``owner`` submodule so model preparation
    instruments it (and every descendant); ``forward`` calls the exact
    bound method object the user passed, exactly once per invocation.
    """

    def __init__(self, method: Any) -> None:
        if not is_bound_method_of_module(method):
            raise TypeError(
                "TLBoundMethodRoot wraps a bound method of an nn.Module; "
                f"got {type(method).__name__}."
            )
        super().__init__()
        owner = method.__self__
        # Registered submodule: model preparation walks the wrapper's tree,
        # so the owner and all its descendants are instrumented exactly as
        # they would be under a plain module-root capture.
        self.add_module(OWNER_ATTRIBUTE, owner)
        # The exact bound-method OBJECT the user passed (never re-fetched by
        # name at call time: an instance-attribute shadow must not swap the
        # entry point between acceptance and execution).
        object.__setattr__(self, "_tl_bound_method", method)
        object.__setattr__(self, "_tl_method_name", str(method.__name__))

    @property
    def tl_owner(self) -> nn.Module:
        """The owning module (``method.__self__``)."""

        return cast(nn.Module, getattr(self, OWNER_ATTRIBUTE))

    @property
    def tl_method_name(self) -> str:
        """Name of the wrapped bound method (e.g. ``"generate"``)."""

        name = object.__getattribute__(self, "_tl_method_name")
        return str(name)

    @property
    def tl_owner_class_name(self) -> str:
        """``type(owner).__name__`` -- the ruled synthetic-root identity."""

        return type(self.tl_owner).__name__

    @property
    def tl_owner_class_qualname(self) -> str:
        """Module-qualified owner class name (identity-evidence spelling)."""

        owner_type = type(self.tl_owner)
        return f"{owner_type.__module__}.{owner_type.__qualname__}"

    @property
    def tl_root_entry_point(self) -> str:
        """The ``bound_method:`` root invocation descriptor for this root."""

        return f"bound_method:{self.tl_owner_class_qualname}.{self.tl_method_name}"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped bound method exactly once (the ruled contract)."""

        # One call expression, no probing, no retry.
        method = object.__getattribute__(self, "_tl_bound_method")
        return method(*args, **kwargs)


def _reject_non_module_ladder_root(model: Any) -> nn.Module:
    """Return the model if it is an ``nn.Module``; refuse typed otherwise.

    F41: the quickstart rungs synthesize or infer inputs from an
    ``nn.Module``'s geometry; a bound-method (or any other callable) root
    takes the gold rung only. Previously this path leaked a bare
    ``AttributeError`` from the ladder.
    """

    if isinstance(model, nn.Module):
        return model
    raise InvalidArgumentError(
        "input_size=/inferred-input capture requires an nn.Module "
        f"root; received {type(model).__name__}. Declared and "
        "inferred input rungs synthesize inputs from a module's "
        "geometry, which a bound-method root does not expose",
        code="input_rung_requires_module_root",
        remedy=("pass a real input (the gold rung), e.g. tl.trace(model.generate, input_ids)"),
        argument="model",
        received_type=type(model).__name__,
    )
