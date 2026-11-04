"""TL-authored synthetic root for bound-method capture (lane F41).

The ruled root contract (foldA MEMO s5 item 9 + D11, a direct maintainer ruling):
``tl.trace`` accepts an ``nn.Module`` OR a bound method of one. The owner
is resolved via ``method.__self__``; this TorchLens-authored wrapper root
calls the bound method EXACTLY ONCE (no probing, no retry) and presents the
owner's module tree as its own: the wrapper shares the owner's ``_modules``,
``_parameters`` and ``_buffers`` registries, so every tree walk
(``named_modules``, ``named_parameters``, ``named_buffers``, ``state_dict``)
yields the owner's descendants at the addresses a plain ``tl.trace(owner, ...)``
gives them (``transformer.h.5.mlp``, never ``owner.transformer.h.5.mlp``). The
owner itself is the capture's root module ``self``: its forward is instrumented
like a submodule, so a generation loop that calls the model once per token
records ``self:1`` .. ``self:N`` while a plain trace records ``self:1``.

The synthetic root discloses TL-authorship -- the ``bound_method:`` root
entry-point fact on the Trace and the ``Op.tl_authored_root`` marker on the
root op record (the output-boundary op records, the terminal records of the
wrapper's DAG) -- and its model identity reads ``type(owner).__name__``,
never the string ``"method"``.

Interim posture (foldA D11, never widened past the ruling): bound-method
captures refuse rerun/append at the ONE rerun identity gate
(``Trace._validate_supplied_model_matches_capture``), because re-executing
a capture of ``owner.generate`` by supplying ``owner`` would pass the
class+weights checks and run ``__call__`` under a report that can say
verified -- a silent-wrongness door the entry-point fact closes.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any, cast

import torch.nn as nn

from ..._errors import InvalidArgumentError

#: Address of the capture's root module (the bound method's owner).
ROOT_ADDRESS = "self"


def is_bound_method_of_module(candidate: Any) -> bool:
    """True when ``candidate`` is a bound method whose owner is an nn.Module."""

    return inspect.ismethod(candidate) and isinstance(
        getattr(candidate, "__self__", None), nn.Module
    )


class TLBoundMethodRoot(nn.Module):
    """TL-authored wrapper root for a bound method of an ``nn.Module``.

    The wrapper shares the owner's module, parameter and buffer registries,
    so it IS the owner's tree for every walk, at the owner's own addresses;
    model preparation instruments the owner (as the root ``self``) and every
    descendant. ``forward`` calls the exact bound method object the user
    passed, exactly once per invocation.
    """

    def __init__(self, method: Any) -> None:
        if not is_bound_method_of_module(method):
            raise TypeError(
                "TLBoundMethodRoot wraps a bound method of an nn.Module; "
                f"got {type(method).__name__}."
            )
        super().__init__()
        owner = method.__self__
        # Shared registries (by reference, never copied): the wrapper's tree
        # walks yield the owner's descendants at their plain addresses, the
        # weight fingerprint and state_dict equal the owner's, and nothing the
        # capture does to the wrapper's tree can diverge from the owner's.
        object.__setattr__(self, "_modules", owner._modules)
        object.__setattr__(self, "_parameters", owner._parameters)
        object.__setattr__(self, "_buffers", owner._buffers)
        # The root's training flag reads the owner's at capture time.
        self.training = bool(owner.training)
        object.__setattr__(self, "_tl_owner", owner)
        # The exact bound-method OBJECT the user passed (never re-fetched by
        # name at call time: an instance-attribute shadow must not swap the
        # entry point between acceptance and execution).
        object.__setattr__(self, "_tl_bound_method", method)
        object.__setattr__(self, "_tl_method_name", str(method.__name__))

    @property
    def tl_owner(self) -> nn.Module:
        """The owning module (``method.__self__``), the capture's root ``self``."""

        return cast(nn.Module, object.__getattribute__(self, "_tl_owner"))

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

    def named_modules(
        self,
        memo: set[nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Module]]:
        """The owner's tree, with the owner at the root position.

        Every tree walk torch builds on this (``modules``, ``named_parameters``,
        ``named_buffers``) and every TorchLens walker therefore sees the owner
        where a plain trace's root sits, never the wrapper.
        """

        owner = cast(nn.Module, object.__getattribute__(self, "_tl_owner"))
        yield from owner.named_modules(memo, prefix, remove_duplicate)

    def train(self, mode: bool = True) -> TLBoundMethodRoot:
        """Set the training mode on the wrapper and on the owner's tree."""

        self.training = bool(mode)
        self.tl_owner.train(mode)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped bound method exactly once (the ruled contract)."""

        # One call expression, no probing, no retry.
        method = object.__getattribute__(self, "_tl_bound_method")
        return method(*args, **kwargs)


def capture_root_module(model: nn.Module) -> nn.Module:
    """Return the module whose identity the root ``self`` reads.

    The owner for a :class:`TLBoundMethodRoot`, the model itself otherwise.
    """

    if isinstance(model, TLBoundMethodRoot):
        return model.tl_owner
    return model


def module_at_address(model: nn.Module, address: str) -> nn.Module:
    """The module at a root-relative address; ``ROOT_ADDRESS`` is the capture's root module."""

    if address == ROOT_ADDRESS:
        return capture_root_module(model)
    return cast(nn.Module, model.get_submodule(address))


def instrumented_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Yield every module whose forward the capture session instruments.

    Every module in the tree except the root object ``tl.trace`` runs: the
    root's forward is the capture's entry call and stays undecorated. On a
    :class:`TLBoundMethodRoot` that entry call is the bound method, and the
    owner (the root ``self`` of the tree walk) IS instrumented, so its own
    calls inside the method are counted like a submodule's.
    """

    for module in model.modules():
        if module is not model:
            yield module


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
        "input_shape=/inferred-input capture requires an nn.Module "
        f"root; received {type(model).__name__}. Declared and "
        "inferred input rungs synthesize inputs from a module's "
        "geometry, which a bound-method root does not expose",
        code="input_rung_requires_module_root",
        remedy=("pass a real input (the gold rung), e.g. tl.trace(model.generate, input_ids)"),
        argument="model",
        received_type=type(model).__name__,
    )
