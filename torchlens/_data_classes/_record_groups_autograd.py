"""The autograd group of an ``Op`` or ``Layer``: ``op.autograd.grad_fn``, ``op.autograd.grads``, ...

The record's autograd facts: its ``grad_fn`` node and class, the saved
gradient records with their shape and dtype, and the derived intermediate
gradient. Stored members read the record's private storage attribute
(``_autograd_<leaf>``, same persisted column as before).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._record_group_base import _LayerGroup, _RecordGroup

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class OpAutograd(_RecordGroup):
    """Autograd facts and saved gradients of one Op.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.ops["linear_1_1"].autograd, tl.types.OpAutograd)
    True
    """

    __slots__ = ()
    _record: Op

    @property
    def derived_grad(self) -> Any:
        """Backend-owned intermediate gradient payload for this Op, or `None`."""

        return self._record._autograd_derived_grad

    @property
    def grad_dtype(self) -> Any:
        """Dtype of this Op's saved output gradient."""

        return self._record._autograd_grad_dtype

    @property
    def grad_fn(self) -> Any:
        """The autograd ``grad_fn`` node of this Op's output tensor, or ``None``."""

        return self._record._autograd_grad_fn

    @property
    def grad_fn_class_name(self) -> Any:
        """Class name of this Op's autograd node."""

        return self._record._autograd_grad_fn_class_name

    @property
    def grad_fn_class_qualname(self) -> Any:
        """Qualified class name of this Op's autograd node."""

        return self._record._autograd_grad_fn_class_qualname

    @property
    def grad_fn_handle(self) -> Any:
        """The live PyTorch autograd node object for this Op; not portable."""

        return self._record._autograd_grad_fn_handle

    def grad_for(self, *, bwd: int) -> Any:
        """Return the saved gradient tensor for one backward pass."""

        return self._record._autograd_grad_for(bwd=bwd)

    @property
    def grad_shape(self) -> Any:
        """Shape of this Op's saved output gradient."""

        return self._record._autograd_grad_shape

    @property
    def grads(self) -> Any:
        """Accessor over the per-backward-pass gradient records saved for this Op."""

        return self._record._autograd_grads

    @property
    def num_autograd_tensors(self) -> Any:
        """Number of tensors the captured forward's autograd saved for this Op's backward."""

        return self._record._autograd_num_autograd_tensors


class LayerAutograd(_LayerGroup):
    """Autograd facts of one Layer.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.layers["linear_1"].autograd, tl.types.LayerAutograd)
    True
    """

    __slots__ = ()
    _record: Layer
    _GROUP = "autograd"

    @property
    def grad_fn(self) -> Any:
        """The autograd ``grad_fn`` node of this layer's output tensor, or ``None``."""

        return self._record._autograd_grad_fn

    @property
    def grad_fn_class_name(self) -> Any:
        """Class name of the autograd node that produced this layer's output, or `None`."""

        return self._record._autograd_grad_fn_class_name

    @property
    def grad_fn_class_qualname(self) -> Any:
        """Qualified class name of the autograd node that produced this layer's output, or `None`."""

        return self._record._autograd_grad_fn_class_qualname

    @property
    def grad_fn_handle(self) -> Any:
        """The live autograd node object for this layer's output, not kept in saved files."""

        return self._record._autograd_grad_fn_handle

    @property
    def num_autograd_tensors(self) -> Any:
        """Number of tensors autograd saved for this layer's backward, as carried by the layer."""

        return self._record._autograd_num_autograd_tensors


__all__ = ["LayerAutograd", "OpAutograd"]
