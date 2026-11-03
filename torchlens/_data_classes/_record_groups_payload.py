"""The payload group of an ``Op`` or ``Layer``: ``op.payload.transformed_out``, ...

The saved payloads beyond the raw output and gradient: the transformed output
and gradient with their shapes and dtypes, and lazy materialization of a
bundle-backed payload. Stored members read the record's private storage
attribute (``_payload_<leaf>``, same persisted column as before).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._errors import ArtifactError
from ._record_group_base import _LayerGroup, _RecordGroup

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class OpPayload(_RecordGroup):
    """Saved payloads of one Op: transformed output and gradient, and lazy materialization.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.ops["linear_1_1"].payload, tl.types.OpPayload)
    True
    """

    __slots__ = ()
    _record: Op

    def materialize_grad(
        self,
        *,
        map_location: str | torch.device = "cpu",
        payload_hints: Any | None = None,
    ) -> Any:
        """Materialize this Op's saved grad from a lazy bundle reference.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.
        payload_hints:
            Optional backend payload hints used during materialization.

        Returns
        -------
        Any
            The materialized grad payload.

        Raises
        ------
        ArtifactError
            If this Op has no ``grad_ref`` to materialize from.
        """

        op = self._record
        current = op._slot("grad")
        if current is not None:
            return current
        ref = op.grad_ref
        if ref is None:
            raise ArtifactError("no grad_ref to materialize from")
        op._internal_set(
            "grad", ref.materialize(map_location=map_location, payload_hints=payload_hints)
        )
        return op.grad

    def materialize_out(
        self,
        *,
        map_location: str | torch.device = "cpu",
        payload_hints: Any | None = None,
    ) -> Any:
        """Materialize this Op's saved out from a lazy bundle reference.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.
        payload_hints:
            Optional backend payload hints used during materialization.

        Returns
        -------
        Any
            The materialized out payload.

        Raises
        ------
        ArtifactError
            If this Op has no ``out_ref`` to materialize from.
        """

        op = self._record
        current = op._slot("out")
        if current is not None:
            return current
        ref = op.out_ref
        if ref is None:
            raise ArtifactError("no out_ref to materialize from")
        op._internal_set(
            "out", ref.materialize(map_location=map_location, payload_hints=payload_hints)
        )
        return op.out

    @property
    def transformed_grad(self) -> Any:
        """This Op's saved gradient after the trace's gradient transform, or `None`."""

        return self._record._payload_transformed_grad

    @property
    def transformed_grad_dtype(self) -> Any:
        """Dtype of this Op's saved transformed gradient."""

        return self._record._payload_transformed_grad_dtype

    @property
    def transformed_grad_shape(self) -> Any:
        """Shape of this Op's saved transformed gradient."""

        return self._record._payload_transformed_grad_shape

    @property
    def transformed_out(self) -> Any:
        """This Op's saved output after the trace's activation transform, or `None`."""

        return self._record._payload_transformed_out

    @property
    def transformed_out_dtype(self) -> Any:
        """Dtype of this Op's saved transformed output."""

        return self._record._payload_transformed_out_dtype

    @property
    def transformed_out_shape(self) -> Any:
        """Shape of this Op's saved transformed output."""

        return self._record._payload_transformed_out_shape


class LayerPayload(_LayerGroup):
    """Saved transformed payloads of one Layer, read from its single pass.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.layers["linear_1"].payload, tl.types.LayerPayload)
    True
    """

    __slots__ = ()
    _record: Layer
    _GROUP = "payload"

    @property
    def transformed_grad(self) -> Any:
        """Transformed gradient of the single pass, for single-pass layers."""

        return self._record._single_pass_or_error("transformed_grad")

    @property
    def transformed_grad_dtype(self) -> Any:
        """Dtype of the saved transformed gradient, or `None`."""

        return self._record._payload_transformed_grad_dtype

    @property
    def transformed_grad_shape(self) -> Any:
        """Shape of the saved transformed gradient, or `None`."""

        return self._record._payload_transformed_grad_shape

    @property
    def transformed_out(self) -> Any:
        """Transformed output of the single pass, for single-pass layers."""

        return self._record._single_pass_or_error("transformed_out")

    @property
    def transformed_out_dtype(self) -> Any:
        """Dtype of the saved transformed output, or `None`."""

        return self._record._payload_transformed_out_dtype

    @property
    def transformed_out_shape(self) -> Any:
        """Shape of the saved transformed output, or `None`."""

        return self._record._payload_transformed_out_shape


__all__ = ["LayerPayload", "OpPayload"]
