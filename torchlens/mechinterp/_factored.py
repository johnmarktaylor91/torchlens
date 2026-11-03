"""``FactoredMatrix``: low-rank ``A @ B`` algebra (vendored from TransformerLens).

Vendored from TransformerLens (https://github.com/TransformerLensOrg/TransformerLens),
file ``transformer_lens/FactoredMatrix.py`` at tag ``v4.2.0``, commit
``bd55eab6737584685b921fdfcd15f191579e64b1``, under the MIT License reproduced
below. Credited in ``docs/acknowledgments.md``.

Adaptations for torchlens (the algebra is unchanged):

- jaxtyping shape annotations removed (torchlens does not depend on jaxtyping);
  the shapes they documented live in the docstrings instead.
- The upstream utility helpers it imported (``transpose``, ``get_corner``,
  ``to_python_int``) are inlined.
- The deprecated ``Vh`` alias is dropped (torchlens ships no deprecation
  aliases; use ``V``), and the ``TensorLike`` duck-type protocol is dropped
  (torchlens passes real tensors).
- Shape-mismatch ``assert`` statements raise :class:`MechInterpError` with a
  stable code, so they survive ``python -O`` and teach a remedy.

The QK and OV circuit constructors that feed this class from verified
head-indexed weights are :func:`torchlens.mechinterp.qk_circuit` and
:func:`torchlens.mechinterp.ov_circuit`.

Spellings DOCUMENTED-UNSTABLE pending the naming session.

MIT License

Copyright (c) 2022 TransformerLensOrg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import numbers
from functools import cached_property
from typing import Any

import numpy as np
import torch

from ._errors import refuse

__all__ = ["FactoredMatrix"]


def _transpose(tensor: torch.Tensor) -> torch.Tensor:
    """Swap the last two dimensions, whatever the leading dimensions."""

    return tensor.transpose(-1, -2)


def _to_python_int(value: Any) -> int | None:
    """Return the Python int for a non-bool integer scalar, else ``None``."""

    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, torch.Tensor):
        is_int_dtype = (
            not value.dtype.is_floating_point
            and not value.dtype.is_complex
            and value.dtype != torch.bool
        )
        if value.ndim == 0 and is_int_dtype:
            return int(value.item())
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0 and np.issubdtype(value.dtype, np.integer):
        return int(value.item())
    return None


def _shape_refusal(message: str, **payload: Any) -> None:
    """Refuse a factor-shape mismatch with the contracted code."""

    refuse(
        code="mi_factored_shape_mismatch",
        message=message,
        remedy="match the inner dimensions (A[..., l, m] with B[..., m, r]) and keep the "
        "leading dimensions broadcastable",
        **payload,
    )


def _index_ndim(item: Any) -> int:
    """Return how many input axes one index item consumes (PyTorch rules)."""

    if item is None or item is Ellipsis:
        return 0
    if isinstance(item, (list, tuple)):
        sequence = torch.as_tensor(item, device="cpu")
        return sequence.ndim if sequence.dtype == torch.bool else 1
    if isinstance(item, torch.Tensor) and item.dtype in (torch.bool, torch.uint8):
        return item.ndim
    if isinstance(item, np.ndarray) and item.dtype in (np.bool_, np.uint8):
        return item.ndim
    return 1


class FactoredMatrix:
    """A matrix represented as the product ``A @ B`` of two factors.

    ``A`` is ``[..., ldim, mdim]`` and ``B`` is ``[..., mdim, rdim]``; leading
    dimensions broadcast. Products, SVD, norm and eigenvalues are computed
    without forming the full ``[..., ldim, rdim]`` product where possible.
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """Construct a factored matrix from factors ``A`` and ``B``.

        Parameters
        ----------
        A:
            Left factor, ``[..., ldim, mdim]``.
        B:
            Right factor, ``[..., mdim, rdim]``.
        """

        self.A: torch.Tensor = A
        self.B: torch.Tensor = B
        if self.A.size(-1) != self.B.size(-2):
            _shape_refusal(
                "Factored matrix must match on inner dimension, shapes were "
                f"a: {tuple(self.A.shape)}, b: {tuple(self.B.shape)}",
                a_shape=list(self.A.shape),
                b_shape=list(self.B.shape),
            )
        self.ldim = self.A.size(-2)
        self.rdim = self.B.size(-1)
        self.mdim = self.B.size(-2)
        self.has_leading_dims = (self.A.ndim > 2) or (self.B.ndim > 2)
        try:
            self.shape = torch.broadcast_shapes(self.A.shape[:-2], self.B.shape[:-2]) + (
                self.ldim,
                self.rdim,
            )
        except RuntimeError as exc:
            _shape_refusal(
                "Cannot broadcast leading dimensions: A has shape "
                f"{tuple(self.A.shape)}, B has shape {tuple(self.B.shape)} ({exc})",
                a_shape=list(self.A.shape),
                b_shape=list(self.B.shape),
            )
        self.A = self.A.broadcast_to(self.shape[:-2] + (self.ldim, self.mdim))
        self.B = self.B.broadcast_to(self.shape[:-2] + (self.mdim, self.rdim))

    def __matmul__(self, other: torch.Tensor | FactoredMatrix) -> Any:
        """Right-multiply by a tensor or another factored matrix.

        A vector (``ndim < 2``) collapses the factorisation and returns a
        vector; a matrix returns a :class:`FactoredMatrix` keeping the smaller
        hidden dimension.
        """

        if isinstance(other, FactoredMatrix):
            return (self @ other.A) @ other.B
        if other.ndim < 2:
            # A vector: collapse the factorisation; the squeeze/unsqueeze keeps
            # broadcasting working.
            return (self.A @ (self.B @ other.unsqueeze(-1))).squeeze(-1)
        if other.size(-2) != self.rdim:
            _shape_refusal(
                "Right matrix must match on inner dimension, shapes were "
                f"self: {tuple(self.shape)}, other: {tuple(other.shape)}",
                self_shape=list(self.shape),
                other_shape=list(other.shape),
            )
        if self.rdim > self.mdim:
            return FactoredMatrix(self.A, self.B @ other)
        return FactoredMatrix(self.AB, other)

    def __rmatmul__(self, other: torch.Tensor | FactoredMatrix) -> Any:
        """Left-multiply by a tensor or another factored matrix."""

        if isinstance(other, FactoredMatrix):
            return other.A @ (other.B @ self)
        if other.size(-1) != self.ldim:
            _shape_refusal(
                "Left matrix must match on inner dimension, shapes were "
                f"self: {tuple(self.shape)}, other: {tuple(other.shape)}",
                self_shape=list(self.shape),
                other_shape=list(other.shape),
            )
        if other.ndim < 2:
            # A vector: collapse the factorisation and return a vector.
            return ((other.unsqueeze(-2) @ self.A) @ self.B).squeeze(-2)
        if self.ldim > self.mdim:
            return FactoredMatrix(other @ self.A, self.B)
        return FactoredMatrix(other, self.AB)

    def __mul__(self, scalar: int | float | torch.Tensor) -> FactoredMatrix:
        """Scalar multiplication (distributes onto the left factor)."""

        if isinstance(scalar, torch.Tensor) and scalar.numel() != 1:
            refuse(
                code="mi_factored_scalar_invalid",
                message="Tensor must be a scalar for use with * but was of shape "
                f"{tuple(scalar.shape)}.",
                remedy="use @ for matrix multiplication; * takes a scalar",
                scalar_shape=list(scalar.shape),
            )
        return FactoredMatrix(self.A * scalar, self.B)

    def __rmul__(self, scalar: int | float | torch.Tensor) -> FactoredMatrix:
        """Scalar multiplication from the left (same as :meth:`__mul__`)."""

        return self * scalar

    @property
    def AB(self) -> torch.Tensor:
        """The full product ``[..., ldim, rdim]`` (expensive in memory)."""

        return self.A @ self.B

    @property
    def BA(self) -> torch.Tensor:
        """The reverse product ``[..., rdim, ldim]``; needs ``ldim == rdim``."""

        if self.rdim != self.ldim:
            _shape_refusal(
                f"Can only take BA if ldim == rdim, shape was {tuple(self.shape)}",
                self_shape=list(self.shape),
            )
        return self.B @ self.A

    @property
    def T(self) -> FactoredMatrix:
        """The transpose, as a factored matrix."""

        return FactoredMatrix(_transpose(self.B), _transpose(self.A))

    @cached_property
    def _svd_cached(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the SVD once per instance (freed with the instance)."""

        Ua, Sa, Vha = torch.linalg.svd(self.A, full_matrices=False)
        Ub, Sb, Vhb = torch.linalg.svd(self.B, full_matrices=False)
        Va = _transpose(Vha)
        Vb = _transpose(Vhb)
        middle = Sa[..., :, None] * _transpose(Va) @ Ub * Sb[..., None, :]
        Um, Sm, Vhm = torch.linalg.svd(middle, full_matrices=False)
        Vm = _transpose(Vhm)
        return Ua @ Um, Sm, Vb @ Vm

    def svd(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(U, S, V)`` with ``U @ S.diag() @ V.transpose(-2, -1) == AB``.

        Shapes: ``U`` ``[..., ldim, mdim]``, ``S`` ``[..., mdim]``, ``V``
        ``[..., rdim, mdim]``.
        """

        return self._svd_cached

    @property
    def U(self) -> torch.Tensor:
        """Left singular vectors ``[..., ldim, mdim]``."""

        return self.svd()[0]

    @property
    def S(self) -> torch.Tensor:
        """Singular values ``[..., mdim]``."""

        return self.svd()[1]

    @property
    def V(self) -> torch.Tensor:
        """Right singular vectors ``[..., rdim, mdim]`` (``AB == U S V^T``)."""

        return self.svd()[2]

    @property
    def eigenvalues(self) -> torch.Tensor:
        """Eigenvalues of ``AB``, computed from ``BA`` (equal up to trailing zeros).

        If ``BA v = k v`` then ``AB (A v) = k (A v)``, so ``A v`` is an
        eigenvector of ``AB`` with eigenvalue ``k``.
        """

        input_matrix = self.BA
        if input_matrix.dtype in (torch.bfloat16, torch.float16):
            # eig is not implemented for 16-bit dtypes on CPU or CUDA.
            input_matrix = input_matrix.to(torch.float32)
        return torch.linalg.eig(input_matrix).eigenvalues

    def _convert_to_slice(self, sequence: tuple[Any, ...], idx: int) -> tuple[Any, ...]:
        """Turn the int entry at ``idx`` into a length-1 slice (keeps the axis)."""

        entry = sequence[idx]
        # bool keeps its legacy int-index treatment, so check plain int first.
        value = int(entry) if isinstance(entry, int) else _to_python_int(entry)
        if value is None:
            return sequence
        # value + 1 selects one element, except at -1 where slice(-1, 0) is empty.
        stop = value + 1 if value != -1 else None
        converted = list(sequence)
        converted[idx] = slice(value, stop)
        return tuple(converted)

    def __getitem__(self, idx: Any) -> FactoredMatrix:
        """Index leading dimensions and rows/columns without forming the product.

        Ellipsis expands across the product's dimensions (new axes excluded);
        integer row and column indices keep singleton dimensions in the
        returned factors; new axes next to the matrix axes reshape the factors.
        """

        if not isinstance(idx, tuple):
            idx = (idx,)
        ellipsis_positions = [i for i, item in enumerate(idx) if item is Ellipsis]
        if len(ellipsis_positions) > 1:
            raise IndexError("An index can only have a single ellipsis")
        if ellipsis_positions:
            consumed = sum(_index_ndim(item) for item in idx)
            position = ellipsis_positions[0]
            # Expand against AB's axes before mapping row/column indices onto factors.
            idx = idx[:position] + (slice(None),) * (self.ndim - consumed) + idx[position + 1 :]
        factor_indices = []
        matrix_newaxes = []
        length = 0
        for item in idx:
            if item is None and length >= self.ndim - 2:
                matrix_newaxes.append(length - (self.ndim - 2))
            else:
                factor_indices.append(item)
            length += _index_ndim(item)
        idx = tuple(factor_indices)
        result = self._index_factors(idx, length)
        for position in matrix_newaxes:
            result = _insert_matrix_newaxis(result, position)
        return result

    def _index_factors(self, idx: tuple[Any, ...], length: int) -> FactoredMatrix:
        """Apply a newaxis-free index of ``length`` consumed axes to both factors."""

        if length <= len(self.shape) - 2:
            return FactoredMatrix(self.A[idx], self.B[idx])
        if length == len(self.shape) - 1:
            idx = self._convert_to_slice(idx, -1)
            return FactoredMatrix(self.A[idx], self.B[idx[:-1]])
        if length == len(self.shape):
            idx = self._convert_to_slice(idx, -1)
            idx = self._convert_to_slice(idx, -2)
            return FactoredMatrix(self.A[idx[:-1]], self.B[idx[:-2] + (slice(None), idx[-1])])
        raise ValueError(f"{idx} is too long an index for a FactoredMatrix with shape {self.shape}")

    def norm(self) -> torch.Tensor:
        """Frobenius norm: the root of the summed squared singular values."""

        return self.S.pow(2).sum(-1).sqrt()

    def __repr__(self) -> str:
        """Return the shape and hidden dimension."""

        return f"FactoredMatrix: Shape({self.shape}), Hidden Dim({self.mdim})"

    def make_even(self) -> FactoredMatrix:
        """Return the balanced factorisation ``(U sqrt(S), sqrt(S) V^T)``.

        The same matrix, with each half carrying half the singular values and
        orthogonal rows/columns.
        """

        return FactoredMatrix(
            self.U * self.S.sqrt()[..., None, :],
            self.S.sqrt()[..., :, None] * _transpose(self.V),
        )

    def corner(self, k: int = 3) -> torch.Tensor:
        """Return the top-left ``k x k`` corner of the product (and of leading dims)."""

        corner = self.A[..., :k, :] @ self.B[..., :, :k]
        return corner[tuple(slice(k) for _ in range(corner.ndim))]

    @property
    def ndim(self) -> int:
        """Number of dimensions of the product."""

        return len(self.shape)

    def collapse_l(self) -> torch.Tensor:
        """Drop the left orthogonal factor ``U``: returns ``S V^T`` ``[..., mdim, rdim]``."""

        return self.S[..., :, None] * _transpose(self.V)

    def collapse_r(self) -> torch.Tensor:
        """Drop the right orthogonal factor: returns ``U S`` ``[..., ldim, mdim]``."""

        return self.U * self.S[..., None, :]

    def unsqueeze(self, k: int) -> FactoredMatrix:
        """Insert a leading dimension at ``k`` in both factors."""

        return FactoredMatrix(self.A.unsqueeze(k), self.B.unsqueeze(k))

    @property
    def pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The factors ``(A, B)``."""

        return (self.A, self.B)


def _insert_matrix_newaxis(result: FactoredMatrix, position: int) -> FactoredMatrix:
    """Insert one new axis beside the matrix axes without forming the product."""

    if position == 0:
        axis = result.ndim - 2
        return FactoredMatrix(result.A.unsqueeze(axis), result.B.unsqueeze(axis))
    if position == 1:
        # Promote the row axis to a leading dimension, keeping the contraction intact.
        return FactoredMatrix(result.A.unsqueeze(-2), result.B.unsqueeze(-3))
    # A trailing axis makes the old column axis the new row axis.
    return FactoredMatrix(_transpose(result.B).unsqueeze(-3), result.A.unsqueeze(-1))
