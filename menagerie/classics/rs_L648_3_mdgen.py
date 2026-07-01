from __future__ import annotations

# FAITHFUL PORT of bjing2016/mdgen @ master (original framework: PyTorch, with two
# not-reasonably-installable dependencies: `esm` (fair-esm, for `esm.rotary_embedding
# .RotaryEmbedding`) and a commented-out (dead) `openfold` import in ipa.py -- see below)
#
# Actually this file is a hybrid: most of it is REAL vendored repo code (rung 2), with two
# small self-contained pieces individually PORTED (rung-3-style transcription, not
# reimplementation) because they live in an external package rather than the mdgen repo
# itself:
#   - `RotaryEmbedding` (14 lines, transcribed verbatim from facebookresearch/esm @ main,
#     esm/rotary_embedding.py -- pure torch, no other ESM dependency) replaces
#     `from esm.rotary_embedding import RotaryEmbedding` in mha.py, since `esm` (fair-esm)
#     is not in the installed base-lib set.
#   - `Linear` / `ipa_point_weights_init_` / the init-helper functions (transcribed
#     verbatim from openfold's primitives.py, which mdgen vendors at
#     mdgen/model/primitives.py -- itself only depends on torch/numpy/scipy) replace
#     `from .primitives import Linear, ipa_point_weights_init_` in ipa.py. The
#     `deepspeed`/`flash_attn` optional-acceleration branches in the original `Linear`
#     are feature-detected (`importlib.util.find_spec`) and both packages are absent
#     here, so `deepspeed_is_installed = False` faithfully matches that runtime state.
#
# Everything else below is REAL vendored MDGen repo code (imports/relative-paths adjusted
# only; the `openfold.utils.feats` import in the original mdgen/model/ipa.py is itself
# dead code inside a `"""..."""` docstring block, never executed, and is dropped here):
#   mdgen/rigid_utils.py    (Rotation, Rigid -- full file, self-contained torch/numpy
#                             quaternion + rotation-matrix rigid-transform math)
#   mdgen/tensor_utils.py   (permute_final_dims, flatten_final_dims)
#   mdgen/model/mha.py      (MultiheadAttention, AttentionWithRoPE's fairseq-derived
#                             multi-head attention -- the real repo's local fork)
#   mdgen/model/ipa.py      (InvariantPointAttention -- AlphaFold2-style IPA)
#   mdgen/model/layers.py   (modulate, TimestepEmbedder, FinalLayer, gelu -- only the
#                             pieces LatentMDGenModel actually uses; the file's unused
#                             StructureModuleTransition*/Attention/SequenceToPair/
#                             PairToSequence/ResidueMLP classes are dropped for brevity,
#                             not because they're unusable)
#   mdgen/model/latent_model.py  (LatentMDGenModel, IPALayer, LatentMDGenLayer,
#                                  AttentionWithRoPE -- the real top-level "MDGen" latent
#                                  diffusion transformer used by the repo's own train.py /
#                                  sim_inference.py via ..wrapper.Wrapper)
#
# MDGen (Jing, Stark, Berger, Jaakkola, 2025, "Generative Modeling of Molecular Dynamics
# Trajectories") is a latent diffusion transformer over per-frame protein backbone/
# torsion latents, interleaving IPA layers (structure-aware) with residue-axis and
# time-axis RoPE self-attention (DiT-style adaLN-modulated transformer blocks), trained
# with the `--prepend_ipa` config used by every real training command in the repo's
# README (sim_condition / tps_condition / inpainting+design). This staging module
# constructs `LatentMDGenModel` with that real config (`prepend_ipa=True, hyena=False,
# design=False`, matching the repo's actual `--sim_condition --prepend_ipa` training
# invocation) and exercises its real `.forward()` path.
import math
import uuid
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch import Tensor
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from esm/rotary_embedding.py (facebookresearch/esm @ main) -- pure torch,
# no other ESM dependency; `esm` (fair-esm) itself is not in the installed base-lib set.
# ---------------------------------------------------------------------------
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings from RoFormer (Su et al.)."""

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            _apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            _apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


# ---------------------------------------------------------------------------
# Vendored from mdgen/rigid_utils.py (Rotation, Rigid -- full file)
# ---------------------------------------------------------------------------
def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors. Written
    out by hand to avoid AMP downcasting.

    Args:
        a: [*, 3, 3] left multiplicand
        b: [*, 3, 3] right multiplicand
    Returns:
        The product ab
    """

    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


@lru_cache(maxsize=None)
def identity_rot_mats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(
    rot: torch.Tensor,
):
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

_CACHED_QUATS = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC,
}


@lru_cache(maxsize=None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)


def quat_multiply(quat1, quat2):
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY", dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2)
    )


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2)
    )


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv


class Rotation:
    """
    A 3D rotation. Depending on how the object is initialized, the
    rotation is represented by either a rotation matrix or a
    quaternion, though both formats are made available by helper functions.
    To simplify gradient computation, the underlying format of the
    rotation cannot be changed in-place. Like Rigid, the class is designed
    to mimic the behavior of a torch Tensor, almost as if each Rotation
    object were a tensor of rotations, in one format or another.
    """

    def __init__(
        self,
        rot_mats: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        normalize_quats: bool = True,
    ):
        """
        Args:
            rot_mats:
                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with
                quats
            quats:
                A [*, 4] quaternion. Mutually exclusive with rot_mats. If
                normalize_quats is not True, must be a unit quaternion
            normalize_quats:
                If quats is specified, whether to normalize quats
        """
        if (rot_mats is None and quats is None) or (rot_mats is not None and quats is not None):
            raise ValueError("Exactly one input argument must be specified")

        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (
            quats is not None and quats.shape[-1] != 4
        ):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # Force full-precision
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rotation:
        """
        Returns an identity Rotation.

        Args:
            shape:
                The "shape" of the resulting Rotation object. See documentation
                for the shape property
            dtype:
                The torch dtype for the rotation
            device:
                The torch device for the new rotation
            requires_grad:
                Whether the underlying tensors in the new rotation object
                should require gradient computation
            fmt:
                One of "quat" or "rot_mat". Determines the underlying format
                of the new object's rotation
        Returns:
            A new identity rotation
        """
        if fmt == "rot_mat":
            rot_mats = identity_rot_mats(
                shape,
                dtype,
                device,
                requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == "quat":
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods

    def __getitem__(self, index: Any) -> Rotation:
        """
        Allows torch-style indexing over the virtual shape of the rotation
        object. See documentation for the shape property.

        Args:
            index:
                A torch index. E.g. (1, 3, 2), or (slice(None,))
        Returns:
            The indexed rotation
        """
        if type(index) != tuple:
            index = (index,)

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __mul__(
        self,
        right: torch.Tensor,
    ) -> Rotation:
        """
        Pointwise left multiplication of the rotation with a tensor. Can be
        used to e.g. mask the Rotation.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(
        self,
        left: torch.Tensor,
    ) -> Rotation:
        """
        Reverse pointwise multiplication of the rotation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    # Properties

    @property
    def shape(self) -> torch.Size:
        """
        Returns the virtual shape of the rotation object. This shape is
        defined as the batch dimensions of the underlying rotation matrix
        or quaternion. If the Rotation was initialized with a [10, 3, 3]
        rotation matrix tensor, for example, the resulting shape would be
        [10].

        Returns:
            The virtual shape of the rotation object
        """
        s = None
        if self._quats is not None:
            s = self._quats.shape[:-1]
        else:
            s = self._rot_mats.shape[:-2]

        return s

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the underlying rotation.

        Returns:
            The dtype of the underlying rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        elif self._quats is not None:
            return self._quats.dtype
        else:
            raise ValueError("Both rotations are None")

    @property
    def device(self) -> torch.device:
        """
        The device of the underlying rotation

        Returns:
            The device of the underlying rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats.device
        elif self._quats is not None:
            return self._quats.device
        else:
            raise ValueError("Both rotations are None")

    @property
    def requires_grad(self) -> bool:
        """
        Returns the requires_grad property of the underlying rotation

        Returns:
            The requires_grad property of the underlying tensor
        """
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        elif self._quats is not None:
            return self._quats.requires_grad
        else:
            raise ValueError("Both rotations are None")

    def get_rot_mats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
        rot_mats = self._rot_mats
        if rot_mats is None:
            if self._quats is None:
                raise ValueError("Both rotations are None")
            else:
                rot_mats = quat_to_rot(self._quats)

        return rot_mats

    def get_quats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a quaternion tensor.

        Depending on whether the Rotation was initialized with a
        quaternion, this function may call torch.linalg.eigh.

        Returns:
            The rotation as a quaternion tensor.
        """
        quats = self._quats
        if quats is None:
            if self._rot_mats is None:
                raise ValueError("Both rotations are None")
            else:
                quats = rot_to_quat(self._rot_mats)

        return quats

    def get_cur_rot(self) -> torch.Tensor:
        """
        Return the underlying rotation in its current form

        Returns:
            The stored rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    # Rotation functions

    def compose_q_update_vec(
        self, q_update_vec: torch.Tensor, normalize_quats: bool = True
    ) -> Rotation:
        """
        Returns a new quaternion Rotation after updating the current
        object's underlying rotation with a quaternion update, formatted
        as a [*, 3] tensor whose final three columns represent x, y, z such
        that (1, x, y, z) is the desired (not necessarily unit) quaternion
        update.

        Args:
            q_update_vec:
                A [*, 3] quaternion update tensor
            normalize_quats:
                Whether to normalize the output quaternion
        Returns:
            An updated Rotation
        """
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: Rotation) -> Rotation:
        """
        Compose the rotation matrices of the current Rotation object with
        those of another.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = rot_matmul(r1, r2)
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: Rotation, normalize_quats: bool = True) -> Rotation:
        """
        Compose the quaternions of the current Rotation object with those
        of another.

        Depending on whether either Rotation was initialized with
        quaternions, this function may call torch.linalg.eigh.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quat_multiply(q1, q2)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply the current Rotation as a rotation matrix to a set of 3D
        coordinates.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] rotated points
        """
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
        """
        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats)
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self) -> Rotation:
        """
        Returns the inverse of the current Rotation.

        Returns:
            The inverse of the current Rotation
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    def unsqueeze(
        self,
        dim: int,
    ) -> Rigid:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the
        shape of the Rotation object.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed Rotation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @staticmethod
    def cat(
        rs: Sequence[Rotation],
        dim: int,
    ) -> Rigid:
        """
        Concatenates rotations along one of the batch dimensions. Analogous
        to torch.cat().

        Note that the output of this operation is always a rotation matrix,
        regardless of the format of input rotations.

        Args:
            rs:
                A list of rotation objects
            dim:
                The dimension along which the rotations should be
                concatenated
        Returns:
            A concatenated Rotation object in rotation matrix format
        """
        rot_mats = [r.get_rot_mats() for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(rot_mats=rot_mats, quats=None)

    def map_tensor_fn(self, fn: Callable[torch.Tensor, torch.Tensor]) -> Rotation:
        """
        Apply a Tensor -> Tensor function to underlying rotation tensors,
        mapping over the rotation dimension(s). Can be used e.g. to sum out
        a one-hot batch dimension.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rotation
        Returns:
            The transformed Rotation object
        """
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def cuda(self) -> Rotation:
        """
        Analogous to the cuda() method of torch Tensors

        Returns:
            A copy of the Rotation in CUDA memory
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Rotation:
        """
        Analogous to the to() method of torch Tensors

        Args:
            device:
                A torch device
            dtype:
                A torch dtype
        Returns:
            A copy of the Rotation using the new device and dtype
        """
        if self._rot_mats is not None:
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def detach(self) -> Rotation:
        """
        Returns a copy of the Rotation whose underlying Tensor has been
        detached from its torch graph.

        Returns:
            A copy of the Rotation whose underlying Tensor has been detached
            from its torch graph
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")


class Rigid:
    """
    A class representing a rigid transformation. Little more than a wrapper
    around two objects: a Rotation object and a [*, 3] translation
    Designed to behave approximately like a single torch tensor with the
    shape of the shared batch dimensions of its component parts.
    """

    def __init__(
        self,
        rots: Optional[Rotation],
        trans: Optional[torch.Tensor],
    ):
        """
        Args:
            rots: A [*, 3, 3] rotation tensor
            trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.to(dtype=torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rigid:
        """
        Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(
        self,
        index: Any,
    ) -> Rigid:
        """
        Indexes the affine transformation with PyTorch-style indices.
        The index is applied to the shared dimensions of both the rotation
        and the translation.

        E.g.::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
            t = Rigid(r, torch.rand(10, 10, 3))
            indexed = t[3, 4:6]
            assert(indexed.shape == (2,))
            assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: A standard torch tensor index. E.g. 8, (10, None, 3),
            or (3, slice(0, 1, None))
        Returns:
            The indexed tensor
        """
        if type(index) != tuple:
            index = (index,)

        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    def __mul__(
        self,
        right: torch.Tensor,
    ) -> Rigid:
        """
        Pointwise left multiplication of the transformation with a tensor.
        Can be used to e.g. mask the Rigid.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(
        self,
        left: torch.Tensor,
    ) -> Rigid:
        """
        Reverse pointwise multiplication of the transformation with a
        tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the shared dimensions of the rotation and
        the translation.

        Returns:
            The shape of the transformation
        """
        s = self._trans.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Rigid's tensors are located.

        Returns:
            The device on which the Rigid's tensors are located
        """
        return self._trans.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the Rigid tensors.

        Returns:
            The dtype of the Rigid tensors
        """
        return self._rots.dtype

    def get_rots(self) -> Rotation:
        """
        Getter for the rotation.

        Returns:
            The rotation object
        """
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """
        Getter for the translation.

        Returns:
            The stored translation
        """
        return self._trans

    def compose_q_update_vec(
        self,
        q_update_vec: torch.Tensor,
    ) -> Rigid:
        """
        Composes the transformation with a quaternion update vector of
        shape [*, 6], where the final 6 columns represent the x, y, and
        z values of a quaternion of form (1, x, y, z) followed by a 3D
        translation.

        Args:
            q_vec: The quaternion update vector.
        Returns:
            The composed transformation.
        """
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self._rots.compose_q_update_vec(q_vec)

        trans_update = self._rots.apply(t_vec)
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose(
        self,
        r: Rigid,
    ) -> Rigid:
        """
        Composes the current rigid object with another.

        Args:
            r:
                Another Rigid object
        Returns:
            The composition of the two transformations
        """
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor.
        Returns:
            The transformed points.
        """
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> Rigid:
        """
        Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[torch.Tensor, torch.Tensor]) -> Rigid:
        """
        Apply a Tensor -> Tensor function to underlying translation and
        rotation tensors, mapping over the translation/rotation dimensions
        respectively.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rigid
        Returns:
            The transformed Rigid object
        """
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)

        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        """
        Converts a transformation to a homogenous transformation tensor.

        Returns:
            A [*, 4, 4] homogenous transformation tensor
        """
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Rigid:
        """
        Constructs a transformation from a homogenous transformation
        tensor.

        Args:
            t: [*, 4, 4] homogenous transformation tensor
        Returns:
            T object with shape [*]
        """
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]

        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        """
        Converts a transformation to a tensor with 7 final columns, four
        for the quaternion followed by three for the translation.

        Returns:
            A [*, 7] tensor representation of the transformation
        """
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(
        t: torch.Tensor,
        normalize_quats: bool = False,
    ) -> Rigid:
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)

        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8,
    ) -> Rigid:
        """
        Implements algorithm 21. Constructs transformations from sets of 3
        points using the Gram-Schmidt algorithm.

        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates used as frame origins
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon value
        Returns:
            A transformation object of shape [*]
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(
        self,
        dim: int,
    ) -> Rigid:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the
        shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(
        ts: Sequence[Rigid],
        dim: int,
    ) -> Rigid:
        """
        Concatenates transformations along a new dimension.

        Args:
            ts:
                A list of T objects
            dim:
                The dimension along which the transformations should be
                concatenated
        Returns:
            A concatenated transformation object
        """
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable[Rotation, Rotation]) -> Rigid:
        """
        Applies a Rotation -> Rotation function to the stored rotation
        object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable[torch.Tensor, torch.Tensor]) -> Rigid:
        """
        Applies a Tensor -> Tensor function to the stored translation.

        Args:
            fn:
                A function of type Tensor -> Tensor to be applied to the
                translation
        Returns:
            A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        """
        Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """

        def fn(t):
            return t * trans_scale_factor

        return self.apply_trans_fn(fn)

    def stop_rot_gradient(self) -> Rigid:
        """
        Detaches the underlying rotation object

        Returns:
            A transformation object with detached rotations
        """

        def fn(r):
            return r.detach()

        return self.apply_rot_fn(fn)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        """
        Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you
        provide the atom positions in the non-standard way, the N atom will
        end up not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care of such cases in
        your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)  # noqa: F841 -- unused in original source too
        ones = sin_c1.new_ones(sin_c1.shape)  # noqa: F841 -- unused in original source too

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        """
        Moves the transformation object to GPU memory

        Returns:
            A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())


# ---------------------------------------------------------------------------
# Vendored from mdgen/tensor_utils.py (permute_final_dims, flatten_final_dims only)
# ---------------------------------------------------------------------------
def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t, no_dims):
    return t.reshape(t.shape[:-no_dims] + (-1,))


# ---------------------------------------------------------------------------
# Vendored from openfold's primitives.py (via mdgen/model/primitives.py):
# Linear / ipa_point_weights_init_ / init helpers only
# ---------------------------------------------------------------------------
deepspeed_is_installed = False


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        precision=None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        deepspeed_is_initialized = (
            deepspeed_is_installed and deepspeed.comm.comm.is_initialized()  # noqa: F821 -- short-circuited, deepspeed_is_installed=False
        )
        if self.precision is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                return nn.functional.linear(
                    input.to(dtype=self.precision), self.weight.to(dtype=self.precision), bias
                ).to(dtype=d)

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return nn.functional.linear(input, self.weight.to(dtype=d), bias)

        return nn.functional.linear(input, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Vendored from mdgen/model/mha.py (fairseq-derived MultiheadAttention fork)
# ---------------------------------------------------------------------------
def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.batch_first = True
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            not self.rot_emb
            and self.enable_torch_version
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            and not need_head_weights
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = (
                attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len)
                .type_as(attn)
                .transpose(1, 0)
            )
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(
                        0
                    ):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


# ---------------------------------------------------------------------------
# Vendored from mdgen/model/ipa.py (InvariantPointAttention)
# ---------------------------------------------------------------------------
class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        inf=1e5,
        eps=1e-8,
        dropout=0.0,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.dropout = dropout
        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3  # noqa: F841 -- unused in original source too

        if self.c_z > 0:
            self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (self.c_z + self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, s, r, z=None, frame_mask=None, attn_mask=None):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        if self.c_z > 0:
            b = self.linear_b(z[0])

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        if self.c_z:
            a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]

        if frame_mask is not None:
            square_mask = frame_mask.unsqueeze(-1) * frame_mask.unsqueeze(-2)
            square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        if frame_mask is not None:
            a = a + square_mask.unsqueeze(-3)
        if attn_mask is not None:
            attn_mask = self.inf * (attn_mask - 1)
            a = a + attn_mask

        a = self.softmax(a)
        a = F.dropout(a, p=self.dropout, training=self.training)
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        if self.c_z > 0:
            o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

            # [*, N_res, H * C_z]
            o_pair = flatten_final_dims(o_pair, 2)

            # [*, N_res, C_s]
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(
                    dtype=z[0].dtype
                )
            )
        else:
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(dtype=s.dtype)
            )
        return s


# ---------------------------------------------------------------------------
# Vendored from mdgen/model/layers.py (modulate, TimestepEmbedder, FinalLayer, gelu)
# ---------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Vendored from mdgen/model/latent_model.py (LatentMDGenModel, IPALayer,
# LatentMDGenLayer, AttentionWithRoPE -- the real top-level model)
# ---------------------------------------------------------------------------
def grad_checkpoint(func, args, checkpointing=False):
    if checkpointing:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(func, *args, use_reentrant=False)
    else:
        return func(*args)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LatentMDGenModel(nn.Module):
    def __init__(self, args, latent_dim):
        super().__init__()
        self.args = args
        if self.args.design:
            assert self.args.prepend_ipa

        self.latent_to_emb = nn.Linear(latent_dim, args.embed_dim)
        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            self.latent_to_emb_f = nn.Linear(7, args.embed_dim)
            self.latent_to_emb_r = nn.Linear(7, args.embed_dim)

        cond_dim = latent_dim
        if self.args.design:
            cond_dim -= 20
        self.cond_to_emb = nn.Linear(cond_dim, args.embed_dim)
        self.mask_to_emb = nn.Embedding(2, args.embed_dim)
        if self.args.design:
            self.x_d_to_emb = nn.Linear(20, args.embed_dim)

        ipa_args = {
            "c_s": args.embed_dim,
            "c_z": 0,
            "c_hidden": args.ipa_head_dim,
            "no_heads": args.ipa_heads,
            "no_qk_points": args.ipa_qk,
            "no_v_points": args.ipa_v,
            "dropout": args.dropout,
        }

        if args.prepend_ipa:
            if not self.args.no_aa_emb:
                self.aatype_to_emb = nn.Embedding(21, args.embed_dim)
            self.ipa_layers = nn.ModuleList(
                [
                    IPALayer(
                        embed_dim=args.embed_dim,
                        ffn_embed_dim=4 * args.embed_dim,
                        mha_heads=args.mha_heads,
                        dropout=args.dropout,
                        use_rotary_embeddings=not args.no_rope,
                        ipa_args=ipa_args,
                    )
                    for _ in range(args.num_layers)
                ]
            )

        self.layers = nn.ModuleList(
            [
                LatentMDGenLayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    dropout=args.dropout,
                    hyena=args.hyena,
                    num_frames=args.num_frames,
                    use_rotary_embeddings=not args.no_rope,
                    use_time_attention=True,
                    ipa_args=ipa_args if args.interleave_ipa else None,
                )
                for _ in range(args.num_layers)
            ]
        )

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            self.emb_to_latent = FinalLayer(args.embed_dim, latent_dim)
        if args.design:
            self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
            self.fc2 = nn.Linear(args.embed_dim, args.embed_dim)
            self.fc3 = nn.Linear(args.embed_dim, args.embed_dim)
            self.emb_to_logits = nn.Linear(args.embed_dim, 20)

        self.t_embedder = TimestepEmbedder(args.embed_dim)
        if args.abs_pos_emb:
            self.register_buffer(
                "pos_embed",
                nn.Parameter(torch.zeros(1, args.crop, args.embed_dim), requires_grad=False),
            )

        if args.abs_time_emb:
            self.register_buffer(
                "time_embed",
                nn.Parameter(torch.zeros(1, args.num_frames, args.embed_dim), requires_grad=False),
            )

        self.args = args
        if self.args.design:
            # DirichletConditionalFlow lives in mdgen/utils.py (design-only path, not
            # ported here; this build always constructs with design=False).
            self.condflow = DirichletConditionalFlow(  # noqa: F821
                K=20, alpha_spacing=0.001, alpha_max=args.alpha_max
            )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.args.prepend_ipa:
            for block in self.ipa_layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.interleave_ipa:
            for block in self.layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.args.abs_pos_emb:
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1], np.arange(self.args.crop)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.args.abs_time_emb:
            time_embed = get_1d_sincos_pos_embed_from_grid(
                self.time_embed.shape[-1], np.arange(self.args.num_frames)
            )
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            # Zero-out output layers:
            nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.emb_to_latent.linear.weight, 0)
            nn.init.constant_(self.emb_to_latent.linear.bias, 0)

    def run_ipa(self, t, mask, start_frames, end_frames, aatype, x_d=None):
        if self.args.sim_condition or self.args.mpnn:
            B, L = mask.shape
            x = torch.zeros(B, L, self.args.embed_dim, device=mask.device)
            if aatype is not None and not self.args.no_aa_emb:
                x = x + self.aatype_to_emb(aatype)
            if self.args.design:
                x = x + self.x_d_to_emb(x_d)  # pass in only the simplex data
            for layer in self.ipa_layers:
                x = layer(x, t, mask, frames=start_frames)
        elif self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            x_f = start_frames.invert().compose(end_frames).to_tensor_7()
            x_r = end_frames.invert().compose(start_frames).to_tensor_7()
            x_f = self.latent_to_emb_f(x_f)
            x_r = self.latent_to_emb_r(x_r)
            if aatype is not None and not self.args.no_aa_emb:
                x_f = x_f + self.aatype_to_emb(aatype)
                x_r = x_r + self.aatype_to_emb(aatype)
            if self.args.design:
                x_f = x_f + self.x_d_to_emb(x_d)
                x_r = x_r + self.x_d_to_emb(x_d)
            for layer in self.ipa_layers:
                x_r = layer(x_r, t, mask, frames=start_frames)
                x_f = layer(x_f, t, mask, frames=end_frames)
            x = x_r + x_f

        # x = x[:, None] + x_latent
        return x

    def forward(
        self,
        x,
        t,
        mask,
        start_frames=None,
        end_frames=None,
        x_cond=None,
        x_cond_mask=None,
        aatype=None,
    ):
        if self.args.dynamic_mpnn:
            x = x[:, [0, -1]]
            x_cond = x_cond[:, [0, -1]]
            x_cond_mask = x_cond_mask[:, [0, -1]]
            mask = mask[:, [0, -1]]
        if self.args.mpnn:
            x = x[:, :1]
            x_cond = x_cond[:, :1]
            x_cond_mask = x_cond_mask[:, :1]
            mask = mask[:, :1]

        if self.args.design:
            x_d = x[..., -20:].mean(1)
        else:
            x_d = None

        x = self.latent_to_emb(x)  # 384 dim token
        if self.args.abs_pos_emb:
            x = x + self.pos_embed

        if self.args.abs_time_emb:
            x = x + self.time_embed[:, :, None]

        if x_cond is not None:
            x = (
                x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)
            )  # token has cond g, tau

        t = self.t_embedder(t * self.args.time_multiplier)[:, None]

        if self.args.prepend_ipa:  # IPA doesn't need checkpointing
            x = (
                x
                + self.run_ipa(t[:, 0], mask[:, 0], start_frames, end_frames, aatype, x_d=x_d)[
                    :, None
                ]
            )

        for layer_idx, layer in enumerate(self.layers):
            x = grad_checkpoint(layer, (x, t, mask, start_frames), self.args.grad_checkpointing)

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            latent = self.emb_to_latent(x, t)
        if self.args.design:  ### this is also kind of weird
            x_l = self.fc2(gelu(self.fc1(x)))
            x_l = x_l.mean(1)
            logits = self.emb_to_logits(gelu(self.fc3(x_l)))
            if self.args.dynamic_mpnn or self.args.mpnn:
                return logits[:, None, :]
            latent[:, :, :, -20:] = latent[:, :, :, -20:] + logits[:, None, :, :]
        return latent

    # x, t, mask, start_frames=None, end_frames=None, x_cond=None, x_cond_mask=None, aatype=None
    def forward_inference(
        self,
        x,
        t,
        mask,
        start_frames=None,
        end_frames=None,
        x_cond=None,
        x_cond_mask=None,
        aatype=None,
    ):
        if not self.args.design or self.args.dynamic_mpnn or self.args.mpnn:
            return self.forward(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype)
        else:
            x_discrete = x[:, :, :, -20:]
            B, T, L, _ = x_discrete.shape
            if (
                not torch.allclose(
                    x_discrete.sum(3), torch.ones((B, T, L), device=x.device), atol=1e-4
                )
                or not (x_discrete >= 0).all()
            ):
                print(
                    f"WARNING: xt.min(): {x_discrete.min()}. Some values of xt do not lie on the simplex. There are "
                    f"we are "
                    f"{(x_discrete < 0).sum()} negative values in xt of shape {x_discrete.shape} that are negative. "
                    f"We are projecting "
                    f"them onto the simplex."
                )

                # x_discrete = simplex_proj(x_discrete)
            latent = self.forward(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype)
            latent_continuous = latent[:, :, :, :-20]
            logits = latent[:, :, :, -20:]

            flow_probs = torch.nn.functional.softmax(logits / self.args.dirichlet_flow_temp, -1)
            if (
                not torch.allclose(
                    flow_probs.sum(3), torch.ones((B, T, L), device=x.device), atol=1e-4
                )
                or not (flow_probs >= 0).all()
            ):
                print(
                    f"WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the "
                    f"simplex. There are we are {(flow_probs < 0).sum()} negative values in flow_probs of shape "
                    f"{flow_probs.shape} that are negative. We are projecting them onto the simplex."
                )
                flow_probs = simplex_proj(flow_probs)  # noqa: F821 -- design-only path (mdgen/utils.py), not ported

            alpha, dalpha_dt = t_to_alpha(t[0], self.args)  # noqa: F821 -- design-only path (mdgen/transport/transport.py), not ported
            alpha = alpha.item()

            if alpha > self.args.alpha_max:
                alpha = self.args.alpha_max - self.condflow.alpha_spacing
            c_factor = self.condflow.c_factor(x_discrete.cpu().numpy(), alpha)
            c_factor = torch.from_numpy(c_factor).to(x_discrete)
            if torch.isnan(c_factor).any():
                print(
                    f"NAN cfactor after: xt.min(): {x_discrete.min()}, flow_probs.min(): {flow_probs.min()}"
                )

                if self.args.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                else:
                    raise RuntimeError(
                        f"NAN cfactor after: xt.min(): {x_discrete.min()}, flow_probs.min(): {flow_probs.min()}"
                    )

            if not (flow_probs >= 0).all():
                print(f"flow_probs.min(): {flow_probs.min()}")
            eye = torch.eye(20).to(x_discrete)
            cond_flows = (eye - x_discrete.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1) * dalpha_dt

            return torch.cat([latent_continuous, flow], -1)


class AttentionWithRoPE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=1 - mask)
        x = x.transpose(0, 1)
        return x


class IPALayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        mha_heads,
        dropout=0.0,
        use_rotary_embeddings=False,
        ipa_args=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

        self.ipa_norm = nn.LayerNorm(self.embed_dim)
        self.ipa = InvariantPointAttention(**ipa_args)

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        shift_msa_l, scale_msa_l, gate_msa_l, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=-1)
        )
        x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(x, mask=mask)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


class LatentMDGenLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        mha_heads,
        dropout=0.0,
        num_frames=50,
        hyena=False,
        use_rotary_embeddings=False,
        use_time_attention=True,
        ipa_args=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hyena = hyena
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_time_attention = use_time_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.embed_dim, 9 * self.embed_dim, bias=True)
        )

        if ipa_args is not None:
            self.ipa_norm = nn.LayerNorm(self.embed_dim)
            self.ipa = InvariantPointAttention(**ipa_args)

        if self.hyena:
            # HyenaOperator lives in mdgen/model/standalone_hyena.py (hyena=True-only
            # path, not ported here; this build always constructs with hyena=False).
            self.mha_t = HyenaOperator(  # noqa: F821
                d_model=self.embed_dim,
                l_max=self.num_frames,
                order=2,
                filter_order=64,
            )

        else:
            self.mha_t = AttentionWithRoPE(
                self.embed_dim,
                self.mha_heads,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
                use_rotary_embeddings=self.use_rotary_embeddings,
            )

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        B, T, L, C = x.shape

        (
            shift_msa_l,
            scale_msa_l,
            gate_msa_l,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t).chunk(9, dim=-1)

        if hasattr(self, "ipa"):
            x = x + self.ipa(self.ipa_norm(x), frames[:, None], frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(
            x.reshape(B * T, L, C),
            mask=mask.reshape(B * T, L),  # [:,None].expand(-1, T, -1).reshape(B * T, L)
        ).reshape(B, T, L, C)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_t, scale_msa_t)
        if self.hyena:
            assert (mask - 1).sum() == 0
            x = (
                self.mha_t(x.transpose(1, 2).reshape(B * L, T, C))
                .reshape(B, L, T, C)
                .transpose(1, 2)
            )
        else:
            x = (
                self.mha_t(
                    x.transpose(1, 2).reshape(B * L, T, C),
                    mask=mask.transpose(1, 2).reshape(B * L, T),
                )
                .reshape(B, L, T, C)
                .transpose(1, 2)
            )
        x = residual + gate_msa_t.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


def build_mdgen():
    torch.manual_seed(0)

    class _Args:
        pass

    args = _Args()
    args.design = False
    args.prepend_ipa = True
    args.tps_condition = False
    args.inpainting = False
    args.dynamic_mpnn = False
    args.mpnn = False
    args.sim_condition = True
    args.embed_dim = 32
    args.mha_heads = 4
    args.ipa_head_dim = 8
    args.ipa_heads = 2
    args.ipa_qk = 2
    args.ipa_v = 2
    args.dropout = 0.0
    args.num_layers = 2
    args.hyena = False
    args.num_frames = 4
    args.no_rope = False
    args.interleave_ipa = False
    args.no_aa_emb = True
    args.abs_pos_emb = False
    args.abs_time_emb = False
    args.crop = 4
    args.grad_checkpointing = False
    args.time_multiplier = 1.0

    latent_dim = 12
    model = LatentMDGenModel(args, latent_dim)
    return model


def example_input_mdgen():
    """Tiny synthetic single-batch input matching the real field schema consumed by
    LatentMDGenModel.forward (x, t, mask, start_frames, end_frames), mirroring the
    real repo's wrapper.py construction of `rigids = Rigid(trans=batch['trans'],
    rots=Rotation(rot_mats=batch['rots']))` fed in as `start_frames`/`end_frames`
    under the `--sim_condition --prepend_ipa` config (B=1 batch, T=4 frames,
    L=4 residues, latent_dim=12)."""
    torch.manual_seed(0)
    B, T, L, latent_dim = 1, 4, 4, 12

    x = torch.randn(B, T, L, latent_dim)
    t = torch.rand(B)
    # mask is [B, T, L] (per-frame residue mask); real wrapper.py builds it via
    # `batch['mask'].unsqueeze(1).expand(-1, T, -1)` from a [B, L] per-residue mask.
    mask = torch.ones(B, T, L)

    rot_mats = torch.eye(3).reshape(1, 1, 3, 3).expand(B, L, 3, 3).contiguous()
    trans = torch.randn(B, L, 3)
    start_frames = Rigid(trans=trans, rots=Rotation(rot_mats=rot_mats))

    return (x, t, mask, start_frames)


MENAGERIE_ENTRIES = [
    ("MDGen", "build_mdgen", "example_input_mdgen", 2025, MENAGERIE_ZOO),
]
