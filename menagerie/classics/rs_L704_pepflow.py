# FAITHFUL PORT of Ced3-han/PepFlowww @ main (original framework: PyTorch; vendored
# verbatim from the real repo -- NOT a from-scratch reimplementation)
#
# PepFlow: "Full-Atom Peptide Design based on Multi-modal Flow Matching" (ICML 2024).
# https://github.com/Ced3-han/PepFlowww  (paper: https://arxiv.org/abs/2406.00735)
#
# This module vendors the REAL PepFlow encoder stack verbatim from the repo:
#   openfold/utils/rigid_utils.py          (AlQuraishi Lab / DeepMind, Apache-2.0; Rigid/Rotation)
#   pepflow/modules/protein/constants.py   (residue/atom geometry constants)
#   pepflow/modules/common/topology.py     (chain/terminus flags)
#   pepflow/modules/common/layers.py       (AngularEncoding etc.)
#   pepflow/modules/common/geometry.py     (construct_3d_basis, dihedral geometry)
#   models_con/utils.py                    (sinusoidal index/time embeddings)
#   models_con/edge.py                     (EdgeEmbedder)
#   models_con/node.py                     (NodeEmbedder)
#   models_con/ipa_pytorch.py              (Invariant Point Attention trunk, "Modified
#                                            code of Openfold's IPA")
#   models_con/ga.py                       (GAEncoder: the IPA + seq-transformer trunk
#                                            used by FlowModel.encode/forward)
#
# Rung-3 justification: the real repo's model files import `torch_scatter`
# (data/utils.py, data/all_atom.py) and a vendored `openfold` package for data
# pipelines/training, neither of which is installed in the base menagerie env and
# neither of which the ENCODER forward pass actually needs. Every architectural
# class/method below is transcribed byte-for-byte from the real source; the only
# edits are import-resolution (inlining cross-file references into this single
# module, since the files above are mutually-local imports within the real repo)
# and three renames done purely to avoid name collisions once flattened into one
# file: `ipa_pytorch.Linear` -> `IPALinear` (collides with torch.nn.Linear import
# style), the module-qualified `ipa_pytorch.X`/`du.X` call sites in GAEncoder are
# rewritten to plain `X` since everything now lives in one namespace. No layer,
# forward-pass equation, or initialization scheme was altered. Dead code that only
# existed to serve the (unneeded, torch_scatter-gated) `all_atom` data pipeline --
# the unused `compute_angles` helper in ipa_pytorch.py, never called by
# InvariantPointAttention.forward -- was dropped rather than vendoring all of
# data/all_atom.py just for one dead function.
#
# `du.create_rigid` (from the real data/utils.py, which itself imports torch_scatter
# at module level purely for unrelated helpers) is reproduced verbatim below -- it is
# a 2-line function that only touches `Rotation`/`Rigid`, with zero torch_scatter
# dependency of its own:
#   def create_rigid(rots, trans):
#       rots = ru.Rotation(rot_mats=rots)
#       return Rigid(rots=rots, trans=trans)
#
# ruff: noqa: E701, E702, E731, E741, F841 -- this file vendors the real repo's
# openfold/utils/rigid_utils.py, pepflow/modules/protein/constants.py, and
# models_con/ipa_pytorch.py verbatim; those files use one-line if/for bodies, `l`/`O`
# variable names matching AlQuraishi-lab/openfold conventions, and (in the vendored,
# unused-by-our-wrapper `IpaScore.forward`) an unused `init_rots` local -- all
# preserved unmodified for fidelity rather than reformatted.

import copy
import enum
import functools
import math
import os
import argparse
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.utils.data import DataLoader


MENAGERIE_ZOO = "ported-pytorch"


# ---- openfold/utils/rigid_utils.py (AlQuraishi Lab / DeepMind, Apache-2.0) ----
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    row_1 = torch.stack(
        [
            a[..., 0, 0] * b[..., 0, 0] + a[..., 0, 1] * b[..., 1, 0] + a[..., 0, 2] * b[..., 2, 0],
            a[..., 0, 0] * b[..., 0, 1] + a[..., 0, 1] * b[..., 1, 1] + a[..., 0, 2] * b[..., 2, 1],
            a[..., 0, 0] * b[..., 0, 2] + a[..., 0, 1] * b[..., 1, 2] + a[..., 0, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[..., 1, 0] * b[..., 0, 0] + a[..., 1, 1] * b[..., 1, 0] + a[..., 1, 2] * b[..., 2, 0],
            a[..., 1, 0] * b[..., 0, 1] + a[..., 1, 1] * b[..., 1, 1] + a[..., 1, 2] * b[..., 2, 1],
            a[..., 1, 0] * b[..., 0, 2] + a[..., 1, 1] * b[..., 1, 2] + a[..., 1, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[..., 2, 0] * b[..., 0, 0] + a[..., 2, 1] * b[..., 1, 0] + a[..., 2, 2] * b[..., 2, 0],
            a[..., 2, 0] * b[..., 0, 1] + a[..., 2, 1] * b[..., 1, 1] + a[..., 2, 2] * b[..., 2, 1],
            a[..., 2, 0] * b[..., 0, 2] + a[..., 2, 1] * b[..., 1, 2] + a[..., 2, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


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
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def identity_rot_mats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)

    return rots


def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


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
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)

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


def quat_multiply(quat1, quat2):
    """Multiply a quaternion by another quaternion."""
    mat = quat1.new_tensor(_QUAT_MULTIPLY)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2)
    )


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = quat.new_tensor(_QUAT_MULTIPLY_BY_VEC)
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
            quats = quats.type(torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.type(torch.float32)

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
    ):
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

    def __getitem__(self, index: Any):
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
    ):
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
    ):
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

    def get_rotvec(self, eps=1e-6) -> torch.Tensor:
        """
        Return the underlying axis-angle rotation vector.

        Follow's scipy's implementation:
        https://github.com/scipy/scipy/blob/HEAD/scipy/spatial/transform/_rotation.pyx#L1385-L1402

        Returns:
            The stored rotation as a axis-angle vector.
        """
        quat = self.get_quats()
        # w > 0 to ensure 0 <= angle <= pi
        flip = (quat[..., :1] < 0).float()
        quat = (-1 * quat) * flip + (1 - flip) * quat

        angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

        angle2 = angle * angle
        small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
        large_angle_scales = angle / torch.sin(angle / 2 + eps)

        small_angles = (angle <= 1e-3).float()
        rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
        rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
        return rot_vec

    # Rotation functions

    def compose_q_update_vec(
        self,
        q_update_vec: torch.Tensor,
        normalize_quats: bool = True,
        update_mask: torch.Tensor = None,
    ):
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
        quat_update = quat_multiply_by_vec(quats, q_update_vec)
        if update_mask is not None:
            quat_update = quat_update * update_mask
        new_quats = quats + quat_update
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r):
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

    def compose_q(self, r, normalize_quats: bool = True):
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

    def invert(self):
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
    ):
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
        rs,
        dim: int,
    ):
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

    def map_tensor_fn(self, fn):
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

    def cuda(self):
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

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]):
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

    def detach(self):
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
        trans = trans.type(torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ):
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
    ):
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
    ):
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
    ):
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
        update_mask: torch.Tensor = None,
    ):
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
        new_rots = self._rots.compose_q_update_vec(q_vec, update_mask=update_mask)

        trans_update = self._rots.apply(t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose_tran_update_vec(
        self,
        t_vec: torch.Tensor,
        update_mask: torch.Tensor = None,
    ):
        """
        Composes the transformation with a quaternion update vector of
        shape [*, 3], where columns represent a 3D translation.

        Args:
            q_vec: The quaternion update vector.
        Returns:
            The composed transformation.
        """
        trans_update = self._rots.apply(t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update

        return Rigid(self._rots, new_translation)

    def compose(
        self,
        r,
    ):
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

    def compose_r(self, rot, order="right"):
        """
        Composes the current rigid object with another.

        Args:
            r:
                Another Rigid object
            order:
                Order in which to perform rotation multiplication.
        Returns:
            The composition of the two transformations
        """
        if order == "right":
            new_rot = self._rots.compose_r(rot)
        elif order == "left":
            new_rot = rot.compose_r(self._rots)
        else:
            raise ValueError(f"Unrecognized multiplication order: {order}")
        return Rigid(new_rot, self._trans)

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

    def invert(self):
        """
        Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn):
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
    def from_tensor_4x4(t: torch.Tensor):
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
    ):
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
    ):
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
    ):
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
        ts,
        dim: int,
    ):
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

    def apply_rot_fn(self, fn):
        """
        Applies a Rotation -> Rotation function to the stored rotation
        object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn):
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

    def scale_translation(self, trans_scale_factor: float):
        """
        Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """
        fn = lambda t: t * trans_scale_factor
        return self.apply_trans_fn(fn)

    def stop_rot_gradient(self):
        """
        Detaches the underlying rotation object

        Returns:
            A transformation object with detached rotations
        """
        fn = lambda r: r.detach()
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
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

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
        c1_rots[..., 2, 0] = -1 * sin_c2
        c1_rots[..., 2, 2] = cos_c2

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

    def cuda(self):
        """
        Moves the transformation object to GPU memory

        Returns:
            A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())


# ---- pepflow/modules/protein/constants.py ----


## others
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

PAD_RESIDUE_INDEX = 21

##
# Residue identities

non_standard_residue_substitutions = {
    "2AS": "ASP",
    "3AH": "HIS",
    "5HP": "GLU",
    "ACL": "ARG",
    "AGM": "ARG",
    "AIB": "ALA",
    "ALM": "ALA",
    "ALO": "THR",
    "ALY": "LYS",
    "ARM": "ARG",
    "ASA": "ASP",
    "ASB": "ASP",
    "ASK": "ASP",
    "ASL": "ASP",
    "ASQ": "ASP",
    "AYA": "ALA",
    "BCS": "CYS",
    "BHD": "ASP",
    "BMT": "THR",
    "BNN": "ALA",
    "BUC": "CYS",
    "BUG": "LEU",
    "C5C": "CYS",
    "C6C": "CYS",
    "CAS": "CYS",
    "CCS": "CYS",
    "CEA": "CYS",
    "CGU": "GLU",
    "CHG": "ALA",
    "CLE": "LEU",
    "CME": "CYS",
    "CSD": "ALA",
    "CSO": "CYS",
    "CSP": "CYS",
    "CSS": "CYS",
    "CSW": "CYS",
    "CSX": "CYS",
    "CXM": "MET",
    "CY1": "CYS",
    "CY3": "CYS",
    "CYG": "CYS",
    "CYM": "CYS",
    "CYQ": "CYS",
    "DAH": "PHE",
    "DAL": "ALA",
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGL": "GLU",
    "DGN": "GLN",
    "DHA": "ALA",
    "DHI": "HIS",
    "DIL": "ILE",
    "DIV": "VAL",
    "DLE": "LEU",
    "DLY": "LYS",
    "DNP": "ALA",
    "DPN": "PHE",
    "DPR": "PRO",
    "DSN": "SER",
    "DSP": "ASP",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
    "DVA": "VAL",
    "EFC": "CYS",
    "FLA": "ALA",
    "FME": "MET",
    "GGL": "GLU",
    "GL3": "GLY",
    "GLZ": "GLY",
    "GMA": "GLU",
    "GSC": "GLY",
    "HAC": "ALA",
    "HAR": "ARG",
    "HIC": "HIS",
    "HIP": "HIS",
    "HMR": "ARG",
    "HPQ": "PHE",
    "HTR": "TRP",
    "HYP": "PRO",
    "IAS": "ASP",
    "IIL": "ILE",
    "IYR": "TYR",
    "KCX": "LYS",
    "LLP": "LYS",
    "LLY": "LYS",
    "LTR": "TRP",
    "LYM": "LYS",
    "LYZ": "LYS",
    "MAA": "ALA",
    "MEN": "ASN",
    "MHS": "HIS",
    "MIS": "SER",
    "MLE": "LEU",
    "MPQ": "GLY",
    "MSA": "GLY",
    "MSE": "MET",
    "MVA": "VAL",
    "NEM": "HIS",
    "NEP": "HIS",
    "NLE": "LEU",
    "NLN": "LEU",
    "NLP": "LEU",
    "NMC": "GLY",
    "OAS": "SER",
    "OCS": "CYS",
    "OMT": "MET",
    "PAQ": "TYR",
    "PCA": "GLU",
    "PEC": "CYS",
    "PHI": "PHE",
    "PHL": "PHE",
    "PR3": "CYS",
    "PRR": "ALA",
    "PTR": "TYR",
    "PYX": "CYS",
    "SAC": "SER",
    "SAR": "GLY",
    "SCH": "CYS",
    "SCS": "CYS",
    "SCY": "CYS",
    "SEL": "SER",
    "SEP": "SER",
    "SET": "SER",
    "SHC": "CYS",
    "SHR": "LYS",
    "SMC": "CYS",
    "SOC": "CYS",
    "STY": "TYR",
    "SVA": "SER",
    "TIH": "ALA",
    "TPL": "TRP",
    "TPO": "THR",
    "TPQ": "ALA",
    "TRG": "LYS",
    "TRO": "TRP",
    "TYB": "TYR",
    "TYI": "TYR",
    "TYQ": "TYR",
    "TYS": "TYR",
    "TYY": "TYR",
    "ALA": "ALA",
    "CYS": "CYS",
    "ASP": "ASP",
    "GLU": "GLU",
    "PHE": "PHE",
    "GLY": "GLY",
    "HIS": "HIS",
    "ILE": "ILE",
    "LYS": "LYS",
    "LEU": "LEU",
    "MET": "MET",
    "ASN": "ASN",
    "PRO": "PRO",
    "GLN": "GLN",
    "ARG": "ARG",
    "SER": "SER",
    "THR": "THR",
    "VAL": "VAL",
    "TRP": "TRP",
    "TYR": "TYR",
    "UNK": "UNK",
}


ressymb_to_resindex = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}

resindex_to_ressymb = {}
for k, v in ressymb_to_resindex.items():
    resindex_to_ressymb[v] = k

BACKBONE_FRAME = 0
OMEGA_FRAME = 1
PHI_FRAME = 2
PSI_FRAME = 3
CHI1_FRAME, CHI2_FRAME, CHI3_FRAME, CHI4_FRAME = 4, 5, 6, 7


class AA(enum.IntEnum):
    ALA = 0
    CYS = 1
    ASP = 2
    GLU = 3
    PHE = 4
    GLY = 5
    HIS = 6
    ILE = 7
    LYS = 8
    LEU = 9
    MET = 10
    ASN = 11
    PRO = 12
    GLN = 13
    ARG = 14
    SER = 15
    THR = 16
    VAL = 17
    TRP = 18
    TYR = 19
    UNK = 20

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:  # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:  # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    @classmethod
    def is_aa(cls, value):
        return (
            (value in ressymb_to_resindex)
            or (value in non_standard_residue_substitutions)
            or (value in cls._member_names_)
        )


num_aa_types = len(AA)

##
# Atom identities


class BBHeavyAtom(enum.IntEnum):
    N = 0
    CA = 1
    C = 2
    O = 3
    CB = 4
    OXT = 14


max_num_heavyatoms = 15
max_num_hydrogens = 16
max_num_allatoms = max_num_heavyatoms + max_num_hydrogens

restype_to_heavyatom_names = {
    AA.ALA: ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", "", "OXT"],
    AA.ARG: ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", "", "OXT"],
    AA.ASN: ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", "", "OXT"],
    AA.ASP: ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", "", "OXT"],
    AA.CYS: ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", "", "OXT"],
    AA.GLN: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", "", "OXT"],
    AA.GLU: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", "", "OXT"],
    AA.GLY: ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", "", "OXT"],
    AA.HIS: ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", "", "OXT"],
    AA.ILE: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", "", "OXT"],
    AA.LEU: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", "", "OXT"],
    AA.LYS: ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", "", "OXT"],
    AA.MET: ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", "", "OXT"],
    AA.PHE: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", "", "OXT"],
    AA.PRO: ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", "", "OXT"],
    AA.SER: ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", "", "OXT"],
    AA.THR: ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", "", "OXT"],
    AA.TRP: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "OXT",
    ],
    AA.TYR: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
        "OXT",
    ],
    AA.VAL: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", "", "OXT"],
    AA.UNK: ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}
for names in restype_to_heavyatom_names.values():
    assert len(names) == max_num_heavyatoms

restype_to_hydrogen_names = {
    AA.ALA: ["H", "H2", "H3", "HA", "HB1", "HB2", "HB3", "HXT", "", "", "", "", "", "", "", ""],
    AA.CYS: ["H", "H2", "H3", "HA", "HB2", "HB3", "HG", "HXT", "", "", "", "", "", "", "", ""],
    AA.ASP: ["H", "H2", "H3", "HA", "HB2", "HB3", "HD2", "HXT", "", "", "", "", "", "", "", ""],
    AA.GLU: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HE2",
        "HXT",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    AA.PHE: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HD1",
        "HD2",
        "HE1",
        "HE2",
        "HZ",
        "HXT",
        "",
        "",
        "",
        "",
    ],
    AA.GLY: ["H", "H2", "H3", "HA2", "HA3", "HXT", "", "", "", "", "", "", "", "", "", ""],
    AA.HIS: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HD1",
        "HD2",
        "HE1",
        "HE2",
        "HXT",
        "",
        "",
        "",
        "",
        "",
    ],
    AA.ILE: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB",
        "HG12",
        "HG13",
        "HG21",
        "HG22",
        "HG23",
        "HD11",
        "HD12",
        "HD13",
        "HXT",
        "",
        "",
    ],
    AA.LYS: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HD2",
        "HD3",
        "HE2",
        "HE3",
        "HZ1",
        "HZ2",
        "HZ3",
        "HXT",
    ],
    AA.LEU: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG",
        "HD11",
        "HD12",
        "HD13",
        "HD21",
        "HD22",
        "HD23",
        "HXT",
        "",
        "",
    ],
    AA.MET: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HE1",
        "HE2",
        "HE3",
        "HXT",
        "",
        "",
        "",
        "",
    ],
    AA.ASN: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HD21",
        "HD22",
        "HXT",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    AA.PRO: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HD2",
        "HD3",
        "HXT",
        "",
        "",
        "",
        "",
        "",
    ],
    AA.GLN: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HE21",
        "HE22",
        "HXT",
        "",
        "",
        "",
        "",
        "",
    ],
    AA.ARG: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HD2",
        "HD3",
        "HE",
        "HH11",
        "HH12",
        "HH21",
        "HH22",
        "HXT",
    ],
    AA.SER: ["H", "H2", "H3", "HA", "HB2", "HB3", "HG", "HXT", "", "", "", "", "", "", "", ""],
    AA.THR: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB",
        "HG1",
        "HG21",
        "HG22",
        "HG23",
        "HXT",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    AA.VAL: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB",
        "HG11",
        "HG12",
        "HG13",
        "HG21",
        "HG22",
        "HG23",
        "HXT",
        "",
        "",
        "",
        "",
    ],
    AA.TRP: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HD1",
        "HE1",
        "HE3",
        "HZ2",
        "HZ3",
        "HH2",
        "HXT",
        "",
        "",
        "",
    ],
    AA.TYR: [
        "H",
        "H2",
        "H3",
        "HA",
        "HB2",
        "HB3",
        "HD1",
        "HD2",
        "HE1",
        "HE2",
        "HH",
        "HXT",
        "",
        "",
        "",
        "",
    ],
    AA.UNK: ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}
for names in restype_to_hydrogen_names.values():
    assert len(names) == max_num_hydrogens

restype_to_allatom_names = {
    restype: restype_to_heavyatom_names[restype] + restype_to_hydrogen_names[restype]
    for restype in AA
}

restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name != ""}
    for resname, atoms in restype_to_heavyatom_names.items()
}

##
# Bond identities


class BondType(enum.IntEnum):
    NoBond = 0
    Single = 1
    Double = 2
    Triple = 3
    AromaticSingle = 5
    AromaticDouble = 6


BT = BondType
restype_to_bonded_atom_name_pairs = {
    AA.ALA: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "HB1", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.CYS: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "SG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("SG", "HG", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.ASP: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "OD1", BT.AromaticDouble),
        ("CG", "OD2", BT.AromaticSingle),
        ("OD2", "HD2", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.GLU: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD", BT.AromaticSingle),
        ("CG", "HG2", BT.AromaticSingle),
        ("CG", "HG3", BT.AromaticSingle),
        ("CD", "OE1", BT.AromaticDouble),
        ("CD", "OE2", BT.AromaticSingle),
        ("OE2", "HE2", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.PHE: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD1", BT.AromaticDouble),
        ("CG", "CD2", BT.AromaticSingle),
        ("CD1", "CE1", BT.AromaticSingle),
        ("CD1", "HD1", BT.AromaticSingle),
        ("CD2", "CE2", BT.AromaticDouble),
        ("CD2", "HD2", BT.AromaticSingle),
        ("CE1", "CZ", BT.AromaticDouble),
        ("CE1", "HE1", BT.AromaticSingle),
        ("CE2", "CZ", BT.AromaticSingle),
        ("CE2", "HE2", BT.AromaticSingle),
        ("CZ", "HZ", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.GLY: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "HA2", BT.AromaticSingle),
        ("CA", "HA3", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.HIS: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "ND1", BT.AromaticSingle),
        ("CG", "CD2", BT.AromaticDouble),
        ("ND1", "CE1", BT.AromaticDouble),
        ("ND1", "HD1", BT.AromaticSingle),
        ("CD2", "NE2", BT.AromaticSingle),
        ("CD2", "HD2", BT.AromaticSingle),
        ("CE1", "NE2", BT.AromaticSingle),
        ("CE1", "HE1", BT.AromaticSingle),
        ("NE2", "HE2", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.ILE: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG1", BT.AromaticSingle),
        ("CB", "CG2", BT.AromaticSingle),
        ("CB", "HB", BT.AromaticSingle),
        ("CG1", "CD1", BT.AromaticSingle),
        ("CG1", "HG12", BT.AromaticSingle),
        ("CG1", "HG13", BT.AromaticSingle),
        ("CG2", "HG21", BT.AromaticSingle),
        ("CG2", "HG22", BT.AromaticSingle),
        ("CG2", "HG23", BT.AromaticSingle),
        ("CD1", "HD11", BT.AromaticSingle),
        ("CD1", "HD12", BT.AromaticSingle),
        ("CD1", "HD13", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.LYS: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD", BT.AromaticSingle),
        ("CG", "HG2", BT.AromaticSingle),
        ("CG", "HG3", BT.AromaticSingle),
        ("CD", "CE", BT.AromaticSingle),
        ("CD", "HD2", BT.AromaticSingle),
        ("CD", "HD3", BT.AromaticSingle),
        ("CE", "NZ", BT.AromaticSingle),
        ("CE", "HE2", BT.AromaticSingle),
        ("CE", "HE3", BT.AromaticSingle),
        ("NZ", "HZ1", BT.AromaticSingle),
        ("NZ", "HZ2", BT.AromaticSingle),
        ("NZ", "HZ3", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.LEU: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD1", BT.AromaticSingle),
        ("CG", "CD2", BT.AromaticSingle),
        ("CG", "HG", BT.AromaticSingle),
        ("CD1", "HD11", BT.AromaticSingle),
        ("CD1", "HD12", BT.AromaticSingle),
        ("CD1", "HD13", BT.AromaticSingle),
        ("CD2", "HD21", BT.AromaticSingle),
        ("CD2", "HD22", BT.AromaticSingle),
        ("CD2", "HD23", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.MET: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "SD", BT.AromaticSingle),
        ("CG", "HG2", BT.AromaticSingle),
        ("CG", "HG3", BT.AromaticSingle),
        ("SD", "CE", BT.AromaticSingle),
        ("CE", "HE1", BT.AromaticSingle),
        ("CE", "HE2", BT.AromaticSingle),
        ("CE", "HE3", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.ASN: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "OD1", BT.AromaticDouble),
        ("CG", "ND2", BT.AromaticSingle),
        ("ND2", "HD21", BT.AromaticSingle),
        ("ND2", "HD22", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.PRO: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("N", "CD", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD", BT.AromaticSingle),
        ("CG", "HG2", BT.AromaticSingle),
        ("CG", "HG3", BT.AromaticSingle),
        ("CD", "HD2", BT.AromaticSingle),
        ("CD", "HD3", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.GLN: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD", BT.AromaticSingle),
        ("CG", "HG2", BT.AromaticSingle),
        ("CG", "HG3", BT.AromaticSingle),
        ("CD", "OE1", BT.AromaticDouble),
        ("CD", "NE2", BT.AromaticSingle),
        ("NE2", "HE21", BT.AromaticSingle),
        ("NE2", "HE22", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.ARG: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD", BT.AromaticSingle),
        ("CG", "HG2", BT.AromaticSingle),
        ("CG", "HG3", BT.AromaticSingle),
        ("CD", "NE", BT.AromaticSingle),
        ("CD", "HD2", BT.AromaticSingle),
        ("CD", "HD3", BT.AromaticSingle),
        ("NE", "CZ", BT.AromaticSingle),
        ("NE", "HE", BT.AromaticSingle),
        ("CZ", "NH1", BT.AromaticSingle),
        ("CZ", "NH2", BT.AromaticDouble),
        ("NH1", "HH11", BT.AromaticSingle),
        ("NH1", "HH12", BT.AromaticSingle),
        ("NH2", "HH21", BT.AromaticSingle),
        ("NH2", "HH22", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.SER: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "OG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("OG", "HG", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.THR: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "OG1", BT.AromaticSingle),
        ("CB", "CG2", BT.AromaticSingle),
        ("CB", "HB", BT.AromaticSingle),
        ("OG1", "HG1", BT.AromaticSingle),
        ("CG2", "HG21", BT.AromaticSingle),
        ("CG2", "HG22", BT.AromaticSingle),
        ("CG2", "HG23", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.VAL: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG1", BT.AromaticSingle),
        ("CB", "CG2", BT.AromaticSingle),
        ("CB", "HB", BT.AromaticSingle),
        ("CG1", "HG11", BT.AromaticSingle),
        ("CG1", "HG12", BT.AromaticSingle),
        ("CG1", "HG13", BT.AromaticSingle),
        ("CG2", "HG21", BT.AromaticSingle),
        ("CG2", "HG22", BT.AromaticSingle),
        ("CG2", "HG23", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.TRP: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD1", BT.AromaticDouble),
        ("CG", "CD2", BT.AromaticSingle),
        ("CD1", "NE1", BT.AromaticSingle),
        ("CD1", "HD1", BT.AromaticSingle),
        ("CD2", "CE2", BT.AromaticDouble),
        ("CD2", "CE3", BT.AromaticSingle),
        ("NE1", "CE2", BT.AromaticSingle),
        ("NE1", "HE1", BT.AromaticSingle),
        ("CE2", "CZ2", BT.AromaticSingle),
        ("CE3", "CZ3", BT.AromaticDouble),
        ("CE3", "HE3", BT.AromaticSingle),
        ("CZ2", "CH2", BT.AromaticDouble),
        ("CZ2", "HZ2", BT.AromaticSingle),
        ("CZ3", "CH2", BT.AromaticSingle),
        ("CZ3", "HZ3", BT.AromaticSingle),
        ("CH2", "HH2", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.TYR: [
        ("N", "H", BT.Single),
        ("N", "H2", BT.Single),
        ("N", "H3", BT.Single),
        ("N", "CA", BT.AromaticSingle),
        ("CA", "C", BT.AromaticSingle),
        ("CA", "CB", BT.AromaticSingle),
        ("CA", "HA", BT.AromaticSingle),
        ("C", "O", BT.AromaticDouble),
        ("C", "OXT", BT.AromaticSingle),
        ("CB", "CG", BT.AromaticSingle),
        ("CB", "HB2", BT.AromaticSingle),
        ("CB", "HB3", BT.AromaticSingle),
        ("CG", "CD1", BT.AromaticDouble),
        ("CG", "CD2", BT.AromaticSingle),
        ("CD1", "CE1", BT.AromaticSingle),
        ("CD1", "HD1", BT.AromaticSingle),
        ("CD2", "CE2", BT.AromaticDouble),
        ("CD2", "HD2", BT.AromaticSingle),
        ("CE1", "CZ", BT.AromaticDouble),
        ("CE1", "HE1", BT.AromaticSingle),
        ("CE2", "CZ", BT.AromaticSingle),
        ("CE2", "HE2", BT.AromaticSingle),
        ("CZ", "OH", BT.AromaticSingle),
        ("OH", "HH", BT.AromaticSingle),
        ("OXT", "HXT", BT.AromaticSingle),
    ],
    AA.UNK: [],
}


restype_to_allatom_bond_matrix = {
    restype: torch.zeros([max_num_allatoms, max_num_allatoms], dtype=torch.long) for restype in AA
}
restype_to_heavyatom_bond_matrix = {
    restype: torch.zeros([max_num_heavyatoms, max_num_heavyatoms], dtype=torch.long)
    for restype in AA
}


def _make_bond_matrices():
    for restype in AA:
        for atom1_name, atom2_name, bond_type in restype_to_bonded_atom_name_pairs[restype]:
            idx1 = restype_to_allatom_names[restype].index(atom1_name)
            idx2 = restype_to_allatom_names[restype].index(atom2_name)
            restype_to_allatom_bond_matrix[restype][idx1, idx2] = bond_type
            restype_to_allatom_bond_matrix[restype][idx2, idx1] = bond_type
            if (
                atom1_name in restype_to_heavyatom_names[restype]
                and atom2_name in restype_to_heavyatom_names[restype]
            ):
                jdx1 = restype_to_heavyatom_names[restype].index(atom1_name)
                jdx2 = restype_to_heavyatom_names[restype].index(atom2_name)
                restype_to_heavyatom_bond_matrix[restype][jdx1, jdx2] = bond_type
                restype_to_heavyatom_bond_matrix[restype][jdx2, jdx1] = bond_type


_make_bond_matrices()


##
# Torsion geometry and ideal coordinates


class Torsion(enum.IntEnum):
    Backbone = 0
    Omega = 1
    Phi = 2
    Psi = 3
    Chi1 = 4
    Chi2 = 5
    Chi3 = 6
    Chi7 = 7


chi_angles_atoms = {
    AA.ALA: [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    AA.ARG: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    AA.ASN: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    AA.ASP: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    AA.CYS: [["N", "CA", "CB", "SG"]],
    AA.GLN: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    AA.GLU: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    AA.GLY: [],
    AA.HIS: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    AA.ILE: [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    AA.LEU: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AA.LYS: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    AA.MET: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    AA.PHE: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AA.PRO: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    AA.SER: [["N", "CA", "CB", "OG"]],
    AA.THR: [["N", "CA", "CB", "OG1"]],
    AA.TRP: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AA.TYR: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AA.VAL: [["N", "CA", "CB", "CG1"]],
}


chi_angles_mask = {
    AA.ALA: [False, False, False, False],  # ALA
    AA.ARG: [True, True, True, True],  # ARG
    AA.ASN: [True, True, False, False],  # ASN
    AA.ASP: [True, True, False, False],  # ASP
    AA.CYS: [True, False, False, False],  # CYS
    AA.GLN: [True, True, True, False],  # GLN
    AA.GLU: [True, True, True, False],  # GLU
    AA.GLY: [False, False, False, False],  # GLY
    AA.HIS: [True, True, False, False],  # HIS
    AA.ILE: [True, True, False, False],  # ILE
    AA.LEU: [True, True, False, False],  # LEU
    AA.LYS: [True, True, True, True],  # LYS
    AA.MET: [True, True, True, False],  # MET
    AA.PHE: [True, True, False, False],  # PHE
    AA.PRO: [True, True, False, False],  # PRO
    AA.SER: [True, False, False, False],  # SER
    AA.THR: [True, False, False, False],  # THR
    AA.TRP: [True, True, False, False],  # TRP
    AA.TYR: [True, True, False, False],  # TYR
    AA.VAL: [True, False, False, False],  # VAL
    AA.UNK: [False, False, False, False],  # UNK
}


chi_pi_periodic = {
    AA.ALA: [False, False, False, False],  # ALA
    AA.ARG: [False, False, False, False],  # ARG
    AA.ASN: [False, False, False, False],  # ASN
    AA.ASP: [False, True, False, False],  # ASP
    AA.CYS: [False, False, False, False],  # CYS
    AA.GLN: [False, False, False, False],  # GLN
    AA.GLU: [False, False, True, False],  # GLU
    AA.GLY: [False, False, False, False],  # GLY
    AA.HIS: [False, False, False, False],  # HIS
    AA.ILE: [False, False, False, False],  # ILE
    AA.LEU: [False, False, False, False],  # LEU
    AA.LYS: [False, False, False, False],  # LYS
    AA.MET: [False, False, False, False],  # MET
    AA.PHE: [False, True, False, False],  # PHE
    AA.PRO: [False, False, False, False],  # PRO
    AA.SER: [False, False, False, False],  # SER
    AA.THR: [False, False, False, False],  # THR
    AA.TRP: [False, False, False, False],  # TRP
    AA.TYR: [False, True, False, False],  # TYR
    AA.VAL: [False, False, False, False],  # VAL
    AA.UNK: [False, False, False, False],  # UNK
}


rigid_group_heavy_atom_positions = {
    AA.ALA: [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    AA.ARG: [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    AA.ASN: [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    AA.ASP: [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    AA.CYS: [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    AA.GLN: [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    AA.GLU: [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    AA.GLY: [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    AA.HIS: [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    AA.ILE: [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    AA.LEU: [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    AA.LYS: [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    AA.MET: [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    AA.PHE: [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    AA.PRO: [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    AA.SER: [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    AA.THR: [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    AA.TRP: [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    AA.TYR: [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    AA.VAL: [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}


# The following tensors are initialized by `_make_rigid_group_constants`
restype_rigid_group_rotation = torch.zeros([21, 8, 3, 3])
restype_rigid_group_translation = torch.zeros([21, 8, 3])
restype_heavyatom_to_rigid_group = torch.zeros([21, 14], dtype=torch.long)
restype_heavyatom_rigid_group_positions = torch.zeros([21, 14, 3])


def _make_rigid_group_constants():
    def _make_rotation_matrix(ex, ey):
        ex_normalized = ex / torch.linalg.norm(ex)

        # make ey perpendicular to ex
        ey_normalized = ey - torch.dot(ey, ex_normalized) * ex_normalized
        ey_normalized /= torch.linalg.norm(ey_normalized)

        eznorm = torch.cross(ex_normalized, ey_normalized)
        m = torch.stack([ex_normalized, ey_normalized, eznorm]).transpose(0, 1)  # (3, 3_index)
        return m

    for restype in AA:
        if restype == AA.UNK:
            continue

        atom_groups = {name: group for name, group, _ in rigid_group_heavy_atom_positions[restype]}
        atom_positions = {
            name: torch.FloatTensor(pos)
            for name, _, pos in rigid_group_heavy_atom_positions[restype]
        }

        # Atom 14 rigid group positions
        for atom_idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
            if (atom_name == "") or (atom_name not in atom_groups):
                continue
            restype_heavyatom_to_rigid_group[restype, atom_idx] = atom_groups[atom_name]
            restype_heavyatom_rigid_group_positions[restype, atom_idx, :] = atom_positions[
                atom_name
            ]

        # 0: backbone to backbone
        restype_rigid_group_rotation[restype, Torsion.Backbone, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Backbone, :] = torch.zeros([3])

        # 1: omega-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Omega, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Omega, :] = torch.zeros([3])

        # 2: phi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Phi, :, :] = _make_rotation_matrix(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=torch.FloatTensor([1.0, 0.0, 0.0]),
        )
        restype_rigid_group_translation[restype, Torsion.Phi, :] = atom_positions["N"]

        # 3: psi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Psi, :, :] = _make_rotation_matrix(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"]
            - atom_positions["N"],  # In accordance to the definition of psi angle
        )
        restype_rigid_group_translation[restype, Torsion.Psi, :] = atom_positions["C"]

        # 4: chi1-frame to backbone
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[restype][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            restype_rigid_group_rotation[restype, Torsion.Chi1, :, :] = _make_rotation_matrix(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
            )
            restype_rigid_group_translation[restype, Torsion.Chi1, :] = base_atom_positions[2]

        # chi2-chi1
        # chi3-chi2
        # chi4-chi3
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[restype][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                restype_rigid_group_rotation[restype, Torsion.Chi1 + chi_idx, :, :] = (
                    _make_rotation_matrix(
                        ex=axis_end_atom_position,
                        ey=torch.FloatTensor([-1.0, 0.0, 0.0]),
                    )
                )
                restype_rigid_group_translation[restype, Torsion.Chi1 + chi_idx, :] = (
                    axis_end_atom_position
                )


_make_rigid_group_constants()


"""
# The following tensors are taken from diffab
"""
backbone_atom_coordinates = {
    AA.ALA: [
        (-0.525, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.ARG: [
        (-0.524, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.ASN: [
        (-0.536, 1.357, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.ASP: [
        (-0.525, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, 0.0, -0.0),  # C
    ],
    AA.CYS: [
        (-0.522, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, 0.0),  # C
    ],
    AA.GLN: [
        (-0.526, 1.361, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA.GLU: [
        (-0.528, 1.361, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.GLY: [
        (-0.572, 1.337, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.517, -0.0, -0.0),  # C
    ],
    AA.HIS: [
        (-0.527, 1.36, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA.ILE: [
        (-0.493, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
    AA.LEU: [
        (-0.52, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.LYS: [
        (-0.526, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA.MET: [
        (-0.521, 1.364, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA.PHE: [
        (-0.518, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, -0.0),  # C
    ],
    AA.PRO: [
        (-0.566, 1.351, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, 0.0),  # C
    ],
    AA.SER: [
        (-0.529, 1.36, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.THR: [
        (-0.517, 1.364, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, -0.0),  # C
    ],
    AA.TRP: [
        (-0.521, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, 0.0),  # C
    ],
    AA.TYR: [
        (-0.522, 1.362, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, -0.0, -0.0),  # C
    ],
    AA.VAL: [
        (-0.494, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
}

bb_oxygen_coordinate = {
    AA.ALA: (2.153, -1.062, 0.0),
    AA.ARG: (2.151, -1.062, 0.0),
    AA.ASN: (2.151, -1.062, 0.0),
    AA.ASP: (2.153, -1.062, 0.0),
    AA.CYS: (2.149, -1.062, 0.0),
    AA.GLN: (2.152, -1.062, 0.0),
    AA.GLU: (2.152, -1.062, 0.0),
    AA.GLY: (2.143, -1.062, 0.0),
    AA.HIS: (2.15, -1.063, 0.0),
    AA.ILE: (2.154, -1.062, 0.0),
    AA.LEU: (2.15, -1.063, 0.0),
    AA.LYS: (2.152, -1.062, 0.0),
    AA.MET: (2.15, -1.062, 0.0),
    AA.PHE: (2.15, -1.062, 0.0),
    AA.PRO: (2.148, -1.066, 0.0),
    AA.SER: (2.151, -1.062, 0.0),
    AA.THR: (2.152, -1.062, 0.0),
    AA.TRP: (2.152, -1.062, 0.0),
    AA.TYR: (2.151, -1.062, 0.0),
    AA.VAL: (2.154, -1.062, 0.0),
}

backbone_atom_coordinates_tensor = torch.zeros([21, 3, 3])
bb_oxygen_coordinate_tensor = torch.zeros([21, 3])


def make_coordinate_tensors():
    for restype, atom_coords in backbone_atom_coordinates.items():
        for atom_id, atom_coord in enumerate(atom_coords):
            backbone_atom_coordinates_tensor[restype][atom_id] = torch.FloatTensor(atom_coord)

    for restype, bb_oxy_coord in bb_oxygen_coordinate.items():
        bb_oxygen_coordinate_tensor[restype] = torch.FloatTensor(bb_oxy_coord)


make_coordinate_tensors()


# ---- pepflow/modules/common/topology.py ----


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Args:
        chain_nb, res_nb
    Returns:
        consec: A flag tensor indicating whether residue-i is connected to residue-(i+1),
                BoolTensor, (B, L-1)[b, i].
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()  # (B, L-1)
    same_chain = chain_nb[:, 1:] == chain_nb[:, :-1]
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag


# ---- pepflow/modules/common/layers.py ----


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes)  # (N, L)
    x = x.clamp(min=0, max=num_classes - 1)
    y = F.one_hot(x, num_classes) * mask[..., None]  # (N, L, C)
    return y


def sample_from(c):
    """sample from c"""
    N, L, K = c.size()
    c = c.view(N * L, K) + 1e-8
    x = torch.multinomial(c, 1).view(N, L)
    return x


class DistanceToBins(nn.Module):
    def __init__(self, dist_min=0.0, dist_max=20.0, num_bins=64, use_onehot=False):
        super().__init__()
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.num_bins = num_bins
        self.use_onehot = use_onehot

        if use_onehot:
            offset = torch.linspace(dist_min, dist_max, self.num_bins)
        else:
            offset = torch.linspace(dist_min, dist_max, self.num_bins - 1)  # 1 overflow flag
            self.coeff = (
                -0.5 / ((offset[1] - offset[0]) * 0.2).item() ** 2
            )  # `*0.2`: makes it not too blurred
        self.register_buffer("offset", offset)

    @property
    def out_channels(self):
        return self.num_bins

    def forward(self, dist, dim, normalize=True):
        """
        Args:
            dist:   (N, *, 1, *)
        Returns:
            (N, *, num_bins, *)
        """
        assert dist.size()[dim] == 1
        offset_shape = [1] * len(dist.size())
        offset_shape[dim] = -1

        if self.use_onehot:
            diff = torch.abs(dist - self.offset.view(*offset_shape))  # (N, *, num_bins, *)
            bin_idx = torch.argmin(diff, dim=dim, keepdim=True)  # (N, *, 1, *)
            y = torch.zeros_like(diff).scatter_(dim=dim, index=bin_idx, value=1.0)
        else:
            overflow_symb = (dist >= self.dist_max).float()  # (N, *, 1, *)
            y = dist - self.offset.view(*offset_shape)  # (N, *, num_bins-1, *)
            y = torch.exp(self.coeff * torch.pow(y, 2))  # (N, *, num_bins-1, *)
            y = torch.cat([y, overflow_symb], dim=dim)  # (N, *, num_bins, *)
            if normalize:
                y = y / y.sum(dim=dim, keepdim=True)

        return y


class PositionalEncoding(nn.Module):
    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer("freq_bands", 2.0 ** torch.linspace(0.0, num_funcs - 1, num_funcs))

    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)  # (..., d, 1)
        code = torch.cat(
            [x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1
        )  # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class AngularEncoding(nn.Module):
    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer(
            "freq_bands",
            torch.FloatTensor(
                [i + 1 for i in range(num_funcs)] + [1.0 / (i + 1) for i in range(num_funcs)]
            ),
        )

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2 * 2 * self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)  # (..., d, 1)
        code = torch.cat(
            [x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1
        )  # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class LayerNorm(nn.Module):
    def __init__(self, normal_shape, gamma=True, beta=True, epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super().__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter("gamma", None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter("beta", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return "normal_shape={}, gamma={}, beta={}, epsilon={}".format(
            self.normal_shape,
            self.gamma is not None,
            self.beta is not None,
            self.epsilon,
        )


# ---- pepflow/modules/common/geometry.py ----


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def align(
    pos_1: torch.Tensor,
    pos_2: torch.Tensor,
    pos_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(L,A,3),(L,A) align pos14_model to pos14_native, return aligned pos"""
    L, A, _ = pos_1.shape
    x = torch.masked_select(pos_1, pos_mask.bool().unsqueeze(-1)).reshape(-1, 3)
    y = torch.masked_select(pos_2, pos_mask.bool().unsqueeze(-1)).reshape(-1, 3)
    xm, ym = x.mean(dim=0), y.mean(dim=0)  # (1,A,3)
    x = x - x.mean(dim=0, keepdim=True)  # (L,A,3)
    y = y - y.mean(dim=0, keepdim=True)  # (L,A,3)
    s = x.T @ y
    u, sigma, vt = torch.linalg.svd(s)
    r = vt.T @ u.T  # (3,3)
    t = ym - r @ xm  #
    pos_1_aligned = ((r @ pos_1.view(-1, 3).T).T + t).reshape(L, A, 3)  # (-1,3) -> (L,A,3)

    return pos_1_aligned, pos_2


def batch_align(
    pos_1: torch.Tensor,
    pos_2: torch.Tensor,
    pos_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,L,A,3),(B,L,A) Batch align pos_1 to pos_2, return aligned pos_1 and pos_2"""
    x = torch.masked_select(pos_1, pos_mask.unsqueeze(-1)).reshape(pos_1.size(0), -1, 3)
    y = torch.masked_select(pos_2, pos_mask.unsqueeze(-1)).reshape(pos_2.size(0), -1, 3)
    xm = x.mean(dim=1, keepdim=True)
    ym = y.mean(dim=1, keepdim=True)
    x = x - xm
    y = y - ym
    s = x.transpose(-1, -2) @ y
    u, sigma, vt = torch.linalg.svd(s)
    r = vt.transpose(-1, -2) @ u.transpose(-1, -2)
    t = ym - (r @ xm.transpose(-1, -2)).transpose(-1, -2)
    pos_1_aligned = (
        (r @ pos_1.reshape(pos_1.size(0), -1, 3).transpose(-1, -2)).transpose(-1, -2) + t
    ).reshape(pos_1.size(0), pos_1.size(1), -1, 3)

    return pos_1_aligned, pos_2


def pairwise_distances(x, y=None, return_v=False):
    """
    Args:
        x:  (B, N, d)
        y:  (B, M, d)
    """
    if y is None:
        y = x
    v = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, d)
    d = safe_norm(v, dim=-1)
    if return_v:
        return d, v
    else:
        return d


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center  # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center  # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)  # (N, L, 3)

    mat = torch.cat(
        [e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1
    )  # (N, L, 3, 3_index)
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)  # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p


def apply_rotation_to_vector(R, p):
    return local_to_global(R, torch.zeros_like(p), p)


def compose_rotation_and_translation(R1, t1, R2, t2):
    """
    Args:
        R1,t1:  Frame basis and coordinate, (N, L, 3, 3), (N, L, 3).
        R2,t2:  Rotation and translation to be applied to (R1, t1), (N, L, 3, 3), (N, L, 3).
    Returns
        R_new <- R1R2
        t_new <- R1t2 + t1
    """
    R_new = torch.matmul(R1, R2)  # (N, L, 3, 3)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_chain(Ts):
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        T_next = compose_rotation_and_translation(R1, t1, R2, t2)
        Ts = Ts[:-2] + [T_next]
    return Ts[0]


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def quaternion_1ijk_to_rotation_matrix(q):
    """
    (1 + ai + bj + ck) -> R
    Args:
        q:  (..., 3)
    """
    b, c, d = torch.unbind(q, dim=-1)
    s = torch.sqrt(1 + b**2 + c**2 + d**2)
    a, b, c, d = 1 / s, b / s, c / s, d / s

    o = torch.stack(
        (
            a**2 + b**2 - c**2 - d**2,
            2 * b * c - 2 * a * d,
            2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d,
            a**2 - b**2 + c**2 - d**2,
            2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c,
            2 * c * d + 2 * a * b,
            a**2 - b**2 - c**2 + d**2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def repr_6d_to_rotation_matrix(x):
    """
    Args:
        x:  6D representations, (..., 6).
    Returns:
        Rotation matrices, (..., 3, 3_index).
    """
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = normalize_vector(a1, dim=-1)
    b2 = normalize_vector(a2 - project_v2v(a2, b1, dim=-1), dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    mat = torch.cat(
        [b1.unsqueeze(-1), b2.unsqueeze(-1), b3.unsqueeze(-1)], dim=-1
    )  # (N, L, 3, 3_index)
    return mat


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    dihed = sgn * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihed = torch.nan_to_num(dihed)
    return dihed


def knn_gather(idx, value):
    """
    Args:
        idx:    (B, N, K)
        value:  (B, M, d)
    Returns:
        (B, N, K, d)
    """
    N, d = idx.size(1), value.size(-1)
    idx = idx.unsqueeze(-1).repeat(1, 1, 1, d)  # (B, N, K, d)
    value = value.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, M, d)
    return torch.gather(value, dim=2, index=idx)


def knn_points(q, p, K):
    """
    Args:
        q: (B, M, d)
        p: (B, N, d)
    Returns:
        (B, M, K), (B, M, K), (B, M, K, d)
    """
    _, L, _ = p.size()
    d = pairwise_distances(q, p)  # (B, N, M)
    dist, idx = d.topk(min(L, K), dim=-1, largest=False)  # (B, M, K), (B, M, K)
    return dist, idx, knn_gather(idx, p)


def angstrom_to_nm(x):
    return x / 10


def nm_to_angstrom(x):
    return x * 10


def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
        chain_nb:   (N, L).
        res_nb:     (N, L).
        mask:       (N, L).
    Returns:
        bb_dihedral:    Omega, Phi, and Psi angles in radian, (N, L, 3).
        mask_bb_dihed:  Masks of dihedral angles, (N, L, 3).
    """
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]  # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)  # (N, L)
    omega_mask = torch.logical_not(N_term_flag)
    phi_mask = torch.logical_not(N_term_flag)
    psi_mask = torch.logical_not(C_term_flag)

    # N-termini don't have omega and phi
    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]),
        pad=(1, 0),
        value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0),
        value=0,
    )

    # C-termini don't have psi
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1),
        value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed
    return bb_dihedral, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        Inter-residue Phi and Psi angles, (N, L, L, 2).
    """
    N, L = pos_atoms.shape[:2]
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]  # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:, :, None].expand(N, L, L, 3),
        pos_N[:, None, :].expand(N, L, L, 3),
        pos_CA[:, None, :].expand(N, L, L, 3),
        pos_C[:, None, :].expand(N, L, L, 3),
    )
    ir_psi = dihedral_from_four_points(
        pos_N[:, :, None].expand(N, L, L, 3),
        pos_CA[:, :, None].expand(N, L, L, 3),
        pos_C[:, :, None].expand(N, L, L, 3),
        pos_N[:, None, :].expand(N, L, L, 3),
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed


def apply_rotation_matrix_to_rot6d(R, O):
    """
    Args:
        R:  (..., 3, 3)
        O:  (..., 6)
    Returns:
        Rotated 6D representation, (..., 6).
    """
    u1, u2 = O[..., :3, None], O[..., 3:, None]  # (..., 3, 1)
    v1 = torch.matmul(R, u1).squeeze(-1)  # (..., 3)
    v2 = torch.matmul(R, u2).squeeze(-1)
    return torch.cat([v1, v2], dim=-1)


def normalize_rot6d(O):
    """
    Args:
        O:  (..., 6)
    """
    u1, u2 = O[..., :3], O[..., 3:]  # (..., 3)
    v1 = F.normalize(u1, p=2, dim=-1)  # (..., 3)
    v2 = F.normalize(u2 - project_v2v(u2, v1), p=2, dim=-1)
    return torch.cat([v1, v2], dim=-1)


def reconstruct_backbone(R, t, aa, chain_nb, res_nb, mask):
    """
    Args:
        R:  (N, L, 3, 3)
        t:  (N, L, 3)
        aa: (N, L)
        chain_nb:   (N, L)
        res_nb:     (N, L)
        mask:       (N, L)
    Returns:
        Reconstructed backbone atoms, (N, L, 4, 3).
    """
    N, L = aa.size()
    # atom_coords = restype_heavyatom_rigid_group_positions.clone().to(t) # (21, 14, 3)
    bb_coords = backbone_atom_coordinates_tensor.clone().to(t)  # (21, 3, 3)
    oxygen_coord = bb_oxygen_coordinate_tensor.clone().to(t)  # (21, 3)
    aa = aa.clamp(min=0, max=20)  # 20 for UNK

    bb_coords = bb_coords[aa.flatten()].reshape(N, L, -1, 3)  # (N, L, 3, 3)
    oxygen_coord = oxygen_coord[aa.flatten()].reshape(N, L, -1)  # (N, L, 3)
    bb_pos = local_to_global(R, t, bb_coords)  # Global coordinates of N, CA, C. (N, L, 3, 3).

    # Compute PSI angle
    bb_dihedral, _ = get_backbone_dihedral_angles(bb_pos, chain_nb, res_nb, mask)
    psi = bb_dihedral[..., 2]  # (N, L)
    # Make rotation matrix for PSI
    sin_psi = torch.sin(psi).reshape(N, L, 1, 1)
    cos_psi = torch.cos(psi).reshape(N, L, 1, 1)
    zero = torch.zeros_like(sin_psi)
    one = torch.ones_like(sin_psi)
    row1 = torch.cat([one, zero, zero], dim=-1)  # (N, L, 1, 3)
    row2 = torch.cat([zero, cos_psi, -sin_psi], dim=-1)  # (N, L, 1, 3)
    row3 = torch.cat([zero, sin_psi, cos_psi], dim=-1)  # (N, L, 1, 3)
    R_psi = torch.cat([row1, row2, row3], dim=-2)  # (N, L, 3, 3)

    # Compute rotoation and translation of PSI frame, and position of O.
    R_psi, t_psi = compose_chain(
        [
            (R, t),  # Backbone
            (R_psi, torch.zeros_like(t)),  # PSI angle
        ]
    )
    O_pos = local_to_global(R_psi, t_psi, oxygen_coord.reshape(N, L, 1, 3))

    bb_pos = torch.cat([bb_pos, O_pos], dim=2)  # (N, L, 4, 3)
    return bb_pos


def reconstruct_backbone_partially(
    pos_ctx, R_new, t_new, aa, chain_nb, res_nb, mask_atoms, mask_recons
):
    """
    Args:
        pos:    (N, L, A, 3).
        R_new:  (N, L, 3, 3).
        t_new:  (N, L, 3).
        mask_atoms: (N, L, A).
        mask_recons:(N, L).
    Returns:
        pos_new:    (N, L, A, 3).
        mask_new:   (N, L, A).
    """
    N, L, A = mask_atoms.size()

    mask_res = mask_atoms[:, :, BBHeavyAtom.CA]
    pos_recons = reconstruct_backbone(R_new, t_new, aa, chain_nb, res_nb, mask_res)  # (N, L, 4, 3)
    pos_recons = F.pad(pos_recons, pad=(0, 0, 0, A - 4), value=0)  # (N, L, A, 3)

    pos_new = torch.where(
        mask_recons[:, :, None, None].expand_as(pos_ctx), pos_recons, pos_ctx
    )  # (N, L, A, 3)

    mask_bb_atoms = torch.zeros_like(mask_atoms)
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(mask_recons[:, :, None].expand_as(mask_atoms), mask_bb_atoms, mask_atoms)

    return pos_new, mask_new


# ---- models_con/utils.py ----


def process_dic(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "module" in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# ---- models_con/edge.py ----


class EdgeEmbedder(nn.Module):
    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32, num_bins=16):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.num_bins = num_bins
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(
            self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms
        )
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms * max_num_atoms, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # Phi and Psi

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(
        self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None
    ):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            trans, sc_trans: (N,L,3)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, : self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, : self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        pair_structure_mask = (
            structure_mask[:, :, None] * structure_mask[:, None, :]
            if structure_mask is not None
            else None
        )

        # Pair identities
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Relative sequential positions
        same_chain = chain_nb[:, :, None] == chain_nb[:, None, :]
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos,
            max=self.max_relpos,
        )  # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        # Distances
        d = angstrom_to_nm(
            torch.linalg.norm(
                pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
                dim=-1,
                ord=2,
            )
        ).reshape(N, L, L, -1)  # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (
            mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]
        ).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        # # trans embed
        # dist_feats = calc_distogram(
        #     trans, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins)
        # if sc_trans == None:
        #     sc_trans = torch.zeros_like(trans)
        # sc_feats = calc_distogram(
        #     sc_trans, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins)

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)  # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all


# ---- models_con/node.py ----


class NodeEmbedder(nn.Module):
    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.feat_dim = feat_dim
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()

        infeat_dim = (
            feat_dim + (self.max_aa_types * max_num_atoms * 3) + self.dihed_embed.get_out_dim(3)
        )
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    # def embed_t(self, timesteps, mask):
    #     timestep_emb = get_time_embedding(
    #         timesteps[:, 0],
    #         self.feat_dim,
    #         max_positions=2056
    #     )[:, None, :].repeat(1, mask.shape[1], 1)
    #     return timestep_emb

    def forward(
        self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None
    ):
        """
        Args:
            aa:         (N, L).
            res_nb:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).
            mask_atoms: (N, L, A).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, : self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, : self.max_num_atoms]

        # Amino acid identity features
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa)  # (N, L, feat)

        # Coordinate features
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA],
            pos_atoms[:, :, BBHeavyAtom.C],
            pos_atoms[:, :, BBHeavyAtom.N],
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)  # (N, L, A, 3)
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))

        aa_expand = aa[:, :, None, None, None].expand(
            N, L, self.max_aa_types, self.max_num_atoms, 3
        )
        rng_expand = (
            torch.arange(0, self.max_aa_types)[None, None, :, None, None]
            .expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
            .to(aa_expand)
        )
        place_mask = aa_expand == rng_expand
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types * self.max_num_atoms * 3)
        if structure_mask is not None:
            # Avoid data leakage at training time
            crd_feat = crd_feat * structure_mask[:, :, None]

        # Backbone dihedral features
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(
            pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue
        )
        dihed_feat = (
            self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]
        )  # (N, L, 3, dihed/3)
        dihed_feat = dihed_feat.reshape(N, L, -1)
        if structure_mask is not None:
            # Avoid data leakage at training time
            dihed_mask = torch.logical_and(
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1),
                    torch.roll(structure_mask, shifts=-1, dims=1),
                ),
            )  # Avoid slight data leakage via dihedral angles of anchor residues
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

        # # timestep
        # timestep_emb = self.embed_t(timesteps, mask_residue)

        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat], dim=-1))  # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]

        # print(f'aa_seq:{aa},aa:{aa_feat},crd:{crd_feat},dihed:{dihed_feat},time:{timestep_emb}')

        # print(f'weight:{self.aatype_embed.weight}') # nan, why?

        return out_feat


# ---- models_con/ipa_pytorch.py ("Modified code of Openfold's IPA") ----
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified code of Openfold's IPA."""


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


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


class IPALinear(nn.Linear):
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
        super(IPALinear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

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
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = IPALinear(self.c, self.c, init="relu")
        self.linear_2 = IPALinear(self.c, self.c, init="relu")
        self.linear_3 = IPALinear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
        self, *, node_embed_size, edge_embed_in, edge_embed_out, num_layers=2, node_dilation=2
    ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = IPALinear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(IPALinear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = IPALinear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            axis=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(
            batch_size * num_res**2, -1
        )
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
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
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = IPALinear(self.c_s, hc)
        self.linear_kv = IPALinear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = IPALinear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = IPALinear(self.c_s, hpkv)

        self.linear_b = IPALinear(self.c_z, self.no_heads)
        self.down_z = IPALinear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = IPALinear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
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
        if _offload_inference:
            z = _z_reference_list
        else:
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
        b = self.linear_b(z[0])

        if _offload_inference:
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2

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
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)

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
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s


class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = IPALinear(self.c, self.c, init="relu")
        self.linear_2 = IPALinear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = IPALinear(self.c, self.c, init="final")
        self.linear_final = IPALinear(self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class RotationVFLayer(nn.Module):
    def __init__(self, dim):
        super(RotationVFLayer, self).__init__()

        self.linear_1 = IPALinear(dim, dim, init="relu")
        self.linear_2 = IPALinear(dim, dim, init="relu")
        self.linear_3 = IPALinear(dim, dim)
        self.final_linear = IPALinear(dim, 6, init="final")
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return self.final_linear(s)


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, use_rot_updates):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self._use_rot_updates = use_rot_updates
        update_dim = 6 if use_rot_updates else 3
        self.linear = IPALinear(self.c_s, update_dim, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class IpaScore(nn.Module):
    def __init__(self, model_conf, diffuser):
        super(IpaScore, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.diffuser = diffuser

        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        for b in range(ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(ipa_conf.c_s)
            self.trunk[f"skip_embed_{b}"] = IPALinear(
                self._model_conf.node_embed_size, self._ipa_conf.c_skip, init="final"
            )
            tfmr_in = ipa_conf.c_s + self._ipa_conf.c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, ipa_conf.seq_tfmr_num_layers
            )
            self.trunk[f"post_tfmr_{b}"] = IPALinear(tfmr_in, ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=ipa_conf.c_s)
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(ipa_conf.c_s)

            if b < ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.torsion_pred = TorsionAngles(ipa_conf.c_s, 1)

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats["res_mask"].type(torch.float32)
        diffuse_mask = (1 - input_feats["fixed_mask"].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats["rigids_t"].type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)
        init_rots = init_rigids.get_rots()

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat(
                [node_embed, self.trunk[f"skip_embed_{b}"](init_node_embed)], dim=-1
            )
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, diffuse_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots(), curr_rigids.get_rots(), input_feats["t"]
        )
        rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            input_feats["t"][:, None, None],
            use_torch=True,
        )
        trans_score = trans_score * node_mask[..., None]
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            "psi": psi_pred,
            "rot_score": rot_score,
            "trans_score": trans_score,
            "final_rigids": curr_rigids,
        }
        return model_out


# ---- data/utils.py: create_rigid (verbatim, 2-line helper; the only piece
# needed from a file that otherwise imports torch_scatter for unrelated code) ----
def create_rigid(rots, trans):
    rots = Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


# ---- models_con/ga.py ----


class GAEncoder(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(
            num_funcs=12
        )  # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5),
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20),
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(
                3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5),
                self._ipa_conf.c_s,
            ),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False
            )
            self.trunk[f"post_tfmr_{b}"] = IPALinear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(timesteps[:, 0], self.feat_dim, max_positions=2056)[
            :, None, :
        ].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(
        self,
        t,
        rotmats_t,
        trans_t,
        angles_t,
        seqs_t,
        node_embed,
        edge_embed,
        generate_mask,
        res_mask,
    ):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(
            torch.cat(
                [
                    node_embed,
                    self.current_seq_embedder(seqs_t),
                    self.embed_t(t, node_mask),
                    self.angles_embedder(angles_t).reshape(num_batch, num_res, -1),
                ],
                dim=-1,
            )
        )
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).bool()
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob


# ---- Menagerie staging wrapper ----
#
# The real repo's trainable forward pass for one diffusion step is
# `FlowModel.encode()` (NodeEmbedder + EdgeEmbedder to build node/edge features from a
# structure) followed by `GAEncoder.forward()` (the IPA + seq-transformer trunk that
# FlowModel.forward()/sample() call at every timestep). `FlowModel` itself cannot be
# constructed here because `models_con/flow_model.py` imports `data.so3_utils` and
# `data.all_atom`, which import `torch_scatter` at module level even though the
# training-loop-only code paths that need them (interpolation/scoring, not the
# encoder trunk) are irrelevant to a single structural forward pass. This wrapper
# composes the three REAL modules exactly as FlowModel.encode()/GAEncoder.forward()
# do, with a single concrete-tensor-friendly forward signature.
class PepFlowEncoder(nn.Module):
    def __init__(self, ipa_conf, node_embed_size=128, edge_embed_size=64, max_num_heavyatoms=15):
        super().__init__()
        self.node_embedder = NodeEmbedder(node_embed_size, max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(edge_embed_size, max_num_heavyatoms)
        self.ga_encoder = GAEncoder(ipa_conf)

    def forward(self, aa, res_nb, chain_nb, pos_heavyatom, mask_heavyatom, t, seqs_t):
        """
        Args (matches the real FlowModel.encode() + GAEncoder.forward() signature):
            aa:             (N, L) long, amino-acid indices (0-20).
            res_nb:         (N, L) long, residue numbering.
            chain_nb:       (N, L) long, chain numbering.
            pos_heavyatom:  (N, L, A, 3) float, heavy-atom coordinates.
            mask_heavyatom: (N, L, A) bool, heavy-atom presence mask.
            t:              (N, 1) float in (0, 1], flow-matching timestep.
            seqs_t:         (N, L) long, current (noised) sequence indices.
        Returns:
            pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob -- exactly the
            4-tuple GAEncoder.forward() returns in the real repo.
        """
        res_mask = mask_heavyatom[:, :, BBHeavyAtom.CA]
        node_embed = self.node_embedder(aa, res_nb, chain_nb, pos_heavyatom, mask_heavyatom)
        edge_embed = self.edge_embedder(aa, res_nb, chain_nb, pos_heavyatom, mask_heavyatom)

        rotmats_t = construct_3d_basis(
            pos_heavyatom[:, :, BBHeavyAtom.CA],
            pos_heavyatom[:, :, BBHeavyAtom.C],
            pos_heavyatom[:, :, BBHeavyAtom.N],
        )
        trans_t = pos_heavyatom[:, :, BBHeavyAtom.CA]
        angles_t = torch.zeros(aa.shape[0], aa.shape[1], 5, dtype=torch.float32)
        generate_mask = torch.zeros_like(res_mask)

        return self.ga_encoder(
            t,
            rotmats_t,
            trans_t,
            angles_t,
            seqs_t,
            node_embed,
            edge_embed,
            generate_mask,
            res_mask.float(),
        )


def _pepflow_ipa_conf():
    # Mirrors configs/learn_angle.yaml `model.encoder.ipa`, downsized for a fast trace
    # (c_s/c_z/heads/blocks are all shrunk; the architecture is unchanged).
    from types import SimpleNamespace

    return SimpleNamespace(
        c_s=32,
        c_z=16,
        c_hidden=16,
        no_heads=2,
        no_qk_points=4,
        no_v_points=4,
        seq_tfmr_num_heads=2,
        seq_tfmr_num_layers=1,
        num_blocks=2,
        stop_grad=False,
    )


def build_pepflow_encoder():
    return PepFlowEncoder(_pepflow_ipa_conf(), node_embed_size=32, edge_embed_size=16)


def example_input_pepflow_encoder():
    N, L, A = 1, 8, 15
    aa = torch.randint(0, 20, (N, L), dtype=torch.long)
    res_nb = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(N, L).clone()
    chain_nb = torch.zeros(N, L, dtype=torch.long)
    pos_heavyatom = torch.randn(N, L, A, 3)
    mask_heavyatom = torch.ones(N, L, A, dtype=torch.bool)
    t = torch.rand(N, 1)
    seqs_t = torch.randint(0, 20, (N, L), dtype=torch.long)
    return (aa, res_nb, chain_nb, pos_heavyatom, mask_heavyatom, t, seqs_t)


MENAGERIE_ENTRIES = [
    (
        "PepFlow-Encoder",
        "build_pepflow_encoder",
        "example_input_pepflow_encoder",
        2024,
        "ported-pytorch",
    ),
]
