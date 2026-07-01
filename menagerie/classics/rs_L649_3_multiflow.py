# SOURCE: vendored from jasonkyuyim/multiflow @ main (+ openfold/utils/rigid_utils.py, the
# vendored OpenFold copy that ships inside the multiflow repo)
# Files: openfold/utils/rigid_utils.py (Rotation, Rigid), multiflow/models/ipa_pytorch.py
#        (InvariantPointAttention, StructureModuleTransition, EdgeTransition, BackboneUpdate,
#        Linear + init helpers -- the compute_angles free function is dropped: it is dead code
#        for the traced forward pass, its only caller in the real repo, and its sole reason for
#        importing multiflow.data.all_atom, which pulls in torch_scatter/Bio.PDB/openfold's
#        larger data pipeline that this base env does not have installed),
#        multiflow/models/node_feature_net.py, multiflow/models/edge_feature_net.py (both use
#        only get_index_embedding/get_time_embedding/calc_distogram from
#        multiflow/models/utils.py, inlined below -- the rest of that file needs mdtraj/pandas/
#        openfold.utils.superimposition for unrelated eval-metric functions),
#        multiflow/models/flow_model.py (FlowModel; ANG_TO_NM_SCALE/NM_TO_ANG_SCALE and
#        create_rigid are the only multiflow.data.utils symbols FlowModel needs, inlined below
#        verbatim from that file)
# MultiFlow: SE(3) flow-matching protein generative model (co-design of backbone structure +
# sequence). FlowModel is the real trainable nn.Module (a LightningModule in the original repo
# just wraps it for the training loop): an IPA + Transformer trunk operating on OpenFold-style
# Rigid backbone frames, alternating InvariantPointAttention, a TransformerEncoder, and
# BackboneUpdate rigid-composition steps. We trace FlowModel directly with the real repo's
# default model.yaml config (aatype_pred=False, so the NUM_TOKENS/MASK_TOKEN_INDEX branch --
# the only remaining multiflow.data.utils dependency -- is never exercised; both constants are
# still defined below, with their real values from openfold's amino-acid table, so the module
# parses standalone). Vendored verbatim aside from flattening relative imports.
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE
# Real values from openfold.np.residue_constants (restype_num, restypes_with_x.index('X')):
# 20 canonical amino acids + 'X' unknown-residue sentinel at index 20. Only exercised when
# aatype_pred_num_tokens == NUM_TOKENS + 1, which the traced default config below never hits.
NUM_TOKENS = 20
MASK_TOKEN_INDEX = 20


def create_rigid(rots, trans):
    rotation = Rotation(rot_mats=rots)
    return Rigid(rots=rotation, trans=trans)


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices."""
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
        emb = nn.functional.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


"""============================================================================================="""
""" openfold/utils/rigid_utils.py (Rotation, Rigid) """
"""============================================================================================="""


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


"""============================================================================================="""
""" multiflow/models/ipa_pytorch.py """
"""============================================================================================="""


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

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
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
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
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
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.dropout_prob = ipa_conf.dropout
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
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
        if self.dropout_prob > 0.0:
            a = self.dropout(a)
        if attention_mask is not None:
            a = a - (1 - attention_mask[:, None, :, :]) * 1e7
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

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(self.c, self.num_torsions * 2, init="final")

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

        self.linear_1 = Linear(dim, dim, init="relu")
        self.linear_2 = Linear(dim, dim, init="relu")
        self.linear_3 = Linear(dim, dim)
        self.final_linear = Linear(dim, 6, init="final")
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
        self.linear = Linear(self.c_s, update_dim, init="final")

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


"""============================================================================================="""
""" multiflow/models/node_feature_net.py """
"""============================================================================================="""


class NodeFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(
                21, self.c_s
            )  # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self.c_s + self._cfg.c_timestep_emb + self._cfg.aatype_pred_num_tokens
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(timesteps[:, 0], self.c_timestep_emb, max_positions=2056)[
            :, None, :
        ].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
        self,
        *,
        so3_t,
        r3_t,
        cat_t,
        res_mask,
        diffuse_mask,
        chain_index,
        pos,
        aatypes,
        aatypes_sc,
    ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask),
        ]
        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            input_feats.append(aatypes_sc)
        if self._cfg.embed_chain:
            input_feats.append(get_index_embedding(chain_index, self.c_pos_emb, max_len=100))
        return self.linear(torch.cat(input_feats, dim=-1))


"""============================================================================================="""
""" multiflow/models/edge_feature_net.py """
"""============================================================================================="""


class EdgeFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        #   c_s, c_p, relpos_k, template_type):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        if self._cfg.embed_chain:
            total_edge_feats += 1
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res, num_res, -1])
        )

    def forward(self, s, t, sc_t, p_mask, diffuse_mask, chain_idx):
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res]
        r = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self._cfg.embed_chain:
            rel_chain = (chain_idx[:, :, None] == chain_idx[:, None, :]).float()
            all_edge_feats.append(rel_chain[..., None])
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats


"""============================================================================================="""
""" multiflow/models/flow_model.py """
"""============================================================================================="""


class FlowModel(nn.Module):
    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        if self._model_conf.aatype_pred:
            node_embed_size = self._model_conf.node_embed_size
            self.aatype_pred_net = nn.Sequential(
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens),
            )

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
                dropout=self._model_conf.transformer_dropout,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_feats):
        node_mask = input_feats["res_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats["diffuse_mask"]
        chain_index = input_feats["chain_idx"]
        res_index = input_feats["res_idx"]
        so3_t = input_feats["so3_t"]
        r3_t = input_feats["r3_t"]
        cat_t = input_feats["cat_t"]
        trans_t = input_feats["trans_t"]
        rotmats_t = input_feats["rotmats_t"]
        aatypes_t = input_feats["aatypes_t"].long()
        trans_sc = input_feats["trans_sc"]
        aatypes_sc = input_feats["aatypes_sc"]

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            pos=res_index,
            aatypes=aatypes_t,
            aatypes_sc=aatypes_sc,
        )

        init_edge_embed = self.edge_feature_net(
            init_node_embed, trans_t, trans_sc, edge_mask, diffuse_mask, chain_index
        )

        # Initial rigids
        init_rigids = create_rigid(rotmats_t, trans_t)
        curr_rigids = create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None]
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        if self._model_conf.aatype_pred:
            pred_logits = self.aatype_pred_net(node_embed)
            pred_aatypes = torch.argmax(pred_logits, dim=-1)
            if self._model_conf.aatype_pred_num_tokens == NUM_TOKENS + 1:
                pred_logits_wo_mask = pred_logits.clone()
                pred_logits_wo_mask[:, :, MASK_TOKEN_INDEX] = -1e9
                pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
            else:
                pred_aatypes = torch.argmax(pred_logits, dim=-1)
        else:
            pred_aatypes = aatypes_t
            pred_logits = nn.functional.one_hot(
                pred_aatypes, num_classes=self._model_conf.aatype_pred_num_tokens
            ).float()
        return {
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
            "pred_logits": pred_logits,
            "pred_aatypes": pred_aatypes,
        }


class _SimpleConf:
    """Minimal attribute-access config container standing in for the real repo's OmegaConf/
    Hydra DictConfig (multiflow/configs/model.yaml), since Hydra is not part of this base env
    and the model code only ever does attribute access (`self._cfg.c_s`) on the config -- never
    dict methods. Values below are copied verbatim from the shipped default model.yaml."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, _SimpleConf(**v) if isinstance(v, dict) else v)


def _default_model_conf():
    node_embed_size = 16
    edge_embed_size = 12
    return _SimpleConf(
        node_embed_size=node_embed_size,
        edge_embed_size=edge_embed_size,
        symmetric=False,
        aatype_pred=False,
        transformer_dropout=0.0,
        aatype_pred_num_tokens=21,
        node_features=dict(
            c_s=node_embed_size,
            c_pos_emb=8,
            c_timestep_emb=8,
            max_num_res=2000,
            timestep_int=1000,
            embed_chain=False,
            embed_aatype=False,
            use_mlp=False,
            aatype_pred_num_tokens=21,
        ),
        edge_features=dict(
            single_bias_transition_n=2,
            c_s=node_embed_size,
            c_p=edge_embed_size,
            relpos_k=64,
            feat_dim=8,
            num_bins=6,
            self_condition=True,
            embed_chain=False,
            embed_diffuse_mask=True,
        ),
        ipa=dict(
            c_s=node_embed_size,
            c_z=edge_embed_size,
            c_hidden=4,
            no_heads=2,
            no_qk_points=2,
            no_v_points=2,
            seq_tfmr_num_heads=2,
            seq_tfmr_num_layers=1,
            num_blocks=2,
            dropout=0.0,
        ),
    )


MENAGERIE_ZOO = "vendored-pytorch"


def build_multiflow():
    torch.manual_seed(0)
    return FlowModel(_default_model_conf())


def example_input_multiflow():
    torch.manual_seed(0)
    batch, n_res = 1, 6
    res_mask = torch.ones(batch, n_res)
    diffuse_mask = torch.ones(batch, n_res)
    chain_idx = torch.zeros(batch, n_res, dtype=torch.long)
    res_idx = torch.arange(n_res).unsqueeze(0).repeat(batch, 1)
    so3_t = torch.rand(batch, 1)
    r3_t = torch.rand(batch, 1)
    cat_t = torch.rand(batch, 1)
    trans_t = torch.randn(batch, n_res, 3)
    rotmats_t = torch.eye(3).view(1, 1, 3, 3).repeat(batch, n_res, 1, 1)
    aatypes_t = torch.zeros(batch, n_res, dtype=torch.long)
    trans_sc = torch.zeros(batch, n_res, 3)
    aatypes_sc = torch.zeros(batch, n_res, 21)
    input_feats = {
        "res_mask": res_mask,
        "diffuse_mask": diffuse_mask,
        "chain_idx": chain_idx,
        "res_idx": res_idx,
        "so3_t": so3_t,
        "r3_t": r3_t,
        "cat_t": cat_t,
        "trans_t": trans_t,
        "rotmats_t": rotmats_t,
        "aatypes_t": aatypes_t,
        "trans_sc": trans_sc,
        "aatypes_sc": aatypes_sc,
    }
    return (input_feats,)


MENAGERIE_ENTRIES = [
    (
        "MultiFlow-FlowModel",
        "build_multiflow",
        "example_input_multiflow",
        2024,
        "SOURCE_AVAILABLE",
    ),
]
