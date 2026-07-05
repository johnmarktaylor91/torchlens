# FAITHFUL PORT of hbb1/2d-gaussian-splatting @ 335ad612f2e783a4e57b9cbc4d1e167bd599fc98
# (scene/gaussian_model.py, utils/general_utils.py, utils/sh_utils.py,
# utils/graphics_utils.py, gaussian_renderer/__init__.py) and its rasterizer submodule
# hbb1/diff-surfel-rasterization @ e0ed0207b3e0669960cfad70852200a4a5847f61
# (cuda_rasterizer/forward.cu) (original framework: PyTorch host code + a custom CUDA/GLM
# rasterization kernel)
#
# Huang, Yu, Chen, Gao 2024 "2D Gaussian Splatting for Geometrically Accurate Radiance
# Fields" (SIGGRAPH 2024, arXiv:2403.17888). 2DGS flattens each 3D Gaussian ellipsoid into
# an oriented 2D disk ("surfel": 2 scale params + a quaternion orientation, not 3), then
# renders it by intersecting each camera ray with the disk's local (u,v) tangent plane
# (a projective "2D-to-2D homography", Eqs. 8-10 of the paper) instead of splatting a
# projected 3D covariance ellipse -- a distinct primitive/rasterization mechanism from
# vanilla 3D Gaussian Splatting.
#
# RUNG-3 JUSTIFICATION: the real renderer (`diff_surfel_rasterization.GaussianRasterizer`)
# is a hand-written CUDA/GLM kernel (cuda_rasterizer/forward.cu + backward.cu, ~40KB) built
# as a torch C++/CUDA extension; it is not a base-lib import and is not reasonably
# installable for a menagerie snapshot build (custom CUDA compile, GPU-arch-pinned,
# unversioned submodule). Per the source ladder this is the "custom-CUDA" rung-3 case:
# faithful port, not vendor.
#
# WHAT IS PORTED, AND FROM WHERE (every mechanism below is transcribed from real code, not
# guessed from the paper):
#   - `build_rotation`, `build_scaling_rotation`, `build_covariance_from_scaling_rotation`:
#     copied near-verbatim from utils/general_utils.py + scene/gaussian_model.py
#     (`GaussianModel.setup_functions`/`get_covariance`) -- REAL PYTHON CODE, only the
#     hardcoded `device="cuda"` was generalized to the input tensor's device.
#   - `eval_sh`: copied verbatim from utils/sh_utils.py (REAL PYTHON CODE); the CUDA
#     `computeColorFromSH` in forward.cu evaluates the identical polynomial and then does
#     `+0.5` and clamps to `>=0`, which is applied here to match the rasterizer's default
#     (non-precomputed-color) path exactly.
#   - `getWorld2View2`, `getProjectionMatrix`: copied verbatim from utils/graphics_utils.py
#     (REAL PYTHON CODE), used to build a real, valid camera pose for the trace input.
#   - The per-gaussian screen-space transform matrix T (whose columns forward.cu calls
#     Tu/Tv/Tw) is computed via the exact formula the repo ITSELF uses on the Python side
#     when `pipe.compute_cov3D_python=True` (gaussian_renderer/__init__.py):
#     `(splat2world[:, [0,1,3]] @ world2pix[:, [0,1,3]])` -- this is real repo Python code,
#     not a re-derivation of the CUDA GLM math.
#   - `_point_image_2dgs` (projected splat center) and `_rasterize_2dgs` (the per-pixel
#     ray-splat intersection + alpha-compositing loop) are TRANSCRIBED from
#     cuda_rasterizer/forward.cu's `compute_aabb` and `renderCUDA` kernels: the k/l/p
#     cross-product intersection (Eqs. 8-10), the low-pass filter `rho = min(rho3d, rho2d)`
#     with `FilterInvSquare = 2.0`, the near-plane cutoff `near_n = 0.2`, the
#     `alpha = min(0.99, opacity * exp(-0.5*rho))` opacity falloff, the `DUAL_VISIABLE`
#     camera-facing normal flip, and the sequential front-to-back alpha-compositing
#     recurrence are all preserved exactly.
#   - DROPPED (GPU-parallelism scaffolding only, not architecture -- dropping it changes
#     performance, not the rendering result): forward.cu's per-tile Gaussian binning/radix
#     sort is replaced by a single global depth sort (mathematically equivalent painter's
#     algorithm since compositing order only depends on depth, not tile membership); the
#     `test_T < 0.0001` early-exit is replaced by evaluating every gaussian at every pixel
#     (the same math, at most negligibly more precise, since deeper terms already contract
#     to ~0 contribution once T is small); the median-depth/distortion regularization
#     outputs (training-time losses) are not computed, only render/alpha/normal/depth maps.
"""2D Gaussian Splatting: oriented-2D-disk ("surfel") radiance field with a ray-splat
homography rasterizer (faithful port of the real CUDA rasterizer's math into dense,
untiled pure-torch ops)."""

import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

FILTER_INV_SQUARE = 2.0  # auxiliary.h: FilterInvSquare
NEAR_N = 0.2  # auxiliary.h: near_n
CUTOFF = 3.0  # forward.cu preprocessCUDA: cutoff (TIGHTBBOX disabled, so fixed 3.0)

# spherical harmonics constants (sh_utils.py)
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]


# --- vendored (near-verbatim, device generalized) from utils/general_utils.py ---
def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    rr = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - rr * z)
    R[:, 0, 2] = 2 * (x * z + rr * y)
    R[:, 1, 0] = 2 * (x * y + rr * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - rr * x)
    R[:, 2, 0] = 2 * (x * z - rr * y)
    R[:, 2, 1] = 2 * (y * z + rr * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


# --- vendored (near-verbatim) from scene/gaussian_model.py's
# GaussianModel.setup_functions / GaussianModel.get_covariance ---
def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
    # 2DGS surfel: scaling has only 2 components (the disk's 2 in-plane axes); the 3rd
    # ("normal") axis is left at scale 1, matching `torch.cat([scaling*mod, ones_like],-1)`.
    RS = build_scaling_rotation(
        torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation
    ).permute(0, 2, 1)
    trans = torch.zeros((center.shape[0], 4, 4), dtype=center.dtype, device=center.device)
    trans[:, :3, :3] = RS
    trans[:, 3, :3] = center
    trans[:, 3, 3] = 1
    return trans


# --- vendored verbatim from utils/sh_utils.py (only degrees 0-2 kept; 2DGS random-init
# build below uses sh_degree=1) ---
def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics at unit directions using hardcoded SH polynomials.
    deg: int SH deg (0-2 supported here). sh: [..., C, (deg+1)**2]. dirs: [..., 3]."""
    assert 0 <= deg <= 2
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = _SH_C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - _SH_C1 * y * sh[..., 1] + _SH_C1 * z * sh[..., 2] - _SH_C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + _SH_C2[0] * xy * sh[..., 4]
                + _SH_C2[1] * yz * sh[..., 5]
                + _SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + _SH_C2[3] * xz * sh[..., 7]
                + _SH_C2[4] * (xx - yy) * sh[..., 8]
            )
    return result


# --- vendored verbatim from utils/graphics_utils.py ---
def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    import math

    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# --- transcribed from cuda_rasterizer/forward.cu's compute_aabb (only the projected
# splat center is needed here; the AABB extent is a tile-culling detail we drop, see
# module header) ---
def _point_image_2dgs(Tu, Tv, Tw, cutoff=CUTOFF):
    t0, t1, t2 = cutoff * cutoff, cutoff * cutoff, -1.0
    d = t0 * (Tw[:, 0] * Tw[:, 0]) + t1 * (Tw[:, 1] * Tw[:, 1]) + t2 * (Tw[:, 2] * Tw[:, 2])
    d = torch.where(d == 0, torch.full_like(d, 1e-12), d)
    f0, f1, f2 = t0 / d, t1 / d, t2 / d
    p_x = f0 * (Tu[:, 0] * Tw[:, 0]) + f1 * (Tu[:, 1] * Tw[:, 1]) + f2 * (Tu[:, 2] * Tw[:, 2])
    p_y = f0 * (Tv[:, 0] * Tw[:, 0]) + f1 * (Tv[:, 1] * Tw[:, 1]) + f2 * (Tv[:, 2] * Tw[:, 2])
    return torch.stack([p_x, p_y], dim=-1)  # [N,2]


# --- transcribed from cuda_rasterizer/forward.cu's renderCUDA (dense/untiled; see module
# header for what was dropped) ---
def _rasterize_2dgs(
    xyz,
    scaling,
    rotation,
    opacity,
    features_dc,
    features_rest,
    world_view_transform,
    full_proj_transform,
    camera_center,
    image_height,
    image_width,
    bg_color,
    sh_degree,
    scaling_modifier=1.0,
):
    device, dtype = xyz.device, xyz.dtype
    n_gauss = xyz.shape[0]
    h, w = image_height, image_width

    # per-gaussian screen-space transform T (real Python `compute_cov3D_python` formula
    # from gaussian_renderer/__init__.py)
    splat2world = build_covariance_from_scaling_rotation(xyz, scaling, scaling_modifier, rotation)
    near, far = 0.01, 100.0
    ndc2pix = torch.tensor(
        [
            [w / 2, 0, 0, (w - 1) / 2],
            [0, h / 2, 0, (h - 1) / 2],
            [0, 0, far - near, near],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=dtype,
    ).T
    world2pix = full_proj_transform @ ndc2pix
    mat = splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]  # [N,3,3]
    # Tu/Tv/Tw are the "sensitivity to local u / local v / homogeneous-1" 3-vectors used by
    # forward.cu; their 3 components are NOT spatial x/y/z, they are generic homogeneous
    # (screen_x, screen_y, w) sensitivities -- matching the CUDA code's opaque float3 usage.
    t_u, t_v, t_w = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]  # each [N,3]

    point_image = _point_image_2dgs(t_u, t_v, t_w)  # [N,2]

    # per-gaussian view-space depth + camera-facing normal flip (DUAL_VISIABLE)
    ones = torch.ones(n_gauss, 1, device=device, dtype=dtype)
    xyz_view = torch.cat([xyz, ones], dim=-1) @ world_view_transform  # [N,4]
    rot_mat = build_rotation(rotation)  # [N,3,3]
    normal_world = rot_mat[:, :, 2]  # disk normal = local z axis = 3rd column of R (scale=1)
    normal_view = normal_world @ world_view_transform[:3, :3]
    cos = -(xyz_view[:, :3] * normal_view).sum(-1)
    sign = torch.where(cos > 0, torch.ones_like(cos), -torch.ones_like(cos))
    normal_view = normal_view * sign.unsqueeze(-1)

    # per-gaussian color from SH (real `eval_sh`; matches computeColorFromSH's default path)
    dirs = xyz - camera_center.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    shs = torch.cat([features_dc, features_rest], dim=1)  # [N,(deg+1)^2,3]
    colors = eval_sh(sh_degree, shs.transpose(1, 2), dirs)  # [N,3]
    colors = (colors + 0.5).clamp_min(0.0)

    # dense per-pixel ray-splat intersection (forward.cu renderCUDA, Eq. 8-10)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    px = xs.unsqueeze(0)  # [1,H,W]
    py = ys.unsqueeze(0)

    tu0, tu1, tu2 = (t_u[:, i].view(n_gauss, 1, 1) for i in range(3))
    tv0, tv1, tv2 = (t_v[:, i].view(n_gauss, 1, 1) for i in range(3))
    tw0, tw1, tw2 = (t_w[:, i].view(n_gauss, 1, 1) for i in range(3))

    k0 = px * tw0 - tu0
    k1 = px * tw1 - tu1
    k2 = px * tw2 - tu2
    l0 = py * tw0 - tv0
    l1 = py * tw1 - tv1
    l2 = py * tw2 - tv2

    p0 = k1 * l2 - k2 * l1
    p1 = k2 * l0 - k0 * l2
    p2 = k0 * l1 - k1 * l0

    p2_safe = torch.where(p2 == 0, torch.full_like(p2, 1e-12), p2)
    s0 = p0 / p2_safe
    s1 = p1 / p2_safe
    rho3d = s0 * s0 + s1 * s1

    d0 = point_image[:, 0].view(n_gauss, 1, 1) - px
    d1 = point_image[:, 1].view(n_gauss, 1, 1) - py
    rho2d = FILTER_INV_SQUARE * (d0 * d0 + d1 * d1)
    rho = torch.minimum(rho3d, rho2d)

    depth = s0 * tw0 + s1 * tw1 + tw2  # [N,H,W]
    power = -0.5 * rho

    valid = (p2 != 0) & (power <= 0.0) & (depth >= NEAR_N)
    alpha = torch.clamp(opacity.view(n_gauss, 1, 1) * torch.exp(power), max=0.99)
    alpha = torch.where(valid & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))

    # front-to-back alpha compositing (painter's algorithm; global depth sort replaces the
    # CUDA per-tile sort, see module header)
    order = torch.argsort(xyz_view[:, 2]).tolist()
    trans = torch.ones(h, w, device=device, dtype=dtype)
    color_accum = torch.zeros(3, h, w, device=device, dtype=dtype)
    normal_accum = torch.zeros(3, h, w, device=device, dtype=dtype)
    depth_accum = torch.zeros(h, w, device=device, dtype=dtype)
    for idx in order:
        a = alpha[idx]
        weight = a * trans
        color_accum = color_accum + colors[idx].view(3, 1, 1) * weight.unsqueeze(0)
        normal_accum = normal_accum + normal_view[idx].view(3, 1, 1) * weight.unsqueeze(0)
        depth_accum = depth_accum + depth[idx] * weight
        trans = trans * (1 - a)

    rendered = color_accum + trans.unsqueeze(0) * bg_color.view(3, 1, 1)
    alpha_map = 1.0 - trans
    return {
        "render": rendered,
        "alpha": alpha_map,
        "normal": normal_accum,
        "depth": depth_accum,
    }


class TwoDGaussianSplat(nn.Module):
    """Oriented-2D-disk radiance field: learnable Gaussian "surfel" parameters rendered via
    the ray-splat homography rasterizer, ported from diff-surfel-rasterization."""

    def __init__(
        self,
        num_gaussians=10,
        image_height=12,
        image_width=12,
        sh_degree=1,
        fovx=0.8,
        fovy=0.8,
        znear=0.01,
        zfar=100.0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.sh_degree = sh_degree

        gen = torch.Generator().manual_seed(0)
        self._xyz = nn.Parameter(torch.randn(num_gaussians, 3, generator=gen) * 0.5)
        self._scaling = nn.Parameter(
            torch.log(torch.rand(num_gaussians, 2, generator=gen) * 0.2 + 0.05)
        )
        rotation_init = torch.randn(num_gaussians, 4, generator=gen)
        self._rotation = nn.Parameter(rotation_init)
        self._opacity = nn.Parameter(torch.zeros(num_gaussians, 1))  # inverse_sigmoid(0.5) = 0
        num_sh = (sh_degree + 1) ** 2
        self._features_dc = nn.Parameter(torch.randn(num_gaussians, 1, 3, generator=gen) * 0.3)
        self._features_rest = nn.Parameter(torch.zeros(num_gaussians, num_sh - 1, 3))

        projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1)
        self.register_buffer("projection_matrix", projection_matrix)
        self.register_buffer("bg_color", torch.zeros(3))

    def forward(self, world_view_transform):
        full_proj_transform = (
            world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.to(world_view_transform.dtype).unsqueeze(0))
            .squeeze(0)
        )
        camera_center = world_view_transform.inverse()[3, :3]

        scaling = torch.exp(self._scaling)
        opacity = torch.sigmoid(self._opacity).squeeze(-1)

        out = _rasterize_2dgs(
            self._xyz,
            scaling,
            self._rotation,
            opacity,
            self._features_dc,
            self._features_rest,
            world_view_transform,
            full_proj_transform,
            camera_center,
            self.image_height,
            self.image_width,
            self.bg_color.to(world_view_transform.dtype),
            self.sh_degree,
        )
        return out["render"]


def build_2d_gaussian_splat():
    torch.manual_seed(0)
    return TwoDGaussianSplat(num_gaussians=10, image_height=12, image_width=12, sh_degree=1)


def example_input_2d_gaussian_splat():
    # a real camera extrinsic built with the repo's own getWorld2View2 (not an arbitrary
    # tensor): identity orientation, pulled back 4 units so the origin (where the random
    # gaussians are centered) is in front of the camera.
    r_np = np.eye(3, dtype=np.float32)
    t_np = np.array([0.0, 0.0, 4.0], dtype=np.float32)
    view = torch.from_numpy(getWorld2View2(r_np, t_np)).transpose(0, 1).float()
    return (view,)


MENAGERIE_ENTRIES = [
    (
        "2D Gaussian Splatting",
        "build_2d_gaussian_splat",
        "example_input_2d_gaussian_splat",
        2024,
        "ported",
    ),
]
