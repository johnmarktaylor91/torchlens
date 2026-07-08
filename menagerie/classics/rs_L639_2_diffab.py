# SOURCE: vendored from luost26/diffab @ main
# Files combined: diffab/models/diffab.py (DiffusionAntibodyDesign), diffab/models/_base.py
# (register_model), diffab/modules/encoders/residue.py (ResidueEmbedding),
# diffab/modules/encoders/pair.py (PairEmbedding), diffab/modules/encoders/ga.py
# (GABlock, GAEncoder), diffab/modules/diffusion/dpm_full.py (EpsilonNet, FullDPM),
# diffab/modules/diffusion/transition.py (VarianceSchedule, PositionTransition,
# RotationTransition, AminoacidCategoricalTransition), diffab/modules/common/geometry.py,
# diffab/modules/common/so3.py, diffab/modules/common/layers.py,
# diffab/modules/common/topology.py, diffab/utils/protein/constants.py (AA, BBHeavyAtom,
# max_num_heavyatoms only -- the atom-coordinate reconstruction tables are unused by
# forward()/encode()).
# Only minimal changes: merged multiple source files into one module and dropped the
# unused sample()/optimize() trajectory-generation methods (forward()/encode() is the
# exercised path for a tiny random-init trace). Architecture (residue/pair embedding,
# geometric-attention (GA) encoder, SE(3) diffusion epsilon-net, SO(3)/position/sequence
# noise transitions) is untouched.
import enum
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- diffab/utils/protein/constants.py (subset actually used by encode()/forward()) ---
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


class BBHeavyAtom(enum.IntEnum):
    N = 0
    CA = 1
    C = 2
    O = 3  # noqa: E741
    CB = 4
    OXT = 14


max_num_heavyatoms = 15


# --- diffab/modules/common/topology.py ---
def get_consecutive_flag(chain_nb, res_nb, mask):
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()
    same_chain = chain_nb[:, 1:] == chain_nb[:, :-1]
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag


# --- diffab/modules/common/geometry.py ---
def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)
    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    mat = torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)
    return mat


def local_to_global(R, t, p):
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]
    p = p.view(N, L, -1, 3).transpose(-1, -2)
    q = torch.matmul(R, p) + t.unsqueeze(-1)
    q = q.transpose(-1, -2).reshape(p_size)
    return q


def global_to_local(R, t, q):
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]
    q = q.reshape(N, L, -1, 3).transpose(-1, -2)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))
    p = p.transpose(-1, -2).reshape(q_size)
    return p


def apply_rotation_to_vector(R, p):
    return local_to_global(R, torch.zeros_like(p), p)


def compose_rotation_and_translation(R1, t1, R2, t2):
    R_new = torch.matmul(R1, R2)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def quaternion_to_rotation_matrix(quaternions):
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


def quaternion_1ijk_to_rotation_matrix(q):
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


def dihedral_from_four_points(p0, p1, p2, p3):
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


def angstrom_to_nm(x):
    return x / 10


def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)
    omega_mask = torch.logical_not(N_term_flag)
    phi_mask = torch.logical_not(N_term_flag)
    psi_mask = torch.logical_not(C_term_flag)

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
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1),
        value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed
    return bb_dihedral, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    N, L = pos_atoms.shape[:2]
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
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


def repr_6d_to_rotation_matrix(x):
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = normalize_vector(a1, dim=-1)
    b2 = normalize_vector(a2 - project_v2v(a2, b1, dim=-1), dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    mat = torch.cat([b1.unsqueeze(-1), b2.unsqueeze(-1), b3.unsqueeze(-1)], dim=-1)
    return mat


# --- diffab/modules/common/so3.py ---
def log_rotation(R):
    trace = R[..., range(3), range(3)].sum(-1)
    if torch.is_grad_enabled():
        min_cos = -0.999
    else:
        min_cos = -1.0
    cos_theta = ((trace - 1) / 2).clamp_min(min=min_cos)
    sin_theta = torch.sqrt(1 - cos_theta**2)
    theta = torch.acos(cos_theta)
    coef = ((theta + 1e-8) / (2 * sin_theta + 2e-8))[..., None, None]
    logR = coef * (R - R.transpose(-1, -2))
    return logR


def skewsym_to_so3vec(S):
    x = S[..., 1, 2]
    y = S[..., 2, 0]
    z = S[..., 0, 1]
    w = torch.stack([x, y, z], dim=-1)
    return w


def so3vec_to_skewsym(w):
    x, y, z = torch.unbind(w, dim=-1)
    o = torch.zeros_like(x)
    S = torch.stack([o, z, -y, -z, o, x, y, -x, o], dim=-1).reshape(w.shape[:-1] + (3, 3))
    return S


def exp_skewsym(S):
    x = torch.linalg.norm(skewsym_to_so3vec(S), dim=-1)
    I = torch.eye(3).to(S).view([1 for _ in range(S.dim() - 2)] + [3, 3])  # noqa: E741
    sinx, cosx = torch.sin(x), torch.cos(x)
    b = (sinx + 1e-8) / (x + 1e-8)
    c = (1 - cosx + 1e-8) / (x**2 + 2e-8)
    S2 = S @ S
    return I + b[..., None, None] * S + c[..., None, None] * S2


def so3vec_to_rotation(w):
    return exp_skewsym(so3vec_to_skewsym(w))


def rotation_to_so3vec(R):
    logR = log_rotation(R)
    w = skewsym_to_so3vec(logR)
    return w


def random_uniform_so3(size, device="cpu"):
    q = F.normalize(torch.randn(list(size) + [4], device=device), dim=-1)
    return rotation_to_so3vec(quaternion_to_rotation_matrix(q))


class ApproxAngularDistribution(nn.Module):
    def __init__(self, stddevs, std_threshold=0.1, num_bins=8192, num_iters=1024):
        super().__init__()
        self.std_threshold = std_threshold
        self.num_bins = num_bins
        self.num_iters = num_iters
        self.register_buffer("stddevs", torch.FloatTensor(stddevs))
        self.register_buffer("approx_flag", self.stddevs <= std_threshold)
        self._precompute_histograms()

    @staticmethod
    def _pdf(x, e, L):
        x = x[:, None]
        c = (1 - torch.cos(x)) / math.pi
        ll = torch.arange(0, L)[None, :]
        a = (2 * ll + 1) * torch.exp(-ll * (ll + 1) * (e**2))
        b = (torch.sin((ll + 0.5) * x) + 1e-6) / (torch.sin(x / 2) + 1e-6)
        f = (c * a * b).sum(dim=1)
        return f

    def _precompute_histograms(self):
        X, Y = [], []
        for std in self.stddevs:
            std = std.item()
            x = torch.linspace(0, math.pi, self.num_bins)
            y = self._pdf(x, std, self.num_iters)
            y = torch.nan_to_num(y).clamp_min(0)
            X.append(x)
            Y.append(y)
        self.register_buffer("X", torch.stack(X, dim=0))
        self.register_buffer("Y", torch.stack(Y, dim=0))

    def sample(self, std_idx):
        size = std_idx.size()
        std_idx = std_idx.flatten()
        prob = self.Y[std_idx]
        bin_idx = torch.multinomial(prob[:, :-1], num_samples=1).squeeze(-1)
        bin_start = self.X[std_idx, bin_idx]
        bin_width = self.X[std_idx, bin_idx + 1] - self.X[std_idx, bin_idx]
        samples_hist = bin_start + torch.rand_like(bin_start) * bin_width

        mean_gaussian = self.stddevs[std_idx] * 2
        std_gaussian = self.stddevs[std_idx]
        samples_gaussian = mean_gaussian + torch.randn_like(mean_gaussian) * std_gaussian
        samples_gaussian = samples_gaussian.abs() % math.pi

        gaussian_flag = self.approx_flag[std_idx]
        samples = torch.where(gaussian_flag, samples_gaussian, samples_hist)
        return samples.reshape(size)


def random_normal_so3(std_idx, angular_distrib, device="cpu"):
    size = std_idx.size()
    u = F.normalize(torch.randn(list(size) + [3], device=device), dim=-1)
    theta = angular_distrib.sample(std_idx)
    w = u * theta[..., None]
    return w


# --- diffab/modules/common/layers.py ---
def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes)
    x = x.clamp(min=0, max=num_classes - 1)
    y = F.one_hot(x, num_classes) * mask[..., None]
    return y


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
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)
        code = torch.cat(
            [x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1
        )
        code = code.reshape(shape)
        return code


class LayerNorm(nn.Module):
    def __init__(self, normal_shape, gamma=True, beta=True, epsilon=1e-10):
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
            y = y * self.gamma
        if self.beta is not None:
            y = y + self.beta
        return y


# --- diffab/modules/encoders/residue.py ---
class ResidueEmbedding(nn.Module):
    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        self.type_embed = nn.Embedding(10, feat_dim, padding_idx=0)
        infeat_dim = (
            feat_dim
            + (self.max_aa_types * max_num_atoms * 3)
            + self.dihed_embed.get_out_dim(3)
            + feat_dim
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

    def forward(
        self,
        aa,
        res_nb,
        chain_nb,
        pos_atoms,
        mask_atoms,
        fragment_type,
        structure_mask=None,
        sequence_mask=None,
    ):
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]

        pos_atoms = pos_atoms[:, :, : self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, : self.max_num_atoms]

        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa)

        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA],
            pos_atoms[:, :, BBHeavyAtom.C],
            pos_atoms[:, :, BBHeavyAtom.N],
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)
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
            crd_feat = crd_feat * structure_mask[:, :, None]

        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(
            pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue
        )
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]
        dihed_feat = dihed_feat.reshape(N, L, -1)
        if structure_mask is not None:
            dihed_mask = torch.logical_and(
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1),
                    torch.roll(structure_mask, shifts=-1, dims=1),
                ),
            )
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

        type_feat = self.type_embed(fragment_type)

        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat, type_feat], dim=-1))
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat


# --- diffab/modules/encoders/pair.py ---
class PairEmbedding(nn.Module):
    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
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
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)

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
        N, L = aa.size()

        pos_atoms = pos_atoms[:, :, : self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, : self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        pair_structure_mask = (
            structure_mask[:, :, None] * structure_mask[:, None, :]
            if structure_mask is not None
            else None
        )

        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]
        feat_aapair = self.aa_pair_embed(aa_pair)

        same_chain = chain_nb[:, :, None] == chain_nb[:, None, :]
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :], min=-self.max_relpos, max=self.max_relpos
        )
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        d = angstrom_to_nm(
            torch.linalg.norm(
                pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :], dim=-1, ord=2
            )
        ).reshape(N, L, L, -1)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (
            mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]
        ).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
        if pair_structure_mask is not None:
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]

        dihed = pairwise_dihedrals(pos_atoms)
        feat_dihed = self.dihedral_embed(dihed)
        if pair_structure_mask is not None:
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all


# --- diffab/modules/encoders/ga.py ---
def _alpha_from_logits(logits, mask, inf=1e5):
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        pair_feat_dim,
        value_dim=32,
        query_key_dim=32,
        num_query_points=8,
        num_value_points=8,
        num_heads=12,
        bias=False,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        self.spatial_coef = nn.Parameter(
            torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.0) - 1.0)),
            requires_grad=True,
        )
        self.proj_query_point = nn.Linear(
            node_feat_dim, num_query_points * num_heads * 3, bias=bias
        )
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(
            node_feat_dim, num_value_points * num_heads * 3, bias=bias
        )

        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim)
            + (num_heads * value_dim)
            + (num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim),
            nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim),
            nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim),
        )
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)
        logits_node = (
            query_l.unsqueeze(2) * key_l.unsqueeze(1) * (1 / np.sqrt(self.query_key_dim))
        ).sum(-1)
        return logits_node

    def _pair_logits(self, z):
        return self.proj_pair_bias(z)

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points, 3)
        query_points = local_to_global(R, t, query_points)
        query_s = query_points.reshape(N, L, self.num_heads, -1)

        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points, 3)
        key_points = local_to_global(R, t, key_points)
        key_s = key_points.reshape(N, L, self.num_heads, -1)

        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points))) / 2)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)
        feat_node = feat_node.sum(dim=2)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points, 3)
        value_points = local_to_global(
            R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points, 3)
        )
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * value_points.unsqueeze(1)
        aggr_points = aggr_points.sum(dim=2)

        feat_points = global_to_local(R, t, aggr_points)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat(
            [
                feat_points.reshape(N, L, -1),
                feat_distance.reshape(N, L, -1),
                feat_direction.reshape(N, L, -1),
            ],
            dim=-1,
        )
        return feat_spatial

    def forward(self, R, t, x, z, mask):
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)

        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class GAEncoder(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GABlock(node_feat_dim, pair_feat_dim, **ga_block_opt) for _ in range(num_layers)]
        )

    def forward(self, R, t, res_feat, pair_feat, mask):
        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, mask)
        return res_feat


# --- diffab/modules/diffusion/transition.py ---
class VarianceSchedule(nn.Module):
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        f_t = torch.cos((np.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("sigmas", sigmas)


class PositionTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        e_rand = torch.randn_like(p_0)
        p_noisy = c0 * p_0 + c1 * e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)
        return p_noisy, e_rand

    def denoise(self, p_t, eps_p, mask_generate, t):
        alpha = self.var_sched.alphas[t].clamp_min(self.var_sched.alphas[-2])
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).view(-1, 1, 1)
        c1 = ((1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).view(-1, 1, 1)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t), torch.randn_like(p_t), torch.zeros_like(p_t)
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        return p_next


class RotationTransition(nn.Module):
    def __init__(
        self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}
    ):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(
            sigma.tolist(), **angular_distrib_inv_opt
        )

        self.register_buffer("_dummy", torch.empty([0]))

    def add_noise(self, v_0, mask_generate, t):
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        # c1 computed but unused in the original repo code too (dead code, kept faithfully).

        e_scaled = random_normal_so3(
            t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device
        )
        E_scaled = so3vec_to_rotation(e_scaled)

        R0_scaled = so3vec_to_rotation(c0 * v_0)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t):
        N, L = mask_generate.size()
        e = random_normal_so3(
            t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device
        )
        e = torch.where((t > 1)[:, None, None].expand(N, L, 3), e, torch.zeros_like(e))
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next


class AminoacidCategoricalTransition(nn.Module):
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        N, L, K = c.size()
        c = c.view(N * L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t):
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]
        c_noisy = (alpha_bar * c_0) + ((1 - alpha_bar) / K)
        c_t = torch.where(mask_generate[..., None].expand(N, L, K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x_0, t):
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float()

        if x_0.dim() == 3:
            c_0 = x_0
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float()

        alpha = self.var_sched.alpha_bars[t][:, None, None]
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]

        theta = ((alpha * c_t) + (1 - alpha) / K) * ((alpha_bar * c_0) + (1 - alpha_bar) / K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, mask_generate, t):
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()
        post = self.posterior(c_t, c_0_pred, t=t)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)
        return post, x_next


# --- diffab/modules/diffusion/dpm_full.py ---
def rotation_matrix_cosine_loss(R_pred, R_true):
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)

    ones = torch.ones([ncol], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction="none")
    loss = loss.reshape(size + [3]).sum(dim=-1)
    return loss


class EpsilonNet(nn.Module):
    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 3),
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 3),
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 20),
            nn.Softmax(dim=-1),
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t)

        res_feat = self.res_feat_mixer(
            torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)
        )
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(
            N, L, 3
        )
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        eps_crd = self.eps_crd_net(in_feat)
        eps_pos = apply_rotation_to_vector(R, eps_crd)
        eps_pos = torch.where(
            mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos)
        )

        eps_rot = self.eps_rot_net(in_feat)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        c_denoised = self.eps_seq_net(in_feat)

        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):
    def __init__(
        self,
        res_feat_dim,
        pair_feat_dim,
        num_steps,
        eps_net_opt={},
        trans_rot_opt={},
        trans_pos_opt={},
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer("position_mean", torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer("position_scale", torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer("_dummy", torch.empty([0]))

    def _normalize_position(self, p):
        return (p - self.position_mean) / self.position_scale

    def _unnormalize_position(self, p_norm):
        return p_norm * self.position_scale + self.position_mean

    def forward(
        self,
        v_0,
        p_0,
        s_0,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        denoise_structure,
        denoise_sequence,
        t=None,
    ):
        N, L = res_feat.shape[:2]
        if t is None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )

        loss_dict = {}

        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict["rot"] = loss_rot

        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction="none").sum(dim=-1)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict["pos"] = loss_pos

        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, target=post_true, reduction="none", log_target=False
        ).sum(dim=-1)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict["seq"] = loss_seq

        return loss_dict


# --- diffab/models/_base.py ---
_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls

    return decorator


# --- diffab/models/diffab.py ---
resolution_to_num_atoms = {"backbone+CB": 5, "full": max_num_heavyatoms}


@register_model("diffab")
class DiffusionAntibodyDesign(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get("resolution", "full")]
        self.residue_embed = ResidueEmbedding(cfg["res_feat_dim"], num_atoms)
        self.pair_embed = PairEmbedding(cfg["pair_feat_dim"], num_atoms)

        self.diffusion = FullDPM(
            cfg["res_feat_dim"],
            cfg["pair_feat_dim"],
            **cfg["diffusion"],
        )

    def encode(self, batch, remove_structure, remove_sequence):
        context_mask = torch.logical_and(
            batch["mask_heavyatom"][:, :, BBHeavyAtom.CA],
            ~batch["generate_flag"],
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa=batch["aa"],
            res_nb=batch["res_nb"],
            chain_nb=batch["chain_nb"],
            pos_atoms=batch["pos_heavyatom"],
            mask_atoms=batch["mask_heavyatom"],
            fragment_type=batch["fragment_type"],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa=batch["aa"],
            res_nb=batch["res_nb"],
            chain_nb=batch["chain_nb"],
            pos_atoms=batch["pos_heavyatom"],
            mask_atoms=batch["mask_heavyatom"],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        R = construct_3d_basis(
            batch["pos_heavyatom"][:, :, BBHeavyAtom.CA],
            batch["pos_heavyatom"][:, :, BBHeavyAtom.C],
            batch["pos_heavyatom"][:, :, BBHeavyAtom.N],
        )
        p = batch["pos_heavyatom"][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p

    def forward(self, batch):
        mask_generate = batch["generate_flag"]
        mask_res = batch["mask"]
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=self.cfg.get("train_structure", True),
            remove_sequence=self.cfg.get("train_sequence", True),
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch["aa"]

        loss_dict = self.diffusion(
            v_0,
            p_0,
            s_0,
            res_feat,
            pair_feat,
            mask_generate,
            mask_res,
            denoise_structure=self.cfg.get("train_structure", True),
            denoise_sequence=self.cfg.get("train_sequence", True),
        )
        return loss_dict


# --- tiny build/example helpers ---
def build_diffab():
    cfg = {
        "res_feat_dim": 16,
        "pair_feat_dim": 16,
        "resolution": "backbone+CB",
        "train_structure": True,
        "train_sequence": True,
        "diffusion": {
            "num_steps": 20,
            "eps_net_opt": {
                "num_layers": 1,
                "encoder_opt": {
                    "ga_block_opt": {
                        "num_heads": 2,
                        "value_dim": 4,
                        "query_key_dim": 4,
                        "num_query_points": 2,
                        "num_value_points": 2,
                    }
                },
            },
        },
    }
    model = DiffusionAntibodyDesign(cfg)
    model.eval()
    return model


def example_input_diffab():
    torch.manual_seed(0)
    N, L, A = 1, 12, 5  # backbone+CB resolution -> 5 heavy atoms per residue
    aa = torch.randint(0, 20, (N, L))
    res_nb = torch.arange(1, L + 1).unsqueeze(0).expand(N, -1).clone()
    chain_nb = torch.zeros(N, L, dtype=torch.long)
    pos_heavyatom = torch.randn(N, L, A, 3)
    mask_heavyatom = torch.ones(N, L, A, dtype=torch.bool)
    fragment_type = torch.ones(N, L, dtype=torch.long)  # 1 = Heavy chain
    generate_flag = torch.zeros(N, L, dtype=torch.bool)
    generate_flag[:, L // 2 : L // 2 + 3] = True  # CDR-like region to denoise
    mask = torch.ones(N, L, dtype=torch.bool)

    batch = {
        "aa": aa,
        "res_nb": res_nb,
        "chain_nb": chain_nb,
        "pos_heavyatom": pos_heavyatom,
        "mask_heavyatom": mask_heavyatom,
        "fragment_type": fragment_type,
        "generate_flag": generate_flag,
        "mask": mask,
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    ("DiffAb", build_diffab, example_input_diffab, 2022, MENAGERIE_ZOO),
]
