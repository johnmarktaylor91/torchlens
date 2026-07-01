# SOURCE: vendored from leeyang/DRfold @ main (DeepE2EPotential module)
#
# DRfold (Li, Zhang et al., Nature Communications 2023) predicts RNA 3D structure
# end-to-end from a sequence + secondary-structure prior: an Evoformer-style trunk
# (48 stacked blocks of row/column MSA attention, outer-product-mean pair update, and
# triangle-multiplicative/-attention pair updates, in the AlphaFold2 lineage but built
# for RNA) produces single (`s`) and pair (`z`) representations with a recycling
# embedder feeding the previous cycle's predicted distance map back in; an Invariant
# Point Attention structure module then iteratively updates per-nucleotide rigid-body
# frames (rotation + translation) to place a fixed local nucleotide-base template.
# Classes below (`PreMSA`/`RecyclingEmbedder`/`MSA2xyzIteration`/`MSA2XYZ` from
# EvoMSA2XYZ.py; `Evoformer`/`EvoBlock` from Evoformer.py; `MSARow`/`MSACol`/
# `MSATrans`/`MSAOPM` from EvoMSA.py; `TriOut`/`TriIn`/`TriAttStart`/`TriAttEnd`/
# `PairTrans` from EvoPair.py; `InvariantPointAttention` from IPA.py;
# `TransitionModule`/`BackboneUpdate`/`TorsionNet`/`StructureModule` from Structure.py;
# `Linear`/`LinearNoBias`/geometry helpers from basic.py) are copied unmodified from the
# real repo files (only the per-file relative imports like `import basic,Structure` are
# flattened since everything now lives in one module, and `fourier_encode_dist`'s unused
# `orig_x`/`include_self` branch is kept as-is). A `DRfoldTraceWrapper.forward()` is added
# purely as a tracing adapter: the real entry point is `MSA2XYZ.pred(msa_, ss, base_x,
# n_cycle)` (see DeepE2EPotential/predict.py `pipeline()`), which is not named `forward`
# and returns a dict of per-cycle numpy-bound coordinate tensors -- the wrapper calls the
# unmodified `.pred()` and returns the final cycle's raw coordinate tensor so `tl.trace`
# has a single-tensor-producing `forward`. No architectural code is added or changed.
import math

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- basic.py ---
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.linear(x)
        return x


class LinearNoBias(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LinearNoBias, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


def batch_atom_transform(k, rotation, translation):
    # k:            L N 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum("bja,bad->bjd", k, rotation) + translation[:, None, :]


def IPA_transform(k, rotation, translation):
    # k:            L d1, d2, 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum("bija,bad->bijd", k, rotation) + translation[:, None, None, :]


def IPA_inverse_transform(k, rotation, translation):
    # k:            L d1, d2, 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum(
        "bija,bad->bijd", k - translation[:, None, None, :], rotation.transpose(-1, -2)
    )


def update_transform(t, tr, rotation, translation):
    return torch.einsum("bja,bad->bjd", t, rotation), torch.einsum(
        "ba,bad->bd", tr, rotation
    ) + translation


def quat2rot(q, L):
    scale = ((q**2).sum(dim=-1, keepdim=True) + 1)[:, :, None]
    u = torch.empty([L, 3, 3], device=q.device)
    u[:, 0, 0] = 1 * 1 + q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2]
    u[:, 0, 1] = 2 * (q[:, 0] * q[:, 1] - 1 * q[:, 2])
    u[:, 0, 2] = 2 * (q[:, 0] * q[:, 2] + 1 * q[:, 1])
    u[:, 1, 0] = 2 * (q[:, 0] * q[:, 1] + 1 * q[:, 2])
    u[:, 1, 1] = 1 * 1 - q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2]
    u[:, 1, 2] = 2 * (q[:, 1] * q[:, 2] - 1 * q[:, 0])
    u[:, 2, 0] = 2 * (q[:, 0] * q[:, 2] - 1 * q[:, 1])
    u[:, 2, 1] = 2 * (q[:, 1] * q[:, 2] + 1 * q[:, 0])
    u[:, 2, 2] = 1 * 1 - q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2]
    return u / scale


# --- EvoMSA.py ---
class MSARow(nn.Module):
    def __init__(self, m_dim, z_dim, N_head=8, c=8):
        super(MSARow, self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1 / math.sqrt(c)
        self.norm1 = nn.LayerNorm(m_dim)
        self.qlinear = LinearNoBias(m_dim, N_head * c)
        self.klinear = LinearNoBias(m_dim, N_head * c)
        self.vlinear = LinearNoBias(m_dim, N_head * c)
        self.norm_z = nn.LayerNorm(z_dim)
        self.zlinear = LinearNoBias(z_dim, N_head)
        self.glinear = Linear(m_dim, N_head * c)
        self.olinear = Linear(N_head * c, m_dim)

    def forward(self, m, z):
        # m : N L 32
        N, L, D = m.shape
        m = self.norm1(m)
        q = self.qlinear(m).reshape(N, L, self.N_head, self.c)  # s rq h c
        k = self.klinear(m).reshape(N, L, self.N_head, self.c)  # s rv h c
        v = self.vlinear(m).reshape(N, L, self.N_head, self.c)
        b = self.zlinear(self.norm_z(z))
        g = torch.sigmoid(self.glinear(m)).reshape(N, L, self.N_head, self.c)
        att = torch.einsum("bqhc,bvhc->bqvh", q, k) * (self.sq_c) + b[None, :, :, :]  # rq rv h
        att = F.softmax(att, dim=2)
        o = torch.einsum("bqvh,bvhc->bqhc", att, v) * g
        m_ = self.olinear(o.reshape(N, L, -1))
        return m_


class MSACol(nn.Module):
    def __init__(self, m_dim, N_head=8, c=8):
        super(MSACol, self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1 / math.sqrt(c)
        self.norm1 = nn.LayerNorm(m_dim)
        self.qlinear = LinearNoBias(m_dim, N_head * c)
        self.klinear = LinearNoBias(m_dim, N_head * c)
        self.vlinear = LinearNoBias(m_dim, N_head * c)

        self.glinear = Linear(m_dim, N_head * c)
        self.olinear = Linear(N_head * c, m_dim)

    def forward(self, m):
        # m : N L 32
        N, L, D = m.shape
        m = self.norm1(m)
        q = self.qlinear(m).reshape(N, L, self.N_head, self.c)  # s rq h c
        k = self.klinear(m).reshape(N, L, self.N_head, self.c)  # s rv h c
        v = self.vlinear(m).reshape(N, L, self.N_head, self.c)

        g = torch.sigmoid(self.glinear(m)).reshape(N, L, self.N_head, self.c)

        att = torch.einsum("slhc,tlhc->stlh", q, k) * (self.sq_c)  # rq rv h
        att = F.softmax(att, dim=1)
        o = torch.einsum("stlh,tlhc->slhc", att, v) * g
        m_ = self.olinear(o.reshape(N, L, -1))
        return m_


class MSATrans(nn.Module):
    def __init__(self, m_dim, c_expand=2):
        super(MSATrans, self).__init__()
        self.c_expand = 4
        self.m_dim = m_dim
        self.norm = nn.LayerNorm(m_dim)
        self.linear1 = Linear(m_dim, m_dim * c_expand)
        self.linear2 = Linear(m_dim * c_expand, m_dim)

    def forward(self, m):
        m = self.norm(m)
        m = self.linear1(m)
        m = self.linear2(F.relu(m))
        return m


class MSAOPM(nn.Module):
    def __init__(self, m_dim, z_dim, c=12):
        super(MSAOPM, self).__init__()
        self.m_dim = m_dim
        self.c = c
        self.norm = nn.LayerNorm(m_dim)
        self.linear1 = Linear(m_dim, c)
        self.linear2 = Linear(m_dim, c)
        self.linear3 = Linear(c * c, z_dim)

    def forward(self, m):
        N, L, D = m.shape
        o = self.norm(m)
        a = self.linear2(o)
        b = self.linear1(o)
        o = torch.einsum("nia,njb->nijab", a, b).mean(dim=0)
        o = self.linear3(o.reshape(L, L, -1))
        return o


# --- EvoPair.py ---
class TriOut(nn.Module):
    def __init__(self, z_dim, c=32):
        super(TriOut, self).__init__()
        self.z_dim = z_dim
        self.norm = nn.LayerNorm(z_dim)
        self.onorm = nn.LayerNorm(c)
        self.alinear = Linear(z_dim, c)
        self.blinear = Linear(z_dim, c)
        self.aglinear = Linear(z_dim, c)
        self.bglinear = Linear(z_dim, c)
        self.glinear = Linear(z_dim, z_dim)
        self.olinear = Linear(c, z_dim)

    def forward(self, z):
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        o = torch.einsum("ilc,jlc->ijc", a, b)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o


class TriIn(nn.Module):
    def __init__(self, z_dim, c=32):
        super(TriIn, self).__init__()
        self.z_dim = z_dim
        self.norm = nn.LayerNorm(z_dim)
        self.onorm = nn.LayerNorm(c)
        self.alinear = Linear(z_dim, c)
        self.blinear = Linear(z_dim, c)
        self.aglinear = Linear(z_dim, c)
        self.bglinear = Linear(z_dim, c)
        self.glinear = Linear(z_dim, z_dim)
        self.olinear = Linear(c, z_dim)

    def forward(self, z):
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        o = torch.einsum("lic,ljc->ijc", a, b)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o


class TriAttStart(nn.Module):
    def __init__(self, z_dim, N_head=4, c=8):
        super(TriAttStart, self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1 / math.sqrt(c)
        self.norm = nn.LayerNorm(z_dim)
        self.qlinear = Linear(z_dim, c * N_head)
        self.klinear = Linear(z_dim, c * N_head)
        self.vlinear = Linear(z_dim, c * N_head)
        self.blinear = Linear(z_dim, N_head)
        self.glinear = Linear(z_dim, c * N_head)
        self.olinear = Linear(c * N_head, z_dim)

    def forward(self, z_):
        L1, L2, D = z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1, L2, self.N_head, self.c)
        k = self.klinear(z).reshape(L1, L2, self.N_head, self.c)
        v = self.vlinear(z).reshape(L1, L2, self.N_head, self.c)
        b = self.blinear(z)
        att = torch.einsum("blhc,bkhc->blkh", q, k) * self.sq_c + b[None, :, :, :]
        att = F.softmax(att, dim=2)
        o = torch.einsum("blkh,bkhc->blhc", att, v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1, L2, self.N_head, self.c)) * o).reshape(
            L1, L2, -1
        )
        o = self.olinear(o)
        return o


class TriAttEnd(nn.Module):
    def __init__(self, z_dim, N_head=4, c=8):
        super(TriAttEnd, self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1 / math.sqrt(c)
        self.norm = nn.LayerNorm(z_dim)
        self.qlinear = Linear(z_dim, c * N_head)
        self.klinear = Linear(z_dim, c * N_head)
        self.vlinear = Linear(z_dim, c * N_head)
        self.blinear = Linear(z_dim, N_head)
        self.glinear = Linear(z_dim, c * N_head)
        self.olinear = Linear(c * N_head, z_dim)

    def forward(self, z_):
        L1, L2, D = z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1, L2, self.N_head, self.c)
        k = self.klinear(z).reshape(L1, L2, self.N_head, self.c)
        v = self.vlinear(z).reshape(L1, L2, self.N_head, self.c)
        b = self.blinear(z)
        att = torch.einsum("blhc,kbhc->blkh", q, k) * self.sq_c + b[None, :, :, :].permute(
            0, 2, 1, 3
        )
        att = F.softmax(att, dim=2)
        o = torch.einsum("blkh,klhc->blhc", att, v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1, L2, self.N_head, self.c)) * o).reshape(
            L1, L2, -1
        )
        o = self.olinear(o)
        return o

    def forward2(self, z_):
        z = z_.permute(1, 0, 2)
        L1, L2, D = z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1, L2, self.N_head, self.c)
        k = self.klinear(z).reshape(L1, L2, self.N_head, self.c)
        v = self.vlinear(z).reshape(L1, L2, self.N_head, self.c)
        b = self.blinear(z)
        att = torch.einsum("blhc,bkhc->blkh", q, k) * self.sq_c + b[None, :, :, :]
        att = F.softmax(att, dim=2)
        o = torch.einsum("blkh,bkhc->blhc", att, v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1, L2, self.N_head, self.c)) * o).reshape(
            L1, L2, -1
        )
        o = self.olinear(o)
        o = o.permute(1, 0, 2)
        return o


class PairTrans(nn.Module):
    def __init__(self, z_dim, c_expand=2):
        super(PairTrans, self).__init__()
        self.z_dim = z_dim
        self.c_expand = c_expand
        self.norm = nn.LayerNorm(z_dim)
        self.linear1 = Linear(z_dim, z_dim * c_expand)
        self.linear2 = Linear(z_dim * c_expand, z_dim)

    def forward(self, z):
        a = self.linear1(self.norm(z))
        a = self.linear2(F.relu(a))
        return a


# --- Evoformer.py ---
class EvoBlock(nn.Module):
    def __init__(self, m_dim, z_dim, docheck=False):
        super(EvoBlock, self).__init__()
        self.msa_row = MSARow(m_dim, z_dim)
        self.msa_col = MSACol(m_dim)
        self.msa_trans = MSATrans(m_dim)

        self.msa_opm = MSAOPM(m_dim, z_dim)

        self.pair_triout = TriOut(z_dim)
        self.pair_triin = TriIn(z_dim)
        self.pair_tristart = TriAttStart(z_dim)
        self.pair_triend = TriAttEnd(z_dim)
        self.pair_trans = PairTrans(z_dim)
        self.docheck = docheck

    def layerfunc_msa_row(self, m, z):
        return self.msa_row(m, z) + m

    def layerfunc_msa_col(self, m):
        return self.msa_col(m) + m

    def layerfunc_msa_trans(self, m):
        return self.msa_trans(m) + m

    def layerfunc_msa_opm(self, m, z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self, z):
        return self.pair_triout(z) + z

    def layerfunc_pair_triin(self, z):
        return self.pair_triin(z) + z

    def layerfunc_pair_tristart(self, z):
        return self.pair_tristart(z) + z

    def layerfunc_pair_triend(self, z):
        return self.pair_triend(z) + z

    def layerfunc_pair_trans(self, z):
        return self.pair_trans(z) + z

    def forward(self, m, z):
        m = m + self.msa_row(m, z)
        m = m + self.msa_col(m)
        m = m + self.msa_trans(m)
        z = z + self.msa_opm(m)
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_tristart(z)
        z = z + self.pair_triend(z)
        z = z + self.pair_trans(z)
        return m, z


class Evoformer(nn.Module):
    def __init__(self, m_dim, z_dim, docheck=False, n_layers=48):
        super(Evoformer, self).__init__()
        # n_layers defaults to the real repo's hardcoded self.layers=[48]; exposed as a
        # constructor arg only so a tiny random-init trace can shrink the depth (a size
        # knob, not an architectural change -- each retained layer is the identical
        # EvoBlock).
        self.layers = [n_layers]
        self.docheck = docheck
        if docheck:
            pass
            # print('will do checkpoint')
        self.evos = nn.ModuleList([EvoBlock(m_dim, z_dim, True) for i in range(self.layers[0])])

    def layerfunc(self, layermodule, m, z):
        m_, z_ = layermodule(m, z)
        return m_, z_

    def forward(self, m, z):
        for i in range(self.layers[0]):
            m, z = self.evos[i](m, z)

        return m, z


# --- IPA.py ---
class InvariantPointAttention(nn.Module):
    def __init__(self, dim_in, dim_z, N_head=8, c=16, N_query=4, N_p_values=6) -> None:
        super(InvariantPointAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_z = dim_z
        self.N_head = N_head
        self.c = c
        self.c_squ = 1.0 / math.sqrt(c)
        self.W_c = math.sqrt(2.0 / (9 * N_query))
        self.W_L = math.sqrt(1.0 / 3)
        self.N_query = N_query
        self.N_p_values = N_p_values
        self.liner_nb_q1 = LinearNoBias(dim_in, self.c * N_head)
        self.liner_nb_k1 = LinearNoBias(dim_in, self.c * N_head)
        self.liner_nb_v1 = LinearNoBias(dim_in, self.c * N_head)

        self.liner_nb_q2 = LinearNoBias(dim_in, N_head * N_query * 3)
        self.liner_nb_k2 = LinearNoBias(dim_in, N_head * N_query * 3)

        self.liner_nb_v3 = LinearNoBias(dim_in, N_head * N_p_values * 3)

        self.liner_nb_z = LinearNoBias(dim_z, N_head)
        self.lastlinear1 = Linear(N_head * dim_z, dim_in)
        self.lastlinear2 = Linear(N_head * c, dim_in)
        self.lastlinear3 = Linear(N_head * N_p_values * 3, dim_in)
        self.gama = nn.ParameterList([nn.Parameter(torch.zeros(N_head))])
        self.cos_f = nn.CosineSimilarity(dim=-1)

    def forward(self, s, z, rot, trans):
        L = s.shape[0]
        q1 = self.liner_nb_q1(s).reshape(L, self.N_head, self.c)  # Lq,
        k1 = self.liner_nb_k1(s).reshape(L, self.N_head, self.c)
        v1 = self.liner_nb_v1(s).reshape(L, self.N_head, self.c)  # lv,h,c

        attmap = torch.einsum("ihc,jhc->ijh", q1, k1) * self.c_squ  # Lq,Lk_v,h
        bias_z = self.liner_nb_z(z)  # L L h

        q2 = self.liner_nb_q2(s).reshape(L, self.N_head, self.N_query, 3)
        k2 = self.liner_nb_k2(s).reshape(L, self.N_head, self.N_query, 3)

        v3 = self.liner_nb_v3(s).reshape(L, self.N_head, self.N_p_values, 3)

        q2 = IPA_transform(q2, rot, trans)  # Lq,self.N_head,self.N_query,3
        k2 = IPA_transform(k2, rot, trans)  # Lk,self.N_head,self.N_query,3

        dismap = ((q2[:, None, :, :, :] - k2[None, :, :, :, :]) ** 2).sum(
            [3, 4]
        )  # Lq,Lk, self.N_head,
        attmap = attmap + bias_z - F.softplus(self.gama[0])[None, None, :] * dismap * self.W_c * 0.5
        o1 = (attmap[:, :, :, None] * z[:, :, None, :]).sum(1)  # Lq, N_head, c_z
        o2 = torch.einsum("abc,dab->dbc", v1, attmap)  # Lq, N_head, c
        o3 = IPA_transform(v3, rot, trans)  # Lv, h, p* ,3
        o3 = IPA_inverse_transform(
            torch.einsum("vhpt,gvh->ghpt", o3, attmap), rot, trans
        )  # Lv, h, p* ,3

        return (
            self.lastlinear1(o1.reshape(L, -1))
            + self.lastlinear2(o2.reshape(L, -1))
            + self.lastlinear3(o3.reshape(L, -1))
        )


# --- Structure.py ---
class TransitionModule(nn.Module):
    def __init__(self, c):
        super(TransitionModule, self).__init__()
        self.c = c
        self.norm1 = nn.LayerNorm(c)
        self.linear1 = Linear(c, c)
        self.linear2 = Linear(c, c)
        self.linear3 = Linear(c, c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, s_):
        s = self.norm1(s_)
        s = F.relu(self.linear1(s))
        s = F.relu(self.linear2(s))
        s = s_ + self.linear3(s)
        return self.norm2(s)


class BackboneUpdate(nn.Module):
    def __init__(self, indim):
        super(BackboneUpdate, self).__init__()
        self.indim = indim
        self.linear = Linear(indim, 6)
        torch.nn.init.zeros_(self.linear.linear.weight)
        torch.nn.init.zeros_(self.linear.linear.bias)

    def forward(self, s, L):
        pred = self.linear(s)
        rot = quat2rot(pred[..., :3], L)
        return rot, pred[..., 3:]  # rot, translation


class TorsionNet(nn.Module):
    def __init__(self, s_dim, c):
        super(TorsionNet, self).__init__()
        self.s_dim = s_dim
        self.c = c
        self.linear1 = Linear(s_dim, c)
        self.linear2 = Linear(c, c)

        self.linear3 = Linear(c, c)
        self.linear4 = Linear(c, c)

        self.linear5 = Linear(c, c)
        self.linear6 = Linear(c, c)

        self.linear7_1 = Linear(c, 1)
        self.linear7_2 = Linear(c, 2)
        self.linear7_3 = Linear(c, 2)

    def forward(self, s_init, s):
        a = self.linear1(s_init) + self.linear2(s)
        a = a + self.linear4(F.relu(self.linear3(F.relu(a))))
        a = a + self.linear6(F.relu(self.linear5(F.relu(a))))
        bondlength = self.linear7_1(F.relu(a))
        angle = self.linear7_2(F.relu(a))
        torsion = self.linear7_3(F.relu(a))

        angle_L = torch.norm(angle, dim=-1, keepdim=True)
        angle = angle / (angle_L + 1e-8)

        torsion_L = torch.norm(torsion, dim=-1, keepdim=True)
        torsion = torsion / (torsion_L + 1e-8)

        return bondlength, angle, angle_L, torsion, torsion_L


class StructureModule(nn.Module):
    def __init__(self, s_dim, z_dim, N_layer, c):
        super(StructureModule, self).__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.N_layer = N_layer
        self.N_head = 8
        self.c = c
        self.use_rmsdloss = False
        self.layernorm_s = nn.LayerNorm(s_dim)
        self.layernorm_z = nn.LayerNorm(z_dim)
        # self.baseframe=self._base_frame()
        # shared weights part
        self.ipa = InvariantPointAttention(c, z_dim, c)
        self.transition = TransitionModule(c)
        self.bbupdate = BackboneUpdate(c)
        self.torsionnet = TorsionNet(s_dim, c)
        self._init_T()

    def _init_T(self):
        self.trans = torch.zeros(3)[None, :]
        self.rot = torch.eye(3)[None, :, :]

    def pred(self, s_init, z, base_x):
        if self.trans.device != s_init.device:
            self.trans = self.trans.to(s_init.device)
        if self.rot.device != s_init.device:
            self.rot = self.rot.to(s_init.device)
        L = s_init.shape[0]
        rot, trans = self.rot.repeat(L, 1, 1), self.trans.repeat(L, 1)
        s = self.layernorm_s(s_init)
        z = self.layernorm_z(z)
        for layer in range(self.N_layer):
            s = s + self.ipa(s, z, rot, trans)
            s = self.transition(s)
            rot_tmp, trans_tmp = self.bbupdate(s, L)
            rot, trans = update_transform(rot_tmp, trans_tmp, rot, trans)

        s = s + self.ipa(s, z, rot, trans)
        s = self.transition(s)
        rot_tmp, trans_tmp = self.bbupdate(s, L)
        rot, trans = update_transform(rot_tmp, trans_tmp, rot, trans)

        predx = base_x + 0.0
        predx = batch_atom_transform(predx, rot, trans)
        return predx, rot, trans


# --- EvoMSA2XYZ.py ---
class PreMSA(nn.Module):
    def __init__(self, seq_dim, msa_dim, m_dim, z_dim):
        super(PreMSA, self).__init__()
        self.msalinear = Linear(msa_dim, m_dim)
        self.qlinear = Linear(seq_dim, z_dim)
        self.klinear = Linear(seq_dim, z_dim)
        self.slinear = Linear(seq_dim, m_dim)
        self.pos = self.compute_pos().float()
        self.pos1d = self.compute_apos()
        self.poslinear = Linear(65, z_dim)
        self.poslinear2 = Linear(14, m_dim)

    def tocuda(self, device):
        self.to(device)
        self.pos.to(device)

    def compute_apos(self, maxL=2000):
        d = torch.arange(maxL)
        m = 14
        d = ((d[:, None] & (1 << torch.arange(m))) > 0).float()
        return d

    def compute_pos(self, maxL=2000):
        a = torch.arange(maxL)
        b = (a[None, :] - a[:, None]).clamp(-32, 32)
        return F.one_hot(b + 32, 65)

    def forward(self, seq, msa):
        if self.pos.device != msa.device:
            self.pos = self.pos.to(msa.device)
        if self.pos1d.device != msa.device:
            self.pos1d = self.pos1d.to(msa.device)
        # msa N L D, seq L D
        N, L, D = msa.shape
        s = self.slinear(seq)
        m = self.msalinear(msa)
        p = self.poslinear2(self.pos1d[:L])

        m = m + s[None, :, :] + p[None, :, :]

        sq = self.qlinear(seq)
        sk = self.klinear(seq)
        z = sq[None, :, :] + sk[:, None, :]

        z = z + self.poslinear(self.pos[:L, :L])
        return m, z


def fourier_encode_dist(x, num_encodings=20, include_self=True):
    # from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class RecyclingEmbedder(nn.Module):
    def __init__(self, m_dim, z_dim, dis_encoding_dim):
        super(RecyclingEmbedder, self).__init__()
        self.linear = Linear(dis_encoding_dim * 2 + 1, z_dim)
        self.dis_encoding_dim = dis_encoding_dim
        self.normz = nn.LayerNorm(z_dim)
        self.normm = nn.LayerNorm(m_dim)

    def forward(self, m, z, x, first):
        cb = x[:, -1]
        dismap = (cb[:, None, :] - cb[None, :, :]).norm(dim=-1)
        dis_z = fourier_encode_dist(dismap, self.dis_encoding_dim)
        if first:
            return 0, self.linear(dis_z)
        else:
            z = self.normz(z) + self.linear(dis_z)
            m = self.normm(m)
            return m, z


class MSA2xyzIteration(nn.Module):
    def __init__(
        self,
        seq_dim,
        msa_dim,
        N_ensemble,
        m_dim=64,
        s_dim=128,
        z_dim=64,
        docheck=True,
        n_evo_layers=48,
    ):
        super(MSA2xyzIteration, self).__init__()
        self.msa_dim = msa_dim
        self.m_dim = m_dim
        self.z_dim = z_dim
        self.seq_dim = seq_dim
        self.N_ensemble = N_ensemble
        self.dis_dim = 36 + 2
        self.pre_z = nn.Linear(4, z_dim)
        self.premsa = PreMSA(seq_dim, msa_dim, m_dim, z_dim)
        self.re_emb = RecyclingEmbedder(m_dim, z_dim, dis_encoding_dim=64)
        self.evmodel = Evoformer(m_dim, z_dim, True, n_layers=n_evo_layers)
        self.slinear = Linear(z_dim, s_dim)

    def pred(self, msa_, ss_, m1_pre, z_pre, pre_x, cycle_index):
        m1_all, z_all, s_all = 0, 0, 0
        N, L, _ = msa_.shape
        for i in range(self.N_ensemble):
            msa_mask = torch.zeros(N, L).to(msa_.device)
            msa_true = msa_ + 0
            seq = msa_true[0] * 1.0  # 22-dim
            msa = torch.cat([msa_true * (1 - msa_mask[:, :, None]), msa_mask[:, :, None]], dim=-1)
            m, z = self.premsa(seq, msa)
            ss = self.pre_z(ss_)
            z = z + ss
            m1_, z_ = self.re_emb(
                m1_pre, z_pre, pre_x, cycle_index == 0
            )  # already added residually
            z = z + z_
            m = torch.cat([(m[0] + m1_)[None, ...], m[1:]], dim=0)
            m, z = self.evmodel(m, z)
            s = self.slinear(m[0])
            m1_all = m1_all + m[0]
            z_all = z_all + z
            s_all = s_all + s
        return m1_all / self.N_ensemble, z_all / self.N_ensemble, s_all / self.N_ensemble


class MSA2XYZ(nn.Module):
    def __init__(
        self,
        seq_dim,
        msa_dim,
        N_ensemble,
        N_cycle,
        m_dim=64,
        s_dim=128,
        z_dim=64,
        docheck=True,
        n_evo_layers=48,
        n_struct_layers=4,
    ):
        super(MSA2XYZ, self).__init__()
        self.msa_dim = msa_dim
        self.m_dim = m_dim
        self.z_dim = z_dim
        self.dis_dim = 36 + 2
        self.N_cycle = N_cycle
        self.msaxyzone = MSA2xyzIteration(
            seq_dim,
            msa_dim,
            N_ensemble,
            m_dim=m_dim,
            s_dim=s_dim,
            z_dim=z_dim,
            n_evo_layers=n_evo_layers,
        )
        self.msa_predor = Linear(m_dim, msa_dim - 1)
        self.dis_predor = Linear(z_dim, self.dis_dim)
        self.slinear = Linear(m_dim, s_dim)

        self.structurenet = StructureModule(
            s_dim, z_dim, n_struct_layers, s_dim
        )  # s_dim,z_dim,N_layer,c

    def pred(self, msa_, ss, base_x, n_cycle):
        predxs = {}
        L = msa_.shape[1]
        m1_pre, z_pre = 0, 0
        x_pre = torch.zeros(L, 3, 3).to(msa_.device)
        for i in range(n_cycle):
            m1, z, s = self.msaxyzone.pred(msa_, ss, m1_pre, z_pre, x_pre, i)
            x = self.structurenet.pred(s, z, base_x)[0]
            m1_pre = m1.detach()
            z_pre = z.detach()
            x_pre = x.detach()
            _pred_dis = F.softmax(self.dis_predor(z), dim=-1)
            predxs[i] = x_pre
            predxs[str(i) + "_score"] = 0

        return predxs


class DRfoldTraceWrapper(nn.Module):
    """Tracing adapter around the real `MSA2XYZ.pred()` entry point (see
    DeepE2EPotential/predict.py `pipeline()`). Not part of the original architecture --
    forwards straight through to `.pred()` and returns the final recycling cycle's
    predicted coordinates as a single tensor."""

    def __init__(
        self,
        seq_dim,
        msa_dim,
        N_ensemble,
        N_cycle,
        m_dim=64,
        s_dim=128,
        z_dim=64,
        n_evo_layers=48,
        n_struct_layers=4,
    ):
        super().__init__()
        self.n_cycle = N_cycle
        self.model = MSA2XYZ(
            seq_dim,
            msa_dim,
            N_ensemble,
            N_cycle,
            m_dim=m_dim,
            s_dim=s_dim,
            z_dim=z_dim,
            n_evo_layers=n_evo_layers,
            n_struct_layers=n_struct_layers,
        )

    def forward(self, msa, ss, base_x):
        predxs = self.model.pred(msa, ss, base_x, self.n_cycle)
        return predxs[self.n_cycle - 1]


def build_drfold():
    # Real caller (DeepE2EPotential/predict.py load_model) constructs
    # MSA2XYZ(msa_dim-1=6, msa_dim=7, N_ensemble=3, N_cycle=4, m_dim=64, s_dim=64, z_dim=64)
    # with the Evoformer trunk's real depth of 48 EvoBlocks and a 4-layer IPA structure
    # module. Sizes shrunk for a tiny random-init trace: m/s/z dims 64->8, N_ensemble
    # 3->1, N_cycle 4->1, Evoformer depth 48->2, structure-module depth 4->1 -- all pure
    # size/repeat-count knobs (identical block classes retained at every remaining
    # repetition), matching the real constructor's own signature (n_evo_layers/
    # n_struct_layers are the module-level `self.layers=[48]` / `N_layer` args exposed as
    # constructor parameters here purely so a tiny trace is fast).
    return DRfoldTraceWrapper(
        seq_dim=6,
        msa_dim=7,
        N_ensemble=1,
        N_cycle=1,
        m_dim=8,
        s_dim=8,
        z_dim=8,
        n_evo_layers=2,
        n_struct_layers=1,
    )


def example_input_drfold():
    torch.manual_seed(0)
    length = 6
    # msa: [N_seqs, L, 6] one-hot nucleotide encoding (real caller: parse_seq() maps
    # A/G/C/U/T to indices 1-4, one_hot(..., 6) with index 0 reserved for gap/other;
    # predict.py stacks the query sequence twice as msa=cat([msa,msa],0)).
    msa_idx = torch.randint(0, 6, (2, length))
    msa = F.one_hot(msa_idx, 6).float()
    # ss: [L, L, 4] secondary-structure pair prior (real caller loads this from a
    # precomputed .npy; consumed by `self.pre_z=nn.Linear(4,z_dim)` in
    # MSA2xyzIteration.pred).
    ss = torch.rand(length, length, 4)
    # base_x: [L, 3, 3] per-nucleotide local base-atom template coordinates (real
    # caller: Get_base(seq, basenpy_standard), one fixed 3x3 template row per base type).
    base_x = torch.randn(length, 3, 3)
    return (msa, ss, base_x)


MENAGERIE_ENTRIES = [
    ("DRfold", "build_drfold", "example_input_drfold", 2023, MENAGERIE_ZOO),
]
