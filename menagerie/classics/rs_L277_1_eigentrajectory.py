# SOURCE: vendored from InhwanBae/EigenTrajectory @ main
# Files combined:
#   EigenTrajectory/model.py (EigenTrajectory -- the ET meta-model)
#   EigenTrajectory/descriptor.py (ETDescriptor -- SVD-based low-rank trajectory descriptor)
#   EigenTrajectory/anchor.py (ETAnchor -- anchor-based refinement in the ET space)
#   EigenTrajectory/normalizer.py (TrajNorm -- origin/rotation/scale trajectory normalization)
#   baseline/stgcnn/model.py (social_stgcnn, ConvTemporalGraphical, st_gcn -- the repo's
#       Social-STGCNN baseline predictor, itself vendored upstream from
#       abduallahmohamed/Social-STGCNN)
#   baseline/stgcnn/bridge.py (generate_adjacency_matrix, model_forward_pre_hook,
#       model_forward, model_forward_post_hook -- the ET<->baseline "hook_func" glue used by
#       EigenTrajectory.forward())
#
# EigenTrajectory (Bae et al., "EigenTrajectory: Low-Rank Descriptors for Multi-Modal
# Trajectory Forecasting", ICCV 2023) is a plug-in low-rank trajectory-descriptor front/back
# end: it projects observed/predicted pedestrian trajectories into a truncated-SVD ET space,
# refines coefficients with learned anchors, and hands off the actual sequence-to-sequence
# prediction to an arbitrary "baseline" trajectory predictor via hook functions. This vendors
# the real ET descriptor/anchor/normalizer code plus the repo's own Social-STGCNN baseline
# (the simplest of the repo's ~9 supported baselines) and its bridge hooks, unmodified.
#
# Import-only fixes applied (no architectural change):
#   - EigenTrajectory.calculate_parameters()/ETAnchor.anchor_generation() (both training-only,
#     called once before training to SVD-fit the descriptors and k-means-fit the anchors) are
#     not invoked here; ETDescriptor.U_obs_trunc/U_pred_trunc and ETAnchor.C_anchor keep their
#     nn.Parameter(torch.zeros(...)) initialization exactly as in the original __init__, so no
#     sklearn dependency is needed to build/trace the model.
#   - hyper_params is a plain dict-like DotDict (repo's utils.utils.DotDict) built inline
#     instead of loaded from a JSON config file; values mirror the repo's own
#     config/eigentrajectory-{baseline}-eth.json template exactly (k=6, num_samples=20,
#     static_dist=0.419, obs_len=8, pred_len=12, traj_dim=2).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn


class DotDict(dict):
    r"""dot.notation access to dictionary attributes (repo's utils.utils.DotDict)"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getstate__ = dict
    __setstate__ = dict.update


# ---------------------------------------------------------------------------
# EigenTrajectory/normalizer.py
# ---------------------------------------------------------------------------
class TrajNorm:
    r"""Normalize trajectory with shape (num_peds, length_of_time, 2)"""

    def __init__(self, ori=True, rot=True, sca=True):
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori, self.traj_rot, self.traj_sca = None, None, None

    def calculate_params(self, traj):
        if self.ori:
            self.traj_ori = traj[:, [-1]]
        if self.rot:
            dir = traj[:, -1] - traj[:, -3]
            rot = torch.atan2(dir[:, 1], dir[:, 0])
            self.traj_rot = torch.stack(
                [
                    torch.stack([rot.cos(), -rot.sin()], dim=1),
                    torch.stack([rot.sin(), rot.cos()], dim=1),
                ],
                dim=1,
            )
        if self.sca:
            self.traj_sca = 1.0 / (traj[:, -1] - traj[:, -3]).norm(p=2, dim=-1)[:, None, None] * 2

    def get_params(self):
        return self.ori, self.rot, self.sca, self.traj_ori, self.traj_rot, self.traj_sca

    def set_params(self, ori, rot, sca, traj_ori, traj_rot, traj_sca):
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori, self.traj_rot, self.traj_sca = traj_ori, traj_rot, traj_sca

    def normalize(self, traj):
        if self.ori:
            traj = traj - self.traj_ori
        if self.rot:
            traj = traj @ self.traj_rot
        if self.sca:
            traj = traj * self.traj_sca
        return traj

    def denormalize(self, traj):
        if self.sca:
            traj = traj / self.traj_sca
        if self.rot:
            traj = traj @ self.traj_rot.transpose(-1, -2)
        if self.ori:
            traj = traj + self.traj_ori
        return traj


# ---------------------------------------------------------------------------
# EigenTrajectory/descriptor.py
# ---------------------------------------------------------------------------
class ETDescriptor(nn.Module):
    r"""EigenTrajectory descriptor model"""

    def __init__(self, hyper_params, norm_ori=True, norm_rot=True, norm_sca=True):
        super().__init__()

        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.traj_normalizer = TrajNorm(ori=norm_ori, rot=norm_rot, sca=norm_sca)

        self.U_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.k)))
        self.U_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))

    def normalize_trajectory(self, obs_traj, pred_traj=None):
        self.traj_normalizer.calculate_params(obs_traj)
        obs_traj_norm = self.traj_normalizer.normalize(obs_traj)
        pred_traj_norm = (
            self.traj_normalizer.normalize(pred_traj) if pred_traj is not None else None
        )
        return obs_traj_norm, pred_traj_norm

    def denormalize_trajectory(self, traj_norm):
        traj = self.traj_normalizer.denormalize(traj_norm)
        return traj

    def to_ET_space(self, traj, evec):
        tdim = evec.size(0)
        M = traj.reshape(-1, tdim).T
        C = evec.T.detach() @ M
        return C

    def to_Euclidean_space(self, C, evec):
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.T.reshape(-1, t, self.dim)
        return traj

    def truncated_SVD(self, traj, k=None, full_matrices=False):
        assert traj.size(2) == self.dim  # NTC
        k = self.k if k is None else k

        M = traj.reshape(-1, traj.size(1) * self.dim).T
        U, S, Vt = torch.linalg.svd(M, full_matrices=full_matrices)

        U_trunc, S_trunc, Vt_trunc = U[:, :k], S[:k], Vt[:k, :]
        return U_trunc, S_trunc, Vt_trunc.T

    def parameter_initialization(self, obs_traj, pred_traj):
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)

        U_obs_trunc, _, _ = self.truncated_SVD(obs_traj_norm)
        U_pred_trunc, _, _ = self.truncated_SVD(pred_traj_norm)

        self.U_obs_trunc = nn.Parameter(U_obs_trunc.to(self.U_obs_trunc.device))
        self.U_pred_trunc = nn.Parameter(U_pred_trunc.to(self.U_pred_trunc.device))

        return pred_traj_norm, U_pred_trunc

    def projection(self, obs_traj, pred_traj=None):
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)
        C_obs = self.to_ET_space(obs_traj_norm, evec=self.U_obs_trunc).detach()
        C_pred = (
            self.to_ET_space(pred_traj_norm, evec=self.U_pred_trunc).detach()
            if pred_traj is not None
            else None
        )
        return C_obs, C_pred

    def reconstruction(self, C_pred):
        pred_traj_norm = [
            self.to_Euclidean_space(C_pred[:, :, s], evec=self.U_pred_trunc) for s in range(self.s)
        ]
        pred_traj = [self.denormalize_trajectory(pred_traj_norm[s]) for s in range(self.s)]
        pred_traj = torch.stack(pred_traj, dim=0)  # SNTC
        return pred_traj

    def forward(self, C_pred):
        return self.reconstruction(C_pred)


# ---------------------------------------------------------------------------
# EigenTrajectory/anchor.py
# ---------------------------------------------------------------------------
class ETAnchor(nn.Module):
    r"""EigenTrajectory anchor model"""

    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim

        self.C_anchor = nn.Parameter(torch.zeros((self.k, self.s)))

    def to_ET_space(self, traj, evec):
        tdim = evec.size(0)
        M = traj.reshape(-1, tdim).T
        C = evec.T.detach() @ M
        return C

    def to_Euclidean_space(self, C, evec):
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.T.reshape(-1, t, self.dim)
        return traj

    def anchor_generation(self, pred_traj_norm, U_pred_trunc):
        from sklearn.cluster import KMeans

        C_pred = self.to_ET_space(pred_traj_norm, evec=U_pred_trunc).T.detach().numpy()

        C_anchor = torch.FloatTensor(
            KMeans(n_clusters=self.s, random_state=0, init="k-means++", n_init=10)
            .fit(C_pred)
            .cluster_centers_.T
        )

        self.C_anchor = nn.Parameter(C_anchor.to(self.C_anchor.device))

    def forward(self, C_pred):
        C_pred_refine = self.C_anchor.unsqueeze(dim=1).detach() + C_pred
        return C_pred_refine


# ---------------------------------------------------------------------------
# baseline/stgcnn/model.py
# Source-code referred from Social-STGCNN at
# https://github.com/abduallahmohamed/Social-STGCNN/tree/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/model.py
# ---------------------------------------------------------------------------
class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_mdn=False,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class social_stgcnn(nn.Module):
    def __init__(
        self,
        n_stgcnn=1,
        n_txpcnn=1,
        input_feat=2,
        output_feat=5,
        seq_len=8,
        pred_seq_len=12,
        kernel_size=3,
    ):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a):
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v


# ---------------------------------------------------------------------------
# baseline/stgcnn/bridge.py
# ---------------------------------------------------------------------------
def generate_adjacency_matrix(v, mask=None):
    # return adjacency matrix for Social-STGCNN baseline
    n_ped = v.size(-1)
    temp = v.permute(0, 2, 3, 1).unsqueeze(dim=-2).repeat_interleave(repeats=n_ped, dim=-2)
    a = (temp - temp.transpose(2, 3)).norm(p=2, dim=-1)
    # inverse kernel
    a_inv = 1.0 / a
    a_inv[a == 0] = 0
    # masking
    a_inv = a_inv if mask is None else a_inv * mask
    # normalize
    a_hat = a_inv + torch.eye(n=n_ped, device=v.device)
    node_degrees = a_hat.sum(dim=-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(n=n_ped, device=v.device) * degs_inv_sqrt
    return torch.eye(n=n_ped, device=v.device) - norm_degs_matrix @ a_hat @ norm_degs_matrix


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)
    v = obs_data[None, :, :, None].detach()
    v = v.permute(0, 3, 1, 2)
    a = generate_adjacency_matrix(v).squeeze(dim=0).detach()
    input_data = (v, a)
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data.permute(0, 2, 3, 1).squeeze(dim=0)
    return pred_data


# ---------------------------------------------------------------------------
# EigenTrajectory/model.py
# ---------------------------------------------------------------------------
class EigenTrajectory(nn.Module):
    r"""The EigenTrajectory model"""

    def __init__(self, baseline_model, hook_func, hyper_params):
        super().__init__()

        self.baseline_model = baseline_model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.static_dist = hyper_params.static_dist

        self.ET_m_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=True)
        self.ET_s_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=False)
        self.ET_m_anchor = ETAnchor(hyper_params=hyper_params)
        self.ET_s_anchor = ETAnchor(hyper_params=hyper_params)

    def calculate_parameters(self, obs_traj, pred_traj):
        # Mask out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
        obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

        # Descriptor initialization
        data_m = self.ET_m_descriptor.parameter_initialization(obs_m_traj, pred_m_traj)
        data_s = self.ET_s_descriptor.parameter_initialization(obs_s_traj, pred_s_traj)

        # Anchor generation
        self.ET_m_anchor.anchor_generation(*data_m)
        self.ET_s_anchor.anchor_generation(*data_s)

    def forward(self, obs_traj, pred_traj=None, addl_info=None):
        n_ped = obs_traj.size(0)

        # Filter out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj = obs_traj[mask]
        obs_s_traj = obs_traj[~mask]
        pred_m_traj_gt = pred_traj[mask] if pred_traj is not None else None
        pred_s_traj_gt = pred_traj[~mask] if pred_traj is not None else None

        # Projection
        C_m_obs, C_m_pred_gt = self.ET_m_descriptor.projection(obs_m_traj, pred_m_traj_gt)
        C_s_obs, C_s_pred_gt = self.ET_s_descriptor.projection(obs_s_traj, pred_s_traj_gt)
        C_obs = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
        C_obs[:, mask], C_obs[:, ~mask] = C_m_obs, C_s_obs  # KN

        # Absolute coordinate
        obs_m_ori = self.ET_m_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_s_ori = self.ET_s_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_ori = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj.device)
        obs_ori[:, mask], obs_ori[:, ~mask] = obs_m_ori, obs_s_ori
        obs_ori -= obs_ori.mean(dim=1, keepdim=True)  # move scene to origin

        # Trajectory prediction
        input_data = self.hook_func.model_forward_pre_hook(C_obs, obs_ori, addl_info)
        output_data = self.hook_func.model_forward(input_data, self.baseline_model)
        C_pred_refine = self.hook_func.model_forward_post_hook(output_data, addl_info)

        # Anchor refinement
        C_m_pred = self.ET_m_anchor(C_pred_refine[:, mask])
        C_s_pred = self.ET_s_anchor(C_pred_refine[:, ~mask])

        # Reconstruction
        pred_m_traj_recon = self.ET_m_descriptor.reconstruction(C_m_pred)
        pred_s_traj_recon = self.ET_s_descriptor.reconstruction(C_s_pred)
        pred_traj_recon = torch.zeros(
            (self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj.device
        )
        pred_traj_recon[:, mask], pred_traj_recon[:, ~mask] = pred_m_traj_recon, pred_s_traj_recon

        output = {"recon_traj": pred_traj_recon}

        if pred_traj is not None:
            C_pred = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj.device)
            C_pred[:, mask], C_pred[:, ~mask] = C_m_pred, C_s_pred

            # Low-rank approximation for gt trajectory
            C_pred_gt = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
            C_pred_gt[:, mask], C_pred_gt[:, ~mask] = C_m_pred_gt, C_s_pred_gt
            C_pred_gt = C_pred_gt.detach()

            # Loss calculation
            error_coefficient = (C_pred - C_pred_gt.unsqueeze(dim=-1)).norm(p=2, dim=0)
            error_displacement = (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement[:, :, -1].min(dim=0)[0].mean()

        return output


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


def _hyper_params():
    # Mirrors config/eigentrajectory-{baseline}-eth.json (the repo's own template config)
    return DotDict(
        {
            "traj_dim": 2,
            "obs_len": 8,
            "pred_len": 12,
            "k": 6,
            "static_dist": 0.419,
            "num_samples": 20,
            "obs_svd": True,
            "pred_svd": True,
            "baseline": "stgcnn",
        }
    )


def build_eigentrajectory():
    # Baseline construction mirrors utils/trainer.py ETSTGCNNTrainer.__init__ exactly
    # (n_stgcnn/n_txpcnn/input_feat/output_feat/kernel_size/seq_len/pred_seq_len), which
    # sizes the Social-STGCNN baseline to operate in the ET (rank-k) coefficient space, not
    # the raw Euclidean trajectory space: seq_len=k+2 (the k ET coefficients concatenated
    # with the 2D scene-origin channel by the bridge hooks), pred_seq_len=k.
    hyper_params = _hyper_params()
    baseline_model = social_stgcnn(
        n_stgcnn=1,
        n_txpcnn=5,
        input_feat=1,
        output_feat=hyper_params.num_samples,
        kernel_size=3,
        seq_len=hyper_params.k + 2,
        pred_seq_len=hyper_params.k,
    )
    hook_func = DotDict(
        {
            "model_forward_pre_hook": model_forward_pre_hook,
            "model_forward": model_forward,
            "model_forward_post_hook": model_forward_post_hook,
        }
    )
    return EigenTrajectory(
        baseline_model=baseline_model, hook_func=hook_func, hyper_params=hyper_params
    )


def example_input_eigentrajectory():
    n_ped = 6
    obs_traj = torch.randn(n_ped, 8, 2)
    pred_traj = torch.randn(n_ped, 12, 2)
    return (obs_traj, pred_traj)


MENAGERIE_ENTRIES = [
    ("EigenTrajectory", build_eigentrajectory, example_input_eigentrajectory, 2023, MENAGERIE_ZOO),
]
