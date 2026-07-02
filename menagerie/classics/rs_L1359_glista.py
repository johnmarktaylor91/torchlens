# FAITHFUL PORT of https://github.com/wukailun/GLISTA @ adf57fcc (original framework: TensorFlow 1.x)
#
# Ported file: GLISTA_cp.py -> `GLISTA_cp` (K. Wu, Y. Guo, Z. Li, C. Zhang, "Sparse Coding with
# Gated Learned ISTA", ICLR 2020). GLISTA unrolls the Iterative Shrinkage-Thresholding Algorithm
# (ISTA) for sparse coding (y = Ax, recover sparse x) into a fixed number of learned layers,
# adding a per-coordinate multiplicative "gain gate" (reweighting shrinkage thresholds by the
# current estimate magnitude, via a learned inverse/relu/exp/sigmoid gain function) and an
# "overshoot gate" (a learned momentum-like update-mixing coefficient) on top of the LISTA-CPSS
# support-selection ("cp"/"ss", couple-parameter + support-selection) shrinkage baseline.
#
# The repo is TensorFlow 1.x static-graph code (`tf.get_variable`/`tf.variable_scope`, and the
# checked-in `GLISTA_cp.py` even contains Python-2 `print` *statement* syntax), and its own
# `shrink_ss_return_index` import (from `utils.tf`, meant to be dropped into the sibling
# `xchen-tamu/linear-lista-cpss` repo tree) is not itself defined anywhere in either repo -- only
# the closely related `shrink_ss` (same repo family, `utils/tf.py`) is. `shrink_ss_return_index`'s
# semantics are unambiguous from `shrink_ss`'s own body: it is the same top-q%-support-preserving
# soft-threshold, additionally returning the complementary support index `cindex_` (used
# downstream as `theta_p = theta_ * cindex` -- the per-coordinate "was NOT selected into the
# top-q% support last iteration" mask that decays theta for support coordinates in later layers).
# This TF1 code cannot run in a modern base env (`tf.get_variable`/graph-mode session API were
# retired/deprecated across TF2 and `linear-lista-cpss`'s own `shrink_ss` additionally needs
# `tf.contrib.distributions.percentile`, removed in TF2). Per the rung ladder this is a RUNG-3
# FAITHFUL PORT: every op of `GLISTA_cp.__init__`/`setup_layers`/`inference` is transcribed
# faithfully into base-env torch (per-timestep `tf.get_variable` params -> per-timestep
# `nn.ParameterList` entries; `tf.matmul`(2D) -> `torch.matmul`/`F.linear`; the 5 gain functions
# `reweight_function/inverse/exp/sigmoid/inverse_variant` and 2 overshoot modes `inv`/`sigm`/`none`
# reproduced verbatim). Batch convention: TF used `y_: [N, batch]` (feature-major); this port keeps
# the standard torch `[batch, N]` convention and transposes the couple of `tf.matmul`s accordingly
# (call-site verified: identical elementwise/threshold semantics, only the batch axis is relocated).

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _shrink_ss_return_index(inputs_, theta_, percent):
    """Faithful port of the (unambiguous) linear-lista-cpss `shrink_ss` semantics, extended to
    also return the complementary support index -- exactly what `shrink_ss_return_index` (used
    but not itself defined in the original GLISTA repo tree) must compute per its call site
    (`theta_p = theta_ * cindex`, i.e. cindex zeroes theta_ for entries selected into the
    top-`percent`% support).

    Special shrink that does not apply soft-shrinkage to entries in the top `percent`% of
    magnitude (per-column / per-batch-sample percentile, matching TF's axis=0 percentile over the
    N-feature axis with the [N, batch] TF layout -> here computed per-batch-row over the N=-1
    axis to match this port's [batch, N] layout).
    """
    abs_ = torch.abs(inputs_)
    # percentile(abs_, 100 - percent) over the feature axis, per batch row, TF keep_dims=True.
    k = max(1, int(round((100.0 - float(percent)) / 100.0 * (abs_.shape[-1] - 1))) + 1)
    k = min(k, abs_.shape[-1])
    thres_ = torch.kthvalue(abs_, k, dim=-1, keepdim=True).values

    index_ = (abs_ > theta_) & (abs_ > thres_)
    index_ = index_.float().detach()  # tf.stop_gradient
    cindex_ = 1.0 - index_

    shrunk = torch.sign(inputs_ * cindex_) * torch.clamp(
        torch.abs(inputs_ * cindex_) - theta_, min=0.0
    )
    out = index_ * inputs_ + shrunk
    return out, cindex_


def _reweight_function(x, D, theta, alti_):
    return 1.0 + alti_ * theta * torch.relu(1 - torch.relu(D * torch.abs(x)))


def _reweight_inverse(x, D, theta, alti):
    return 1.0 + alti * theta * 0.2 / (0.001 + torch.abs(D * x))


def _reweight_exp(x, D, theta, alti):
    return 1.0 + alti * theta * torch.exp(-D * torch.abs(x))


def _reweight_sigmoid(x, D, theta, alti):
    return 1.0 + alti * theta * torch.sigmoid(-D * torch.abs(x))


def _reweight_inverse_variant(x, D, theta, alti, epsilon):
    return 1.0 + alti * theta * 0.2 / (epsilon + torch.abs(D * x))


def _gain(x, D, theta, alti_, epsilon, gain_fun):
    if gain_fun == "relu":
        return _reweight_function(x, D, theta, alti_) + 0.0 * epsilon
    if gain_fun == "inv":
        return _reweight_inverse(x, D, theta, alti_) + 0.0 * epsilon
    if gain_fun == "exp":
        return _reweight_exp(x, D, theta, alti_) + 0.0 * epsilon
    if gain_fun == "sigm":
        return _reweight_sigmoid(x, D, theta, alti_) + 0.0 * epsilon
    if gain_fun == "inv_v":
        return _reweight_inverse_variant(x, D, theta, alti_, epsilon)
    if gain_fun == "none":
        return 1.0 + 0.0 * _reweight_function(x, D, theta, alti_) + 0.0 * epsilon
    raise ValueError(f"unknown gain_fun {gain_fun!r}")


def _overshoot(overshoot_enabled, alti, part_1, part_2):
    if overshoot_enabled:
        return 1.0 - alti * part_1 * part_2
    return 1.0 + 0.0 * alti * part_1 * part_2


class GLISTA_cp(nn.Module):
    """Faithful port of GLISTA_cp.GLISTA_cp (see header). Constructor mirrors the original
    positional signature; `inference(y)` mirrors the original `inference(y_, x0_=None)` but with
    the torch-native [batch, N]/[batch, M] batch-first convention (TF used [N, batch]/[M, batch])."""

    def __init__(
        self,
        A,
        T,
        B,
        lam,
        uf,
        percent,
        max_percent,
        untied,
        coord,
        alti,
        overshoot,
        gain,
        gain_fun,
        over_fun,
        both_gate,
        T_combine,
        T_middle,
    ):
        super().__init__()
        A = np.asarray(A, dtype=np.float32)
        self._A_np = A
        self._T = T
        self._overshoot = overshoot
        self._gain = gain
        self._p = percent
        self._maxp = max_percent
        self._M, self._N = A.shape
        self._scale = 1.001 * np.linalg.norm(A, ord=2) ** 2
        theta0 = (
            (lam / self._scale).astype(np.float32)
            if hasattr(lam, "astype")
            else np.float32(lam / self._scale)
        )
        self._alti_init = alti
        self._logep = -2.0
        theta_vec = np.ones((self._N, 1), dtype=np.float32) * theta0

        self.gain_gate = []
        self.over_gate = []
        if both_gate:
            for i in range(T):
                self.over_gate.append(over_fun)
                if gain_fun == "combine":
                    self.gain_gate.append("inv" if i > T_combine else "relu")
                else:
                    self.gain_gate.append(gain_fun)
        else:
            for i in range(T):
                if i > T_middle:
                    self.gain_gate.append(gain_fun)
                    self.over_gate.append("none")
                else:
                    self.gain_gate.append("none")
                    self.over_gate.append(over_fun)

        ps = [(t + 1) * percent for t in range(T)]
        self._ps = np.clip(ps, 0.0, max_percent)
        self._untied = untied

        # --- setup_layers(): trainable parameters, one set per unrolled timestep ---
        # NOTE (faithful-port gotcha): the original computes a local `W = eye(N) - B@A`
        # ([N,N]) but then initializes the actual trainable `Ws_` variable with `B`
        # ([N,M]) instead -- `W` is dead code, never used to init `Ws_`/`W_g_`. `W_g_` is
        # separately glorot-initialized, also never using the numpy `W_g` array. We
        # faithfully reproduce this: Ws (and B_g) are initialized from B = A.T/scale,
        # shape [N,M]; W_g/b_g are independent glorot-uniform-initialized parameters.
        B_np = (A.T / self._scale).astype(np.float32)
        B_g_np = B_np.copy()
        b_g_np = np.zeros((self._N, 1), dtype=np.float32)
        D_np = np.ones((self._N, 1), dtype=np.float32)

        if not untied:
            w_shared = nn.Parameter(torch.from_numpy(B_np.copy()))
            self.Ws = nn.ParameterList([w_shared for _ in range(T)])
        else:
            self.Ws = nn.ParameterList(
                [nn.Parameter(torch.from_numpy(B_np.copy())) for _ in range(T)]
            )

        self.thetas = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(theta_vec.copy())) for _ in range(T)]
        )
        self.altis = nn.ParameterList(
            [nn.Parameter(torch.full((1,), float(alti))) for _ in range(T)]
        )
        self.altis_over = nn.ParameterList(
            [nn.Parameter(torch.full((1,), float(alti))) for _ in range(T)]
        )
        log_eps = []
        for t in range(T):
            if t < 7:
                log_eps.append(self._logep)
            elif t <= 10:
                log_eps.append(self._logep - 2.0)
            else:
                log_eps.append(self._logep - 2.0)  # `_logeq` in the original is a typo/dead branch
        self.log_epsilons = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(v))) for v in log_eps]
        )
        self.Ds = nn.ParameterList([nn.Parameter(torch.from_numpy(D_np.copy())) for _ in range(T)])
        self.Ds_over = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(D_np.copy())) for _ in range(T)]
        )

        # B_g, b_g, W_g are shared (single glorot-init instance, replicated T times, as in the
        # original `B_g_ = B_g_ * self._T` list-replication of ONE tf.get_variable).
        Bg_shared = nn.Parameter(torch.empty(*B_g_np.shape))
        nn.init.xavier_uniform_(Bg_shared)
        self.B_g = nn.ParameterList([Bg_shared for _ in range(T)])

        bg_shared = nn.Parameter(torch.empty(*b_g_np.shape))
        nn.init.xavier_uniform_(bg_shared)
        self.b_g = nn.ParameterList([bg_shared for _ in range(T)])

        # W_g's numpy initializer (`eye(N) - B@A`) is likewise dead code in the original --
        # `tf.get_variable(shape=W_g.shape, initializer=glorot_uniform())` only reuses its
        # *shape* ([N, N]), not its values.
        Wg_shared = nn.Parameter(torch.empty(self._N, self._N))
        nn.init.xavier_uniform_(Wg_shared)
        self.W_g = nn.ParameterList([Wg_shared for _ in range(T)])

        self.register_buffer("kA", torch.from_numpy(A.copy()))

    def inference(self, y):
        """y: [batch, M]. Returns list of per-iteration sparse-code estimates [batch, N]."""
        batch = y.shape[0]
        xh = torch.zeros(batch, self._N, dtype=y.dtype, device=y.device)
        xhs = [xh]

        theta_p = None
        for t in range(self._T):
            W_ = self.Ws[t]
            theta_ = self.thetas[t].squeeze(-1)  # [N]
            B_g = self.B_g[t]
            W_g = self.W_g[t]
            b_g = self.b_g[t].squeeze(-1)  # [N]
            D = self.Ds[t].squeeze(-1)
            alti = self.altis[t]
            D_over = self.Ds_over[t].squeeze(-1)
            alti_over = self.altis_over[t]
            log_epsilon = self.log_epsilons[t]
            percent = float(self._ps[t])

            By = y @ W_.T  # tf.matmul(W_, y_) with y_:[M,batch] -> here y@W_.T, [batch,N]
            part_1_sig = torch.sigmoid(xh @ W_g.T + y @ B_g.T + b_g)
            part_2_sig = torch.abs(By)

            if t == 0:
                in_ = _gain(
                    xh, D, theta_ * 0.0 + 1.0, alti, torch.exp(log_epsilon), self.gain_gate[t]
                )
                part_2_inv = theta_
            else:
                in_ = _gain(
                    xh, D, theta_p * 0.0 + 1.0, alti, torch.exp(log_epsilon), self.gain_gate[t]
                )
                part_2_inv = theta_p

            res = y - (in_ * xh) @ self.kA.T
            xh_title, cindex = _shrink_ss_return_index(in_ * xh + res @ W_.T, theta_, percent)

            part_1_inv = 1.0 / (torch.abs(xh_title - xh) + 0.1) + 0.0 * part_1_sig
            if self.over_gate[t] == "inv":
                g_ = _overshoot(self._overshoot, alti_over, part_1_inv, part_2_inv) + 0.0 * D_over
            elif self.over_gate[t] == "sigm":
                g_ = _overshoot(self._overshoot, alti_over, part_1_sig, part_2_sig) + 0.0 * D_over
            else:  # "none"
                g_ = 1.0 + 0.0 * alti_over * part_1_sig * part_2_sig + 0.0 * D_over

            theta_p = theta_ * cindex
            xh = g_ * xh_title + (1 - g_) * xh
            xhs.append(xh)

        return xhs

    def forward(self, y):
        return self.inference(y)[-1]


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_glista():
    rng = np.random.RandomState(0)
    M, N = 25, 50
    A = rng.randn(M, N).astype(np.float32)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)  # unit-norm columns, matching the paper setup
    model = GLISTA_cp(
        A=A,
        T=4,
        B=1000,
        lam=0.4,
        uf="inv",
        percent=0.1,
        max_percent=1.0,
        untied=True,
        coord=True,
        alti=1.0,
        overshoot=True,
        gain=True,
        gain_fun="inv",
        over_fun="inv",
        both_gate=False,
        T_combine=2,
        T_middle=1,
    )
    model.eval()
    return model


def example_input_glista():
    """y: [batch, M] linear-measurement tensor (y = Ax + noise)."""
    batch = 4
    return torch.randn(batch, 25)


MENAGERIE_ENTRIES = [
    ("GLISTA", build_glista, example_input_glista, 2020, "ported-pytorch"),
]
