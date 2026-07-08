# FAITHFUL PORT of https://github.com/VITA-Group/LISTA-CPSS @ master (1598fbc1)
# (models/LISTA_cp.py :: LISTA_cp; utils/tf.py :: shrink_free) (original framework: TensorFlow 1.x)
#
# "Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and
# Thresholds" (Chen, Liu, Chen, Xu, Wang, NeurIPS 2018 spotlight). The repo
# implements LISTA_cp (Learned ISTA with tied/coupled weights, the paper's LISTA-CP
# baseline that LISTA-CPSS builds on) and LISTA_cpss. It is TensorFlow-1.x
# (tf.variable_scope/tf.get_variable graph-mode API), which cannot be installed
# alongside the torch-only base env; a PyTorch reimplementation is straightforward
# because the architecture is fully specified by a small self-contained class.
#
# This module is a faithful port of models/LISTA_cp.py's LISTA_cp class, transcribed
# op-for-op from the real TF1 code (reproduced below for traceability):
#
#     def setup_layers(self):
#         W = (np.transpose(self._A) / self._scale).astype(np.float32)
#         with tf.variable_scope(self._scope, reuse=False) as vs:
#             self._kA_ = tf.constant(value=self._A, dtype=tf.float32)
#             if not self._untied:  # tied model
#                 Ws_.append(tf.get_variable(name='W', dtype=tf.float32, initializer=W))
#                 Ws_ = Ws_ * self._T
#             for t in range(self._T):
#                 thetas_.append(tf.get_variable(name="theta_%d" % (t+1),
#                                  dtype=tf.float32, initializer=self._theta))
#                 if self._untied:  # untied model
#                     Ws_.append(tf.get_variable(name="W_%d" % (t+1),
#                                  dtype=tf.float32, initializer=W))
#         self.vars_in_layer = list(zip(Ws_, thetas_))
#
#     def inference(self, y_, x0_=None):
#         xhs_ = []
#         xh_ = tf.zeros(shape=(self._N, batch_size)) if x0_ is None else x0_
#         xhs_.append(xh_)
#         with tf.variable_scope(self._scope, reuse=True) as vs:
#             for t in range(self._T):
#                 W_, theta_ = self.vars_in_layer[t]
#                 res_ = y_ - tf.matmul(self._kA_, xh_)
#                 xh_ = shrink_free(xh_ + tf.matmul(W_, res_), theta_)
#                 xhs_.append(xh_)
#         return xhs_
#
# and utils/tf.py's shrink_free:
#
#     def shrink_free(input_, theta_):
#         return tf.sign(input_) * tf.maximum(tf.abs(input_) - theta_, 0.0)
#
# Constructor init math (self._scale, self._theta from LISTA_cp.__init__) and the
# tied-vs-untied weight-sharing behavior are preserved exactly; only the TF1
# variable-scope plumbing is replaced with plain nn.Parameter storage (a torch
# nn.Module has no TF-style implicit variable graph, so parameters are declared
# directly instead of created via tf.get_variable inside a scope).

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def shrink_free(x, theta):
    """Soft-threshold shrinkage without a positivity constraint on theta
    (verbatim port of utils/tf.py's shrink_free)."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0.0)


class LISTA_cp(nn.Module):
    """Faithful torch port of models/LISTA_cp.py's LISTA_cp: Learned ISTA with
    weight coupling (Chen et al., NeurIPS 2018).

    Args:
        A (Tensor or ndarray): (M, N) sensing/dictionary matrix.
        T (int): number of unrolled layers (depth).
        lam (float): initial threshold value (pre-scaling), matching the real
            constructor's `lam` argument.
        untied (bool): if True, each of the T layers gets its own W_t (matches
            the real `_untied` flag); if False (tied model), one shared W is
            reused across all T layers exactly as the real
            `Ws_ = Ws_ * self._T` line does.
        coord (bool): if True, use a per-coordinate (per-row-of-A) threshold
            vector instead of a scalar threshold, matching the real `_coord`
            flag's `self._theta = np.ones((self._N, 1)) * self._theta`.
    """

    def __init__(self, A, T, lam=0.4, untied=False, coord=False):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32)
        M, N = A.shape
        self.T = T
        self.M = M
        self.N = N
        self.untied = untied
        self.coord = coord

        scale = 1.001 * torch.linalg.matrix_norm(A, ord=2) ** 2
        theta_init = lam / scale
        if coord:
            theta_init = torch.ones(N, 1) * theta_init

        # constant sensing matrix (registered as a non-trainable buffer, matching
        # the real tf.constant self._kA_)
        self.register_buffer("A", A)

        W = A.t() / scale  # (N, M), matches real `W = transpose(A) / scale`

        if untied:
            self.W = nn.ParameterList([nn.Parameter(W.clone()) for _ in range(T)])
        else:
            # tied model: one shared weight matrix reused every layer (real
            # code builds a single tf.get_variable and repeats its handle T
            # times via `Ws_ = Ws_ * self._T`)
            self.W_shared = nn.Parameter(W.clone())

        self.theta = nn.ParameterList(
            [
                nn.Parameter(theta_init.clone() if coord else theta_init.clone().reshape(1))
                for _ in range(T)
            ]
        )

    def forward(self, y, x0=None):
        """
        Args:
            y (Tensor): (M, batch) measurement matrix, matches the real
                `inference(self, y_, x0_=None)` signature/orientation.
            x0 (Tensor | None): (N, batch) initial sparse-code estimate;
                zeros if None (matches the real default).

        Returns:
            Tensor: (N, batch) final layer's sparse-code estimate (the last
            element of the real code's `xhs_` list).
        """
        batch_size = y.shape[-1]
        if x0 is None:
            xh = torch.zeros(self.N, batch_size, dtype=y.dtype, device=y.device)
        else:
            xh = x0

        for t in range(self.T):
            W_t = self.W[t] if self.untied else self.W_shared
            theta_t = self.theta[t]
            res = y - torch.matmul(self.A, xh)
            xh = shrink_free(xh + torch.matmul(W_t, res), theta_t)

        return xh


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
#
# The real repo's own default experiment config (config.py) uses a Gaussian
# sensing matrix A of shape (M, N) with M < N (compressed sensing regime), T
# unrolled layers, lam=0.4. We keep those defaults at a small scale.
# ---------------------------------------------------------------------------
_M = 10
_N = 20
_T = 4
_BATCH = 3


def build_lista_cp():
    torch.manual_seed(0)
    A = np.random.RandomState(0).normal(size=(_M, _N)).astype(np.float32) / np.sqrt(_M)
    model = LISTA_cp(A, T=_T, lam=0.4, untied=False, coord=False)
    model.eval()
    return model


def example_input_lista_cp():
    torch.manual_seed(0)
    y = torch.randn(_M, _BATCH)
    return (y,)


MENAGERIE_ENTRIES = [
    ("LISTA-CP", "build_lista_cp", "example_input_lista_cp", 2018, MENAGERIE_ZOO),
]
