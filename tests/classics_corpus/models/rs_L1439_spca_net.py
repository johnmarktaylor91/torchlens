# SOURCE: vendored from xianchaoxiu/SPCA-Net @ d8364f530908c20e0327be64378a8ac0923bf52e
# https://raw.githubusercontent.com/xianchaoxiu/SPCA-Net/main/SPCA-Net/demo.py
#
# SPCA-Net ("Tuning-Free Structured Sparse PCA via Deep Unfolding Networks",
# L. Chen & X. Xiu, 2025 44th Chinese Control Conference (CCC 2025)). The repo's
# `demo.py` unfolds an ADMM iteration for structured sparse PCA into a stack of
# learnable `ADMMStage` modules (`DeepADMM`): each stage performs a linear/SVD-based
# X-update (learned step size `eta` applied through a `nn.Linear` correction term,
# followed by an SVD retraction back onto the Stiefel manifold), a row-sparse
# (l2,1-norm soft-threshold) Y-update, an element-wise (l1-norm soft-threshold)
# Z-update, and dual-variable (Lambda, Pi) ascent steps -- exactly the deep-unfolded
# ADMM iterates from the paper, with the learnable eta/lambda/mu/alpha/beta
# parameters per stage.
#
# `ADMMStage` and `DeepADMM` are transcribed verbatim from `demo.py`. No
# architectural changes were made -- every parameter, matrix update, and SVD/soft-
# threshold step in `ADMMStage.forward` / `DeepADMM.forward` is unchanged. Only the
# module-level CLI/training/data-loading block (`load_data`, `train_deep_admm`,
# `export_model_parameters`, `save_results_to_mat`, and the `if __name__ ==
# "__main__"` script) is dropped since it is training/IO plumbing, not part of the
# traced architecture. `DeepADMM.forward` samples its own random initial `X` via
# `torch.randn`, so `example_input_spca_net` supplies only the input matrix `A`
# (the real, only, forward argument) -- to keep the trace deterministic under
# TorchLens's RNG-managed capture, `build_spca_net` seeds `torch.manual_seed` once
# at construction time exactly like `DeepADMM`'s own internal `torch.randn` init.

import torch
import torch.nn as nn


class ADMMStage(nn.Module):
    def __init__(self, d, k):
        super(ADMMStage, self).__init__()
        # Learnable parameters
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.lambda_param = nn.Parameter(torch.tensor(1e-3))
        self.mu_param = nn.Parameter(torch.tensor(1e-3))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Linear layer for X update
        self.linear = nn.Linear(d * k, d * k, bias=False)
        # initialize linear layer weights to identity
        nn.init.eye_(self.linear.weight)

    def x_update(self, X, Y, Z, Lambda, Pi, A):
        d, k = X.size()

        # Compute M
        M = (
            self.alpha * (X - Y + Lambda / self.alpha)
            + self.beta * (X - Z + Pi / self.beta)
            + A @ A.t() @ X
        )

        # Linear approximation
        X_hat = X - self.eta * (self.linear(M.view(-1)).view(d, k))

        # SVD layer
        U, _, V = torch.svd(X_hat)
        X_new = U @ V.t()

        return X_new

    def y_update(self, X, Lambda):
        # Compute row-wise l2 norm
        row_norms = torch.norm(X + Lambda / self.alpha, dim=1, keepdim=True)
        # Avoid division by zero
        row_norms = torch.clamp(row_norms, min=1e-10)
        # Soft thresholding using ReLU
        scale = torch.relu(row_norms - self.lambda_param / self.alpha) / row_norms
        Y = (X + Lambda / self.alpha) * scale
        return Y

    def z_update(self, X, Pi):
        # Element-wise soft thresholding
        Z = torch.sign(X + Pi / self.beta) * torch.relu(
            torch.abs(X + Pi / self.beta) - self.mu_param / self.beta
        )
        return Z

    def forward(self, X, Y, Z, Lambda, Pi, A):
        # Update X
        X_new = self.x_update(X, Y, Z, Lambda, Pi, A)

        # Update Y
        Y_new = self.y_update(X_new, Lambda)

        # Update Z
        Z_new = self.z_update(X_new, Pi)

        # Update Lagrange multipliers
        Lambda_new = Lambda + self.alpha * (X_new - Y_new)
        Pi_new = Pi + self.beta * (X_new - Z_new)

        return X_new, Y_new, Z_new, Lambda_new, Pi_new


class DeepADMM(nn.Module):
    def __init__(self, d, k, num_stages=5):
        super(DeepADMM, self).__init__()
        self.d = d
        self.k = k
        self.num_stages = num_stages

        # Create multiple ADMM stages
        self.stages = nn.ModuleList([ADMMStage(d, k) for _ in range(num_stages)])

        # storage for final state
        self.final_X = None
        self.final_Y = None
        self.final_Z = None

    def forward(self, A):
        # Initialize variables
        X = torch.randn(self.d, self.k, device=A.device)
        X = self.orthogonalize(X)
        Y = X.clone()
        Z = X.clone()
        Lambda = torch.zeros_like(X)
        Pi = torch.zeros_like(X)

        # Store intermediate results
        X_history = [X]

        # Apply ADMM stages
        for stage in self.stages:
            X, Y, Z, Lambda, Pi = stage(X, Y, Z, Lambda, Pi, A)
            X_history.append(X)

        # store final state
        self.final_X = X
        self.final_Y = Y
        self.final_Z = Z

        return X, X_history

    @staticmethod
    def orthogonalize(X):
        # Orthogonalize X using SVD
        U, _, V = torch.svd(X)
        return U @ V.t()


def build_spca_net():
    torch.manual_seed(0)
    d = 12
    k = 4
    num_stages = 3
    model = DeepADMM(d, k, num_stages)
    model.eval()
    return model


def example_input_spca_net():
    torch.manual_seed(0)
    d = 12
    n_samples = 8
    return torch.randn(d, n_samples)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SPCA-Net", "build_spca_net", "example_input_spca_net", 2025, MENAGERIE_ZOO),
]
