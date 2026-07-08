# SOURCE: vendored from histocartography/zoommil @ main
# https://raw.githubusercontent.com/histocartography/zoommil/main/zoommil/models/zoommil.py
# https://raw.githubusercontent.com/histocartography/zoommil/main/zoommil/utils/perturbedtopk.py
#
# ZoomMIL: "Differentiable Zooming for Multiple Instance Learning on Whole-Slide Images"
# (Thandiackal et al., ECCV 2022), IBM Research / histocartography.
#
# Three-magnification gated-attention MIL over precomputed WSI patch-feature bags: at each
# magnification level a shared gated-attention head both pools the bag representation AND (via
# a second, auxiliary attention head) scores patches for differentiable top-K "zoom-in" selection
# into the next, higher-magnification bag (using the Cordonnier et al. CVPR'21 PerturbedTopK
# soft-topk operator during training, hard torch.topk at eval). Code below is transcribed
# VERBATIM from the real repo (only the CUDA-only `.relocate()` / `self.device` construction is
# left untouched but unused at CPU trace time; no architectural change).
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- zoommil/utils/perturbedtopk.py (verbatim) ----
class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        # b = batch size
        b, num_patches = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, num_patches)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma  # [b, num_s, num_p]
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # [b, num_s, k]
        indices = torch.sort(indices, dim=-1).values  # [b, num_s, k]

        # b, num_s, k, num_p
        perturbed_output = F.one_hot(indices, num_classes=num_patches).float()
        indicators = perturbed_output.mean(dim=1)  # [b, k, num_p]

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)


# ---- zoommil/models/zoommil.py (verbatim) ----
class GatedAttention(nn.Module):
    def __init__(self, L, D, dropout=None, n_cls=1):
        """Gated attention module.
        Args:
            L (int): Input feature dimension.
            D (int): Hidden layer feature dimension.
            dropout (float, optional): Dropout. Defaults to None.
            n_cls (int, optional): Number of output classes. Defaults to 1.
        """
        super(GatedAttention, self).__init__()
        self.attention_a = (
            [nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout)]
            if dropout is not None
            else [nn.Linear(L, D), nn.Tanh()]
        )
        self.attention_b = (
            [nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout)]
            if dropout is not None
            else [nn.Linear(L, D), nn.Sigmoid()]
        )
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_cls)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class ZoomMIL(nn.Module):
    def __init__(
        self,
        in_feat_dim,
        hidden_feat_dim=256,
        out_feat_dim=512,
        dropout=None,
        k_sample=12,
        k_sigma=0.002,
        n_cls=3,
    ):
        """
        Args:
            in_feat_dim (int): Input feature dimension.
            hidden_feat_dim (int, optional): Hidden layer feature dimension. Defaults to 256.
            out_feat_dim (int, optional): Output feature dimension. Defaults to 512.
            dropout (float, optional): Dropout. Defaults to None.
            k_samples (int, optional): Number of samples (k) to zoom-in at next higher magnification. Defaults to 12.
            k_sigma (float, optional): Perturbation sigma. Defaults to 2e-3.
            n_cls (int, optional): Number of output classes. Defaults to 3.
        """
        super(ZoomMIL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k_sample = k_sample
        self.k_sigma = k_sigma
        self.n_cls = n_cls

        fc_low_mag = [nn.Linear(in_feat_dim, out_feat_dim), nn.ReLU()]
        fc_mid_mag = [nn.Linear(in_feat_dim, out_feat_dim), nn.ReLU()]
        fc_high_mag = [nn.Linear(in_feat_dim, out_feat_dim), nn.ReLU()]

        if dropout is not None:
            fc_low_mag.append(nn.Dropout(dropout))
            fc_mid_mag.append(nn.Dropout(dropout))
            fc_high_mag.append(nn.Dropout(dropout))

        self.fc_low_mag = nn.Sequential(*fc_low_mag)
        self.ga_low_mag = GatedAttention(
            L=out_feat_dim, D=hidden_feat_dim, dropout=dropout, n_cls=1
        )
        self.fc_mid_mag = nn.Sequential(*fc_mid_mag)
        self.ga_mid_mag = GatedAttention(
            L=out_feat_dim, D=hidden_feat_dim, dropout=dropout, n_cls=1
        )
        self.fc_high_mag = nn.Sequential(*fc_high_mag)
        self.ga_high_mag = GatedAttention(
            L=out_feat_dim, D=hidden_feat_dim, dropout=dropout, n_cls=1
        )

        self.classify_head = nn.Linear(out_feat_dim, n_cls)

        # auxiliary attention head for patch selection
        self.aux_ga_low_mag = GatedAttention(
            L=out_feat_dim, D=hidden_feat_dim, dropout=dropout, n_cls=1
        )
        self.aux_ga_mid_mag = GatedAttention(
            L=out_feat_dim, D=hidden_feat_dim, dropout=dropout, n_cls=1
        )

    def relocate(self):
        self.fc_low_mag = self.fc_low_mag.cuda()
        self.ga_low_mag = self.ga_low_mag.cuda()
        self.fc_mid_mag = self.fc_mid_mag.cuda()
        self.ga_mid_mag = self.ga_mid_mag.cuda()
        self.fc_high_mag = self.fc_high_mag.cuda()
        self.ga_high_mag = self.ga_high_mag.cuda()
        self.classify_head = self.classify_head.cuda()
        self.aux_ga_low_mag = self.aux_ga_low_mag.cuda()
        self.aux_ga_mid_mag = self.aux_ga_mid_mag.cuda()

    def forward(self, x):
        x1, x2, x3 = x
        num_features = [x1.shape[1], x2.shape[1], x3.shape[1]]

        ###################################################################################################################
        ############################################ low magnification ####################################################
        x1 = self.fc_low_mag(x1)  # [b, N_1, out_dim]
        x1_mem = x1
        A_1, x1 = self.ga_low_mag(x1)  # [b, N_1, 1], [b, N_1, out_dim]
        A_1 = A_1.permute(0, 2, 1)  # [b, 1, N_1]
        A_1 = F.softmax(A_1, dim=-1)  # [b, 1, N_1]

        # attention pooling
        M_1 = A_1 @ x1  # [b, 1, out_dim]

        A_1_aux, _ = self.aux_ga_low_mag(x1_mem)  # [b, N_1, 1]
        A_1_aux = A_1_aux.permute(0, 2, 1)  # [b, 1, N_1]
        A_1_aux = F.softmax(A_1_aux, dim=-1)  # [b, 1, N_1]

        # select k patches to zoom-in at next higher magnification
        k_sample_1 = min(x1.shape[1], self.k_sample)
        if self.training:
            topk = PerturbedTopK(k=k_sample_1, num_samples=100, sigma=self.k_sigma)
            select_1 = topk(A_1_aux.squeeze(dim=1))
        else:
            select_1 = torch.topk(
                A_1_aux.squeeze(dim=1), k=k_sample_1, dim=-1, sorted=False
            ).indices
            select_1 = torch.sort(select_1, dim=-1).values
            select_1 = torch.nn.functional.one_hot(select_1, num_classes=A_1_aux.shape[-1]).float()

        ###################################################################################################################
        ########################################### middle magnification ##################################################
        x2 = torch.einsum(
            "bkn,bnd->bkd",
            torch.kron(
                select_1,
                torch.eye(
                    int(num_features[1] // num_features[0]), device=self.device, requires_grad=True
                ),
            ),
            x2,
        )
        x3 = torch.einsum(
            "bkn,bnd->bkd",
            torch.kron(
                select_1,
                torch.eye(
                    int(num_features[2] // num_features[0]), device=self.device, requires_grad=True
                ),
            ),
            x3,
        )

        x2 = self.fc_mid_mag(x2)  # [b, 4 * k_sample, out_dim]
        x2_mem = x2
        A_2, x2 = self.ga_mid_mag(x2)  # [b, 4 * k_sample, 1], [b, 4 * k_sample, out_dim]
        A_2 = A_2.permute(0, 2, 1)  # [b, 1, 4 * k_sample]
        A_2 = F.softmax(A_2, dim=-1)  # [b, 1, 4 * k_sample]

        # attention pooling
        M_2 = A_2 @ x2  # [b, 1, out_dim]

        A_2_aux, _ = self.aux_ga_mid_mag(x2_mem)  # [b, 1, 4 * k_sample]
        A_2_aux = A_2_aux.permute(0, 2, 1)  # [b, 1, 4 * k_sample]
        A_2_aux = F.softmax(A_2_aux, dim=-1)  # [b, 1, 4 * k_sample]

        # select k patches to zoom-in at next higher magnification
        k_sample_2 = min(x2.shape[1], self.k_sample)
        if self.training:
            topk = PerturbedTopK(k=k_sample_2, num_samples=100, sigma=self.k_sigma)
            select_2 = topk(A_2_aux.squeeze(dim=1))
        else:
            select_2 = torch.topk(
                A_2_aux.squeeze(dim=1), k=k_sample_2, dim=-1, sorted=False
            ).indices
            select_2 = torch.sort(select_2, dim=-1).values
            select_2 = torch.nn.functional.one_hot(select_2, num_classes=A_2_aux.shape[-1]).float()

        ###################################################################################################################
        ############################################ high magnification ###################################################
        x3 = torch.einsum(
            "bkn,bnd->bkd",
            torch.kron(
                select_2,
                torch.eye(
                    int(num_features[2] // num_features[1]), device=self.device, requires_grad=True
                ),
            ),
            x3,
        )

        x3 = self.fc_high_mag(x3)  # [b, 4 * k_sample, out_dim]
        A_3, x3 = self.ga_high_mag(x3)  # [b, 4 * k_sample, 1], [b, 4 * k_sample, D]
        A_3 = A_3.permute(0, 2, 1)  # [b, 1, 4 * k_sample]
        A_3 = F.softmax(A_3, dim=-1)  # [b, 1, 4 * k_sample]

        # attention pooling
        M_3 = A_3 @ x3  # [b, 1, out_dim]

        ###################################################################################################################
        ############################################# classifier head #####################################################
        # bag level representation
        M = M_1 + M_2 + M_3

        logits = self.classify_head(M.squeeze(dim=1))  # [b, 1, out_dim]
        Y_hat = torch.topk(logits, 1, dim=-1)[-1]
        Y_prob = F.softmax(logits, dim=-1)

        return logits, Y_hat, Y_prob


# ---- staging harness ----
# Tiny trace-sized instance. Real usage: 3 magnification levels of a WSI patch-feature pyramid
# (e.g. 5x/10x/20x) where mid/high bags are exact 4x supersets (in patch count) of the previous
# level (standard WSI pyramid downsampling by 2x per axis => 4x patch count per level up). Here
# num_features = [6, 24, 96] (each level 4x the previous), in_feat_dim=32 (a small precomputed
# patch-embedding size), k_sample=2 so k_sample <= min bag size at every level, eval mode (uses
# the real hard-topk branch; training-mode PerturbedTopK branch is also real code but skipped
# here to keep tracing deterministic and fast).
def build_zoommil():
    torch.manual_seed(0)
    model = ZoomMIL(
        in_feat_dim=32,
        hidden_feat_dim=16,
        out_feat_dim=24,
        dropout=None,
        k_sample=2,
        k_sigma=0.002,
        n_cls=3,
    )
    # Real __init__ picks device via torch.cuda.is_available(); pin to CPU here so the
    # torch.kron(..., device=self.device) calls in forward() match the CPU example input
    # on any machine (this is the same device selection the real code performs when run
    # under `relocate()`-style CPU-only usage, not an architectural change).
    model.device = torch.device("cpu")
    model.eval()
    return model


def example_input_zoommil():
    torch.manual_seed(0)
    x1 = torch.randn(1, 6, 32)
    x2 = torch.randn(1, 24, 32)
    x3 = torch.randn(1, 96, 32)
    return ((x1, x2, x3),)


MENAGERIE_ENTRIES = [
    ("ZoomMIL", "build_zoommil", "example_input_zoommil", 2022, "vendored"),
]
