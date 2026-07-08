# SOURCE: vendored from https://github.com/wyuaq/FPN-OAMP-THz-Channel-Estimation @ main
# (model.py)
#
# FPN-OAMP (Fixed-Point-Network Orthogonal Approximate Message Passing, Yu
# et al., IEEE JSTSP 2023, arXiv:2211.15939): a deep-unrolled fixed-point
# solver for hybrid-field channel estimation in THz ultra-massive MIMO
# systems. Each fixed-point iteration alternates a closed-form linear
# estimator (`linear_estimator`, an OAMP-style gradient step using the
# precomputed measurement-matrix pseudo-inverse `W_pinv`) with a learned
# nonlinear "denoiser" (`nonlinear_estimator`, a residual CNN operating on
# the channel reshaped into an (8,16,16) tensor). `forward()` runs this
# fixed-point recursion under `torch.no_grad()` until convergence (or
# `max_depth` iterations), then performs one additional gradient-attached
# iteration when in training mode -- this is the real Deep Equilibrium /
# fixed-point-network training trick from the paper, executed exactly as in
# the real repo. `FPN_OAMP` is copied verbatim below (no architecture
# change; only the module-level `device` global is overridden to `"cpu"` at
# staging build time so construction/tracing works on a CPU-only or
# CPU-preferred trace host, matching the real repo's own
# `"cuda:0" if torch.cuda.is_available() else "cpu"` device-selection
# pattern).

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

device = "cpu"

measurements = torch.tensor
step_size = torch.tensor
latent_solution = torch.tensor
noise_level = torch.tensor
final_solution = torch.tensor


class FPN_OAMP(nn.Module):
    def __init__(
        self,
        A,
        lat_layers=3,
        contraction_factor=0.99,
        eps=1.0e-2,
        max_depth=15,
        structure="ResNet",
        num_channels=64,
    ):
        super(FPN_OAMP, self).__init__()
        self.A = A.to(device)  # measurement matrix
        self.W_pinv = torch.from_numpy(np.linalg.pinv(A)).to(
            device
        )  # pinv of the measurement matrix
        self.step = self.A.shape[1] / (torch.trace(torch.mm(self.W_pinv, self.A))).to(
            device
        )  # step size 'eta' (constant in each iteration)
        self.W_pinv_mul_step = (self.step * self.W_pinv).to(
            device
        )  # pinv of the measurement matrix, multiplied by the step size
        self._lat_layers = lat_layers  # number of latent ResNet blocks
        self.gamma = contraction_factor  # The targeted Lipschitz constant to ensure contraction mapping (self.gamma < 1)
        self.eps = (
            eps  # The relative tolerance which determines when the iterations should terminate
        )
        self.max_depth = max_depth  # The maximum # of iterations during *training* stage (Note: one can run for an arbitrary number of iterations during testing!)
        self.structure = (
            structure  # The structure of the latent layers, currently only supports 'ResNet'
        )
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.unflatten = nn.Unflatten(
            1, (8, 16, 16)
        )  # (8: 4subarrays * 2channels/subarray(real+imag)), 16*16: the size of the subarray
        self.depth = 0.0
        self.use_layer_norm = True  # the default choice is *True*.
        self.use_LS_initialization = True  # the default choice is *True*.
        self.use_contraction_safeguard = False  # the default choice is *False*, since the normal training producture can alrealy lead to a contractive operator!
        self.input_convs = nn.Conv2d(
            8, num_channels, kernel_size=3, stride=1, padding=(1, 1), bias=True
        )
        if self.use_layer_norm:
            self.input_layer_norm = nn.LayerNorm([num_channels, 16, 16])
        self.latent_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=num_channels,
                        kernel_size=3,
                        stride=1,
                        padding=(1, 1),
                    ),
                    self.relu,
                    nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=num_channels,
                        kernel_size=3,
                        stride=1,
                        padding=(1, 1),
                    ),
                    self.relu,
                )
                for _ in range(lat_layers)
            ]
        )  # the residual block
        if self.use_layer_norm:
            self.latent_layer_norm = nn.ModuleList(
                [nn.LayerNorm([num_channels, 16, 16]) for _ in range(lat_layers)]
            )
        self.output_convs = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1),
            self.leaky_relu,
            nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=1, stride=1),
        )  # no activation in the last layer

    def name(self):
        """
        Assign name to model depending on the architecture.
        """
        if self.structure == "ResNet":
            return "FPN_OAMP_ResNet"
        else:
            print("\nWarning: unsupported backbone network...\n")

    def device(self):
        device = next(self.parameters()).data.device
        return device

    def initialize_solution(self, y: measurements):
        """
        Initialize solution with an all-zero vector or the LS solution
        """
        batch_size = y.shape[0]
        if self.use_LS_initialization:
            h_init = torch.matmul(self.W_pinv, y.unsqueeze(-1)).squeeze(-1)
        else:
            h_init = torch.zeros(batch_size, self.A.shape[1], device=self.device())
        return h_init

    def linear_estimator(self, y: measurements, h: latent_solution) -> latent_solution:
        """
        Linear estimator (LE)
        """
        h = h.unsqueeze(-1)
        y = y.unsqueeze(-1)

        h = torch.matmul(self.W_pinv_mul_step, y - torch.matmul(self.A, h)).squeeze(-1) + h.squeeze(
            -1
        )
        y = y.squeeze(-1)

        return h  # shape: batch_size * 2N (N is the number of antennas)

    def nonlinear_estimator(self, y: measurements, h: latent_solution) -> latent_solution:
        """
        Nonlinear estimator (NLE)
        """
        batch_size = y.shape[0]

        if self.structure == "ResNet":
            # use unflatten to reshape h from column vector to tensor form
            h = self.unflatten(h)

            # input convolution
            h = self.leaky_relu(self.input_convs(h))
            if self.use_layer_norm:
                h = self.input_layer_norm(h)

            # latent convolution
            for idx, conv in enumerate(self.latent_convs):
                res = conv(h)  # 'res' stands for 'residual'
                h = h + res
                if self.use_layer_norm:
                    h = self.latent_layer_norm[idx](h)

            # output convolution
            h = self.output_convs(h)

            # flatten
            h = h.view(batch_size, -1)

            # normalize h with the contraction factor
            h = self.gamma * h

        else:
            print("\nWarning: unsupported backbone network...\n")

        return h  # shape: batch_size * 2N (N is the number of antennas)

    def latent_space_forward(self, y: measurements, h: latent_solution) -> latent_solution:
        """Fixed point operator in latent space (linear estimator + nonlinear estimator)"""
        h = self.linear_estimator(y, h)
        output_LE = h.detach()
        h = self.nonlinear_estimator(y, h)
        output_NLE = h.detach()

        return h, output_LE, output_NLE  # shape: batch_size * 2N (N is the number of antennas)

    def normalize_lip_const(self, y: measurements, h: latent_solution):
        """Scale the weights of the NLE to make it gamma Lipschitz.

        ## For MLP backbone (R denotes the NLE):
        It should hold that |R(h1,y) - R(h2,y)| <= gamma * |h1-h2| for all h1
        and h2. If this doesn't hold, then we must rescale the MLP.
        To rescale, we should multiply R by
            norm_fact = gamma * |u-w| / |R(u,y) - R(w,y)|,
        averaged over a batch of samples, i.e. R <-- norm_fact * R.

        ## For ResNet backbone (R denotes the NLE):
        Consider R = I + Conv. To rescale, ideally we multiply R by
            norm_fact = gamma * |u-w| / |R(u,y) - R(w,y)|,
        averaged over a batch of samples, i.e. R <-- norm_fact * R. The
        issue is that ResNets include an identity operation, which we don't
        wish to rescale. So, instead we use
            R <-- I + norm_fact * Conv,
        which is accurate up to an identity term scaled by (norm_fact - 1).
        If we do this often enough, then norm_fact ~ 1.0 and the identity
        term is negligible.
        """
        noise_h = torch.randn(h.size(), device=self.device())
        h_hat1 = self.latent_space_forward(y, h.copy() + noise_h)[0]
        h_hat2 = self.latent_space_forward(y, h)[0]
        R_diff_norm = torch.mean(torch.norm(h_hat1 - h_hat2, dim=1))
        noise_norm = torch.mean(torch.norm(noise_h, dim=1))
        R_is_gamma_lip = R_diff_norm <= (self.gamma * noise_norm)
        if not R_is_gamma_lip:
            violation_ratio = self.gamma * noise_norm / R_diff_norm
            normalize_factor = violation_ratio ** (1.0 / (2 * self._lat_layers))
            print("\n normalizing...")
            for i in range(self._lat_layers):
                if self.structure == "ResNet":
                    self.latent_convs[i][0].weight.data *= normalize_factor
                    self.latent_convs[i][0].bias.data *= normalize_factor
                    self.latent_convs[i][2].weight.data *= normalize_factor
                    self.latent_convs[i][2].bias.data *= normalize_factor
            self.input_convs.weight.data *= normalize_factor
            self.input_convs.bias.data *= normalize_factor
            self.output_convs[0].weight.data *= normalize_factor
            self.output_convs[0].bias.data *= normalize_factor
            self.output_convs[2].weight.data *= normalize_factor
            self.output_convs[2].bias.data *= normalize_factor

    def forward(self, y: measurements, depth_warning=False):
        """FPN-OAMP forward propagation
        Training process:
        With gradients *detached*, find the fixed point. Then attach gradient and perform one additional iteration to calculate the gradient.

        Testing process:
        With gradients *detached*, find the fixed point.
        """
        with torch.no_grad():
            self.depth = 0.0
            h = self.initialize_solution(y)
            h_prev = np.Inf * torch.ones(h.shape, device=self.device())
            termination = False
            while not termination and self.depth < self.max_depth:
                h_prev = h.clone()
                h = self.latent_space_forward(y, h)[0]  # original
                res_norm = torch.max(torch.norm(h - h_prev, dim=1))
                self.depth += 1.0
                termination = res_norm <= self.eps

            if self.training and self.use_contraction_safeguard:
                # normalize the parameters of the neural network to safeguard the contractive mapping
                self.normalize_lip_const(h_prev, y)

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        attach_gradients = self.training
        if attach_gradients:
            h = self.latent_space_forward(y, h)[0]
            return h
        else:
            return h.detach()


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). The real repo
# drives training with `num_measurements=512`, `num_antennas=1024`
# (`A.shape == (512, 2048)`, since the model internally works with the
# real+imag-concatenated 2N=2048-dim channel vector) and a real
# compressed-sensing measurement matrix loaded from a `.mat` file via
# `load_CS_matrix` (`channel_AoSA/generate_measurement_data.py`, not
# available without their proprietary dataset generation pipeline). The
# `Unflatten(1, (8,16,16))` in `nonlinear_estimator` hardcodes the 2048-dim
# channel-vector width (8*16*16=2048), so the antenna dimension can't be
# shrunk without touching the architecture; we shrink `num_measurements`
# (rows of A, still a valid instance of the model's CS measurement setup)
# and `lat_layers`/`num_channels`/`max_depth` for a fast CPU trace, using a
# random matrix in place of the real (dataset-dependent) CS dictionary --
# the model architecture and forward computation graph are unchanged.
# ---------------------------------------------------------------------------
_BATCH = 2
_NUM_MEASUREMENTS = 32
_NUM_ANTENNAS_FLAT = 2048  # fixed by nn.Unflatten(1, (8, 16, 16)) = 8*16*16


def build_fpn_oamp():
    torch.manual_seed(0)
    A = torch.randn(_NUM_MEASUREMENTS, _NUM_ANTENNAS_FLAT)
    model = FPN_OAMP(
        A=A,
        lat_layers=1,
        contraction_factor=0.99,
        eps=1e-2,
        max_depth=3,
        structure="ResNet",
        num_channels=4,
    )
    model.eval()
    return model


def example_input_fpn_oamp():
    torch.manual_seed(0)
    return torch.randn(_BATCH, _NUM_MEASUREMENTS)


MENAGERIE_ENTRIES = [
    ("FPN-OAMP", "build_fpn_oamp", "example_input_fpn_oamp", 2023, MENAGERIE_ZOO),
]
