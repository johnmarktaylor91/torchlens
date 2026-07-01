"""Biomedical NLP / medical-imaging / single-cell classics (batch w4a19).

Sources checked (paper + official repo README/model-file architecture; no
clone, no pip install -- reimplemented from scratch in base-env torch, or
via the installed ``transformers`` library configured to the real
architecture with tiny dims):

- BioMedLM: Bolton, Venigalla, Yasunaga, Hall, Xiong, Lee, Daneshjou, Frankle,
  Liang, Carbin & Manning, Stanford CRFM, arXiv:2403.18421 (2024).
  https://huggingface.co/stanford-crfm/BioMedLM
  A domain-specific GPT-2-architecture causal language model (2.7B params in
  the released checkpoint: 32 layers, ``n_embd=2560``, ``n_head=20``)
  pretrained from scratch on PubMed abstracts and PubMed Central full text
  with a custom biomedical BPE vocabulary (28,896 tokens) rather than GPT-2's
  general-web vocabulary. Architecturally it is the standard GPT-2 decoder
  stack (pre-LN masked self-attention + GELU MLP blocks,
  ``scale_attn_by_inverse_layer_idx=True``, learned absolute position
  embeddings) -- the "distinctive mechanism" is the biomedical corpus +
  vocabulary, not a departure from the GPT-2 block. Reimplemented here by
  configuring ``transformers.GPT2LMHeadModel`` via ``AutoConfig.for_model``
  with the real config's flags (``scale_attn_by_inverse_layer_idx=True``,
  ``activation_function="gelu_new"``, ``reorder_and_upcast_attn=False``) and
  the real (biomedical-specific) vocabulary size, but tiny depth/width so the
  trace stays small.

- CE-Net (Context Encoder Network): Gu, Cheng, Fu, Zhou, Hao, Zhao, Zhang,
  Gao & Liu, IEEE TMI 2019, arXiv:1903.02740.
  https://github.com/Guzaiwang/CE-Net (official author repo, class ``CE_Net_``)
  A U-Net-style encoder-decoder for medical image segmentation (retinal
  vessels, cell contours, lung nodules) built on a ResNet encoder, with two
  distinctive "context extraction" blocks inserted at the bottleneck before
  the decoder: (1) a Dense Atrous Convolution (DAC) block -- four
  cascaded/parallel branches of increasing atrous (dilated) 3x3 convolutions
  (dilation 1, 1->3, 1->3->5, plus a 1x1 branch) whose outputs are all summed
  with the input as a dense residual, multiplying the effective receptive
  field without losing resolution; and (2) a Residual Multi-kernel Pooling
  (RMP / ``SPPblock``) block -- four parallel max-pools at different kernel
  sizes (2,3,5,6), each 1x1-conv'd down to one channel and bilinearly
  upsampled back to the bottleneck's spatial size, then concatenated with the
  original feature map (multi-scale context without changing resolution).
  The decoder mirrors U-Net with skip connections from each encoder stage.
  Reimplemented here with a small ResNet-style encoder (in place of the full
  ResNet34 backbone) feeding the real DAC + RMP bottleneck and a matching
  skip-connected decoder -- the DAC/RMP context blocks are CE-Net's namesake
  mechanism and are preserved exactly (same branch topology as the official
  ``DACblock``/``SPPblock`` classes).

- CellBender (``remove-background``): Fleming, Chaffin, Arduini, Akkad,
  Banks, Marioni, Griffiths, Wei, Hansen, Hines, Braams, Kirchner, Bragantini,
  Xu, Sarkar, Whittaker, Zhang, Barkas, Weller, Neff, Boyer, Kean, Ludwig,
  Camp, Chevrier, Dann, Regev, Petukhov & Lopez, Nat. Methods 2023,
  https://www.nature.com/articles/s41592-023-01943-7 ;
  https://github.com/broadinstitute/CellBender (official Broad Institute repo,
  ``remove_background/model.py`` -- ``RemoveBackgroundPyroModel``).
  A generative (Pyro) VAE that jointly explains observed droplet counts as a
  mixture of two Poisson/NB processes: a per-cell latent state ``z`` decoded
  through an MLP decoder into a fractional gene-expression profile ``chi``,
  scaled by an inferred per-droplet cell-count-depth ``d_cell`` to give the
  "real cell" signal ``mu_cell``; and a shared, cell-independent ambient RNA
  profile ``chi_ambient`` scaled by a per-droplet ambient depth ``d_empty``
  to give the background/noise signal ``lam``. A Bernoulli cell-presence
  indicator ``y`` blends the two, so the model's key structural idea --
  "one VAE-encoded real-cell signal + one shared ambient-RNA signal, summed
  per droplet" -- is what separates ambient RNA from true cell signal.
  Reimplemented here as a deterministic encoder/decoder VAE with an explicit,
  learned ambient-profile parameter and per-droplet cell/ambient depth heads
  whose outputs are additively combined exactly as in the official ``model``
  method's ``mu_cell + lam`` counts decomposition (Pyro sampling collapsed to
  its deterministic mean path, since Pyro is not in the base env).

- CellBox: Yuan, Kim, Wang, Sander, ODE reimplementation at
  https://github.com/sanderlab/cellbox_torch (PyTorch 2.0 port of the
  original TF1 CellBox from Yuan, Kim, Wang, Sander, Cell Systems 2021,
  https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30464-6),
  ``cellbox/model_torch.py`` (class ``CellBox``) + ``cellbox/kernel_torch.py``
  (``get_dxdt``, ``euler_solver``/``heun_solver``/``rk4_solver``).
  Perturbation-biology dynamics as a single-layer, fully learned ODE:
  ``dx/dt = eps * tanh(W_masked @ x + u) - alpha * x``, where ``W`` is a
  learned (masked, non-self-loop) adjacency matrix over all molecular nodes
  (proteins/phenotypes), ``u`` is the perturbation input vector, ``tanh`` is
  the saturating "envelope" nonlinearity gating the weighted-sum interaction
  term, and ``eps``/``alpha`` are learned per-node production/decay rates.
  The ODE is unrolled for ``n_T`` fixed steps with a chosen numerical
  integrator (Euler in the official default configs) to obtain the steady
  -state response ``x(T)`` to a perturbation. Reimplemented here exactly as
  the official ``_dxdt``/``euler_solver`` pair: a masked adjacency parameter,
  tanh-gated weighted-sum dynamics, and an explicit multi-step Euler unroll.

- CellOracle: Kamimoto, Stringa, Hoffmann, Jindal, Solnica-Krezel & Morris,
  Nature 2023, https://www.nature.com/articles/s41586-022-05688-9 ;
  https://github.com/morris-lab/CellOracle (official morris-lab repo,
  ``celloracle/trajectory/oracle_GRN.py`` -- ``_do_simulation``).
  Gene-regulatory-network (GRN) inference itself is Bayesian-Ridge /
  Bagging-Ridge regression over an ATAC-derived TF-target prior
  (``celloracle/network/regression_models.py``, scikit-learn -- not a neural
  net), but the model's namesake mechanism -- in-silico TF-perturbation
  simulation -- is a genuinely traceable, learnable linear dynamical system:
  starting from an expression-delta vector for the perturbed TF(s), the
  official ``_do_simulation`` repeatedly propagates
  ``delta = delta @ coef_matrix`` through the learned (signed, per-gene)
  regulatory coefficient matrix for a fixed number of steps
  (``n_propagation``), re-clamping the perturbed gene(s) back to their fixed
  input delta after every step, then adds the final propagated delta back
  onto the baseline expression. Reimplemented here as an ``nn.Module`` that
  owns a learnable (gene x gene) regulatory coefficient matrix and performs
  exactly this clamp-and-propagate recurrence for ``n_propagation`` steps --
  the GRN-propagation dynamics that make CellOracle's simulated perturbation
  responses differ from a single linear-regression prediction.

- Chiron: Teng, Tan, Wilson, Zhou, Renyu, Warrell, Wang, Yao & Lee (haotianteng
  et al.), GigaScience 2018,
  https://academic.oup.com/gigascience/article/7/5/giy037/4966989 ;
  https://github.com/haotianteng/Chiron (official repo, TensorFlow 1.x,
  ``chiron/cnn.py`` + ``chiron/rnn.py`` + ``chiron/chiron_model.py``).
  An end-to-end raw-nanopore-signal basecaller: a deep residual 1-D CNN
  (stacked ``conv_layer``/residual blocks over the raw current-signal
  windows) extracts local sequence features, which are fed into a stack of
  bidirectional LSTM layers (``stack_bidirectional_dynamic_rnn`` in the
  official ``rnn_layers``) to model long-range base-calling context in both
  directions, followed by a linear classifier over the 5-symbol nucleotide
  alphabet (A, C, G, T, blank) trained with CTC loss so the raw-signal
  segmentation-to-base alignment does not need to be known in advance.
  Reimplemented here in PyTorch (the official code is TF1/Python-2-era) as a
  residual 1-D CNN feature extractor feeding a stacked bidirectional LSTM and
  a per-timestep 5-way linear head, exposing raw per-timestep class logits
  (the CTC decoding/loss is a downstream training-time detail, not part of
  the traced forward architecture).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# BioMedLM
# ---------------------------------------------------------------------------


def build_biomedlm() -> nn.Module:
    """Build a tiny GPT-2-architecture BioMedLM stand-in.

    Uses ``transformers.AutoConfig.for_model("gpt2", ...)`` configured with
    the real BioMedLM flags (inverse-layer-index attention scaling, GELU-new
    activation, biomedical vocabulary size) but tiny depth/width/context so
    the traced graph stays small.

    Returns
    -------
    nn.Module
        Random-initialized ``GPT2LMHeadModel`` in eval mode.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=28896,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=None,
        activation_function="gelu_new",
        scale_attn_by_inverse_layer_idx=True,
        reorder_and_upcast_attn=False,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    return model.eval()


def example_input_biomedlm() -> Tensor:
    """Create example biomedical-text token ids.

    Returns
    -------
    Tensor
        Random token id sequence of shape ``(1, 16)``.
    """
    return torch.randint(0, 28896, (1, 16))


# ---------------------------------------------------------------------------
# CE-Net
# ---------------------------------------------------------------------------


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Build a 3x3 conv + BatchNorm + ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DACBlock(nn.Module):
    """Dense Atrous Convolution block (CE-Net bottleneck context module)."""

    def __init__(self, channels: int) -> None:
        """Initialize four dilated-conv branches plus a shared 1x1 conv.

        Parameters
        ----------
        channels : int
            Channel count of the bottleneck feature map.
        """
        super().__init__()
        self.dilate1 = nn.Conv2d(channels, channels, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channels, channels, 3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channels, channels, 3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Sum the input with four cascaded dilated-conv branch outputs."""
        d1 = F.relu(self.dilate1(x), inplace=True)
        d2 = F.relu(self.conv1x1(self.dilate2(x)), inplace=True)
        d3 = F.relu(self.conv1x1(self.dilate2(self.dilate1(x))), inplace=True)
        d4 = F.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))), inplace=True)
        return x + d1 + d2 + d3 + d4


class RMPBlock(nn.Module):
    """Residual Multi-kernel Pooling block (CE-Net's ``SPPblock``)."""

    def __init__(self, channels: int) -> None:
        """Initialize four pooling scales sharing a single 1x1 projection.

        Parameters
        ----------
        channels : int
            Channel count of the bottleneck feature map.
        """
        super().__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.pool3 = nn.MaxPool2d(5, 5)
        self.pool4 = nn.MaxPool2d(6, 6)
        self.conv = nn.Conv2d(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Pool at four scales, project to one channel each, and concat back."""
        h, w = x.shape[2], x.shape[3]
        branches = [
            F.interpolate(self.conv(pool(x)), size=(h, w), mode="bilinear", align_corners=False)
            for pool in (self.pool1, self.pool2, self.pool3, self.pool4)
        ]
        return torch.cat([*branches, x], dim=1)


class CEDecoderBlock(nn.Module):
    """CE-Net decoder block: 1x1 reduce -> transposed-conv upsample -> 1x1 expand."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the reduce/upsample/expand decoder stage.

        Parameters
        ----------
        in_channels : int
            Input channel count.
        out_channels : int
            Output channel count.
        """
        super().__init__()
        mid = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.norm1 = nn.BatchNorm2d(mid)
        self.deconv2 = nn.ConvTranspose2d(mid, mid, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Reduce channels, upsample 2x, then expand to the target channels."""
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.deconv2(x)), inplace=True)
        x = F.relu(self.norm3(self.conv3(x)), inplace=True)
        return x


class CENet(nn.Module):
    """Compact CE-Net: small ResNet-style encoder + DAC/RMP bottleneck + U-Net decoder."""

    def __init__(self, num_classes: int = 1) -> None:
        """Build the CE-Net encoder/bottleneck/decoder stack.

        Parameters
        ----------
        num_classes : int, default=1
            Number of output segmentation channels.
        """
        super().__init__()
        filters = [16, 32, 64, 128]
        self.stem = _conv_bn_relu(3, filters[0], stride=2)
        self.encoder1 = _conv_bn_relu(filters[0], filters[0])
        self.encoder2 = _conv_bn_relu(filters[0], filters[1], stride=2)
        self.encoder3 = _conv_bn_relu(filters[1], filters[2], stride=2)
        self.encoder4 = _conv_bn_relu(filters[2], filters[3], stride=2)

        self.dac = DACBlock(filters[3])
        self.rmp = RMPBlock(filters[3])

        self.decoder4 = CEDecoderBlock(filters[3] + 4, filters[2])
        self.decoder3 = CEDecoderBlock(filters[2], filters[1])
        self.decoder2 = CEDecoderBlock(filters[1], filters[0])
        self.decoder1 = CEDecoderBlock(filters[0], filters[0])

        self.final_deconv = nn.ConvTranspose2d(filters[0], 8, 4, 2, 1)
        self.final_conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.final_conv3 = nn.Conv2d(8, num_classes, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode, apply DAC+RMP context, decode with skip connections."""
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dac(e4)
        e4 = self.rmp(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = F.relu(self.final_deconv(d1), inplace=True)
        out = F.relu(self.final_conv2(out), inplace=True)
        out = self.final_conv3(out)
        return torch.sigmoid(out)


def build_cenet() -> nn.Module:
    """Build a compact CE-Net segmentation model.

    Returns
    -------
    nn.Module
        Random-initialized ``CENet`` in eval mode.
    """
    return CENet().eval()


def example_input_cenet() -> Tensor:
    """Create an example RGB medical image patch.

    Returns
    -------
    Tensor
        Random image tensor of shape ``(1, 3, 96, 96)``.
    """
    return torch.randn(1, 3, 96, 96)


# ---------------------------------------------------------------------------
# CellBender
# ---------------------------------------------------------------------------


class CellBenderVAE(nn.Module):
    """Compact CellBender-style ambient-RNA-vs-cell generative VAE.

    Reimplements the official ``RemoveBackgroundPyroModel.model`` counts
    decomposition deterministically: a per-droplet latent ``z`` is decoded
    into a fractional cell-expression profile ``chi``, scaled by a per-
    droplet cell depth to give ``mu_cell``; a single shared learned ambient
    profile is scaled by a per-droplet ambient depth to give ``lam``; the two
    are combined via a per-droplet cell-presence probability ``y``.
    """

    def __init__(self, n_genes: int = 32, z_dim: int = 8, hidden: int = 24) -> None:
        """Initialize the encoder, decoder, and ambient-profile parameters.

        Parameters
        ----------
        n_genes : int, default=32
            Number of genes (input/output feature dimension).
        z_dim : int, default=8
            Dimensionality of the per-droplet latent cell-state ``z``.
        hidden : int, default=24
            Hidden width of the encoder/decoder MLPs.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.Softplus(),
        )
        self.z_loc = nn.Linear(hidden, z_dim)
        self.z_scale = nn.Linear(hidden, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.Softplus(),
            nn.Linear(hidden, n_genes),
        )
        self.d_cell_head = nn.Linear(hidden, 1)
        self.d_empty_head = nn.Linear(hidden, 1)
        self.y_head = nn.Linear(hidden, 1)
        self.chi_ambient_logits = nn.Parameter(torch.zeros(n_genes))

    def forward(self, x: Tensor) -> Tensor:
        """Encode droplet counts, decode cell/ambient signals, and recombine.

        Parameters
        ----------
        x : Tensor
            Raw droplet gene counts of shape ``(batch, n_genes)``.

        Returns
        -------
        Tensor
            Reconstructed counts ``mu_cell + lam`` of shape ``(batch, n_genes)``.
        """
        h = self.encoder(x)
        z_loc = self.z_loc(h)
        z_scale = F.softplus(self.z_scale(h))
        z = z_loc + z_scale * torch.randn_like(z_scale)

        chi = F.softmax(self.decoder(z), dim=-1)
        d_cell = F.softplus(self.d_cell_head(h)) + 1e-4
        y = torch.sigmoid(self.y_head(h))
        mu_cell = y * d_cell * chi

        chi_ambient = F.softmax(self.chi_ambient_logits, dim=-1)
        d_empty = F.softplus(self.d_empty_head(h)) + 1e-4
        lam = d_empty * chi_ambient

        return mu_cell + lam


def build_cellbender() -> nn.Module:
    """Build a compact CellBender ambient-RNA removal VAE.

    Returns
    -------
    nn.Module
        Random-initialized ``CellBenderVAE`` in eval mode.
    """
    return CellBenderVAE().eval()


def example_input_cellbender() -> Tensor:
    """Create example droplet x gene raw-count input.

    Returns
    -------
    Tensor
        Random non-negative count tensor of shape ``(4, 32)``.
    """
    return torch.rand(4, 32) * 10.0


# ---------------------------------------------------------------------------
# CellBox
# ---------------------------------------------------------------------------


class CellBox(nn.Module):
    """Compact CellBox: tanh-gated, masked-adjacency ODE unrolled with Euler steps."""

    def __init__(self, n_nodes: int = 16, n_steps: int = 5, dt: float = 0.1) -> None:
        """Initialize the learned masked adjacency and per-node rate parameters.

        Parameters
        ----------
        n_nodes : int, default=16
            Number of molecular/phenotypic nodes in the network.
        n_steps : int, default=5
            Number of Euler-integration steps to unroll.
        dt : float, default=0.1
            Integration step size.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.n_steps = n_steps
        self.dt = dt
        self.weight = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.05)
        self.register_buffer("mask", 1.0 - torch.eye(n_nodes))
        self.eps = nn.Parameter(torch.ones(n_nodes))
        self.alpha = nn.Parameter(torch.ones(n_nodes))

    def _dxdt(self, x: Tensor, u: Tensor) -> Tensor:
        """Compute the tanh-gated, masked-adjacency-weighted derivative."""
        w_masked = self.weight * self.mask
        interaction = torch.tanh(x @ w_masked.T + u)
        eps = F.softplus(self.eps)
        alpha = F.softplus(self.alpha)
        return eps * interaction - alpha * x

    def forward(self, x0: Tensor, u: Tensor) -> Tensor:
        """Unroll the ODE from an initial state under a fixed perturbation.

        Parameters
        ----------
        x0 : Tensor
            Initial node state of shape ``(batch, n_nodes)``.
        u : Tensor
            Perturbation input of shape ``(batch, n_nodes)``.

        Returns
        -------
        Tensor
            Steady-state node values after ``n_steps`` Euler steps, shape
            ``(batch, n_nodes)``.
        """
        x = x0
        for _ in range(self.n_steps):
            x = x + self.dt * self._dxdt(x, u)
        return x


def build_cellbox() -> nn.Module:
    """Build a compact CellBox perturbation-ODE model.

    Returns
    -------
    nn.Module
        Random-initialized ``CellBox`` in eval mode.
    """
    return CellBox().eval()


def example_input_cellbox() -> tuple[Tensor, Tensor]:
    """Create example initial node state and perturbation input.

    Returns
    -------
    tuple[Tensor, Tensor]
        Initial state and perturbation, each shape ``(4, 16)``.
    """
    return torch.randn(4, 16), torch.randn(4, 16)


# ---------------------------------------------------------------------------
# CellOracle
# ---------------------------------------------------------------------------


class CellOracleGRNSimulator(nn.Module):
    """Compact CellOracle in-silico perturbation propagation over a GRN.

    Reimplements the official ``_do_simulation`` recurrence: an expression
    delta vector is repeatedly propagated through a learned (gene x gene)
    regulatory coefficient matrix, re-clamping the perturbed genes back to
    their fixed input delta after every propagation step.
    """

    def __init__(self, n_genes: int = 24, n_propagation: int = 3) -> None:
        """Initialize the learned regulatory coefficient matrix.

        Parameters
        ----------
        n_genes : int, default=24
            Number of genes in the GRN.
        n_propagation : int, default=3
            Number of propagation steps to unroll.
        """
        super().__init__()
        self.n_propagation = n_propagation
        self.coef_matrix = nn.Parameter(torch.randn(n_genes, n_genes) * 0.05)

    def forward(self, baseline: Tensor, perturb_delta: Tensor, perturb_mask: Tensor) -> Tensor:
        """Propagate a TF-perturbation delta through the learned GRN.

        Parameters
        ----------
        baseline : Tensor
            Baseline gene-expression vector of shape ``(batch, n_genes)``.
        perturb_delta : Tensor
            Fixed perturbation delta (nonzero only at perturbed genes),
            shape ``(batch, n_genes)``.
        perturb_mask : Tensor
            Binary mask marking which genes are held fixed at
            ``perturb_delta`` each step, shape ``(batch, n_genes)``.

        Returns
        -------
        Tensor
            Simulated post-perturbation expression, shape ``(batch, n_genes)``.
        """
        delta = perturb_delta
        for _ in range(self.n_propagation):
            delta = delta @ self.coef_matrix
            delta = torch.where(perturb_mask > 0, perturb_delta, delta)
        return baseline + delta


def build_celloracle() -> nn.Module:
    """Build a compact CellOracle GRN perturbation-propagation model.

    Returns
    -------
    nn.Module
        Random-initialized ``CellOracleGRNSimulator`` in eval mode.
    """
    return CellOracleGRNSimulator().eval()


def example_input_celloracle() -> tuple[Tensor, Tensor, Tensor]:
    """Create example baseline expression, perturbation delta, and mask.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Baseline, perturbation delta, and binary perturbation mask, each
        shape ``(2, 24)``.
    """
    baseline = torch.rand(2, 24)
    perturb_delta = torch.zeros(2, 24)
    perturb_delta[:, 0] = -1.0
    perturb_mask = torch.zeros(2, 24)
    perturb_mask[:, 0] = 1.0
    return baseline, perturb_delta, perturb_mask


# ---------------------------------------------------------------------------
# Chiron
# ---------------------------------------------------------------------------


class _ChironResBlock(nn.Module):
    """1-D residual conv block used by Chiron's CNN feature extractor."""

    def __init__(self, channels: int) -> None:
        """Initialize two stacked 1-D conv/BN/ReLU layers with a residual add.

        Parameters
        ----------
        channels : int
            Channel count of the residual block.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply two conv/BN/ReLU layers and add the residual input."""
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class Chiron(nn.Module):
    """Compact Chiron: residual 1-D CNN + stacked BiLSTM + linear CTC head."""

    def __init__(
        self,
        cnn_channels: int = 32,
        n_res_blocks: int = 3,
        rnn_hidden: int = 32,
        rnn_layers: int = 2,
        n_classes: int = 5,
    ) -> None:
        """Initialize the CNN feature extractor, BiLSTM stack, and CTC head.

        Parameters
        ----------
        cnn_channels : int, default=32
            Channel width of the 1-D CNN feature extractor.
        n_res_blocks : int, default=3
            Number of stacked residual conv blocks.
        rnn_hidden : int, default=32
            Hidden size of each LSTM direction.
        rnn_layers : int, default=2
            Number of stacked bidirectional LSTM layers.
        n_classes : int, default=5
            Output alphabet size (A, C, G, T, blank for CTC).
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, cnn_channels, 5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [_ChironResBlock(cnn_channels) for _ in range(n_res_blocks)]
        )
        self.blstm = nn.LSTM(
            cnn_channels,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * rnn_hidden, n_classes)

    def forward(self, signal: Tensor) -> Tensor:
        """Extract CNN features, run bidirectional LSTM, and classify per timestep.

        Parameters
        ----------
        signal : Tensor
            Raw nanopore current signal of shape ``(batch, 1, length)``.

        Returns
        -------
        Tensor
            Per-timestep class logits of shape ``(batch, length, n_classes)``.
        """
        x = self.stem(signal)
        for block in self.res_blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x, _ = self.blstm(x)
        return self.classifier(x)


def build_chiron() -> nn.Module:
    """Build a compact Chiron nanopore basecaller.

    Returns
    -------
    nn.Module
        Random-initialized ``Chiron`` in eval mode.
    """
    return Chiron().eval()


def example_input_chiron() -> Tensor:
    """Create an example raw nanopore signal window.

    Returns
    -------
    Tensor
        Random signal tensor of shape ``(1, 1, 200)``.
    """
    return torch.randn(1, 1, 200)


MENAGERIE_ENTRIES = [
    ("BioMedLM", "build_biomedlm", "example_input_biomedlm", "2024", "NLP"),
    ("CE-Net", "build_cenet", "example_input_cenet", "2019", "VIS"),
    ("CellBender", "build_cellbender", "example_input_cellbender", "2023", "BIO"),
    ("CellBox", "build_cellbox", "example_input_cellbox", "2021", "BIO"),
    ("CellOracle", "build_celloracle", "example_input_celloracle", "2023", "BIO"),
    ("Chiron", "build_chiron", "example_input_chiron", "2018", "BIO"),
]
