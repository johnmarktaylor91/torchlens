"""Single-cell / genomics classics: scPRINT, scScope, scVAE, Sei, Selene (DanQ), SpaIM.

Sources checked (repo READMEs / architecture source files, no clone/install):
  - scPRINT: https://github.com/cantinilab/scPRINT
    (scprint/model/model.py, encoders.py, decoders.py -- gene-token bidirectional
    transformer with a continuous expression-value encoder and a
    zero-inflated-negative-binomial expression decoder head). Nature Communications
    scale foundation model; the fixed-vocabulary d_model=256/nlayers=8/flash-attention
    training config is not reproduced here -- this is a small, faithful
    reimplementation of the token construction + encoder + ZINB decoder mechanism.
  - scScope: https://github.com/AltschulerWu-Lab/scScope
    (scscope/scscope/large_scale_processing.py:``Inference`` -- TensorFlow 1.x).
    Distinctive mechanism: a batch-effect-removal linear layer subtracted from the
    input, followed by a depth-``T`` recurrent self-consistent autoencoder where each
    recurrence step after the first computes an "imputation" correction (an MLP over
    the previous reconstruction) applied only at the originally-zero input entries,
    then re-encodes; outputs of all ``T`` steps are the ``T`` decoder reconstructions.
  - scVAE: https://github.com/scvae/scvae
    (scvae/models/variational_autoencoder.py -- TensorFlow). Distinctive mechanism:
    a standard Gaussian-latent VAE whose reconstruction head parameterizes a
    zero-inflated negative binomial (count) likelihood rather than
    Gaussian/Bernoulli pixels -- mean, dispersion, and zero-inflation-logit heads.
  - Sei: https://github.com/FunctionLab/sei-framework (model/sei.py, PyTorch,
    reproduced near-verbatim at reduced width/depth/sequence length). Distinctive
    mechanism: a residual dilated-conv tower (3 pooled stages + 5 stacked dilated
    residual conv blocks) followed by a fixed (non-trainable) B-spline temporal
    basis-function pooling layer before the final classifier.
  - Selene / DanQ: https://github.com/FunctionLab/selene (models/danQ.py, PyTorch).
    Distinctive mechanism (representative Selene model-zoo entry, distinct from the
    already-catalogued DeepSEA): one-hot DNA conv+ReLU+maxpool front end feeding a
    bidirectional LSTM over the pooled positions, then an MLP classifier.
  - SpaIM: https://github.com/QSong-github/SpaIM (src/model.py, PyTorch). Distinctive
    mechanism ("ReST" -- REconstruction via Style Transfer): separate content and
    style encoders for single-cell (sc) and spatial-transcriptomics (st) expression
    profiles; the decoder reconstructs an expression profile by multiplicatively
    modulating a content code with a style code (AdaIN-like feature modulation) at
    two decoder depths, enabling sc-content + st-style cross-domain imputation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# scPRINT
# ---------------------------------------------------------------------------


class GeneTokenEncoder(nn.Module):
    """Embed integer gene ids into ``d_model``, then LayerNorm (scPRINT GeneEncoder)."""

    def __init__(self, num_genes: int, d_model: int) -> None:
        """Initialize the gene-id embedding table.

        Parameters
        ----------
        num_genes:
            Size of the gene vocabulary (including the CLS token id 0).
        d_model:
            Embedding / model dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_genes, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_ids: Tensor) -> Tensor:
        """Embed gene ids.

        Parameters
        ----------
        gene_ids:
            Long tensor ``(batch, genes)``.

        Returns
        -------
        Tensor
            ``(batch, genes, d_model)``.
        """
        return self.norm(self.embedding(gene_ids))


class ContinuousExprEncoder(nn.Module):
    """Project scalar expression values into ``d_model`` (scPRINT ContinuousValueEncoder)."""

    def __init__(self, d_model: int) -> None:
        """Initialize the value-projection MLP.

        Parameters
        ----------
        d_model:
            Embedding / model dimension.
        """
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: Tensor) -> Tensor:
        """Encode per-gene expression scalars.

        Parameters
        ----------
        values:
            Float tensor ``(batch, genes)``.

        Returns
        -------
        Tensor
            ``(batch, genes, d_model)``.
        """
        x = F.relu(self.linear1(values.unsqueeze(-1)))
        return self.norm(self.linear2(x))


class ZINBExprDecoder(nn.Module):
    """Zero-inflated negative binomial expression head (scPRINT ExprDecoder)."""

    def __init__(self, d_model: int) -> None:
        """Initialize the ZINB parameter heads.

        Parameters
        ----------
        d_model:
            Model dimension of the transformer output.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
        )
        self.pred_var_zero = nn.Linear(d_model, 3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict ZINB mean / dispersion / zero-inflation logits.

        Parameters
        ----------
        x:
            Transformer output ``(batch, genes, d_model)``.

        Returns
        -------
        tuple of Tensor
            ``(mean, disp, zero_logits)`` each ``(batch, genes)``.
        """
        h = self.fc(x)
        pred, var, zero_logits = self.pred_var_zero(h).split(1, dim=-1)
        mean = F.softmax(pred.squeeze(-1), dim=-1)
        disp = torch.exp(torch.clamp(var.squeeze(-1), max=15.0))
        return mean, disp, zero_logits.squeeze(-1)


class ScPRINT(nn.Module):
    """Compact scPRINT core: gene+value tokens -> bidirectional transformer -> ZINB head."""

    def __init__(
        self,
        num_genes: int = 32,
        d_model: int = 48,
        nhead: int = 4,
        nlayers: int = 2,
    ) -> None:
        """Initialize the transformer core.

        Parameters
        ----------
        num_genes:
            Gene vocabulary size (id 0 is reserved for the CLS token).
        d_model:
            Transformer model dimension.
        nhead:
            Number of self-attention heads.
        nlayers:
            Number of encoder layers.
        """
        super().__init__()
        self.gene_encoder = GeneTokenEncoder(num_genes, d_model)
        self.value_encoder = ContinuousExprEncoder(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.expr_decoder = ZINBExprDecoder(d_model)
        self.cell_emb_norm = nn.LayerNorm(d_model)

    def forward(self, gene_ids: Tensor, values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the bidirectional gene-token encoder and ZINB decoder.

        Parameters
        ----------
        gene_ids:
            Long tensor ``(batch, genes)``; position 0 is the CLS token.
        values:
            Float tensor ``(batch, genes)`` of (depth-normalized) expression counts.

        Returns
        -------
        tuple of Tensor
            ``(cell_embedding, mean, disp, zero_logits)``.
        """
        token = self.gene_encoder(gene_ids) + self.value_encoder(values)
        out = self.transformer(token)
        cell_embedding = self.cell_emb_norm(out[:, 0, :])
        mean, disp, zero_logits = self.expr_decoder(out)
        return cell_embedding, mean, disp, zero_logits


def build_scprint() -> nn.Module:
    """Build a small scPRINT gene-token transformer.

    Returns
    -------
    nn.Module
        Random-initialized :class:`ScPRINT` in eval mode.
    """
    return ScPRINT(num_genes=32, d_model=48, nhead=4, nlayers=2).eval()


def example_input_scprint() -> list[Tensor]:
    """Return an example ``[gene_ids (1,32), values (1,32)]`` pair.

    Returns
    -------
    list of Tensor
        Gene id sequence (CLS at position 0) and matching expression values.
    """
    g = 32
    gene_ids = torch.randint(1, 32, (1, g))
    gene_ids[:, 0] = 0
    values = torch.rand(1, g) * 5.0
    values[:, 0] = 0.0
    return [gene_ids, values]


# ---------------------------------------------------------------------------
# scScope
# ---------------------------------------------------------------------------


class ScScope(nn.Module):
    """scScope: batch-effect removal + T-step recurrent self-consistent autoencoder.

    Reimplements ``Inference`` from ``large_scale_processing.py``: an experimental
    batch-effect linear correction is subtracted from the input once; a shared
    encoder/decoder pair is then applied ``T`` times, where every recurrence step
    after the first imputes a correction for the originally-zero input entries
    (via a small MLP over the previous reconstruction) before re-encoding.
    """

    def __init__(
        self, input_dim: int = 40, latent_dim: int = 8, hidden_dim: int = 24, steps: int = 3
    ) -> None:
        """Initialize the recurrent autoencoder.

        Parameters
        ----------
        input_dim:
            Number of genes in the input expression vector.
        latent_dim:
            Dimension of the scScope latent feature ("h_c" in the paper).
        hidden_dim:
            Width of the single encoder/decoder hidden layer.
        steps:
            Recurrence depth ``T``.
        """
        super().__init__()
        self.steps = steps
        self.batch_effect = nn.Linear(1, input_dim, bias=False)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.to_recon = nn.Linear(hidden_dim, input_dim)
        self.impute1 = nn.Linear(input_dim, 64)
        self.impute2 = nn.Linear(64, input_dim)

    def forward(self, x: Tensor, exp_batch_idx: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """Run the ``T``-step recurrent self-consistent encode/decode loop.

        Parameters
        ----------
        x:
            Expression matrix ``(batch, genes)``.
        exp_batch_idx:
            One-hot experimental-batch indicator ``(batch, 1)`` (a single batch
            group is used for tracing simplicity).

        Returns
        -------
        tuple of (list of Tensor, list of Tensor)
            ``(output_list, latent_code_list)``, one reconstruction and one latent
            code per recurrence step.
        """
        batch_effect_removal = self.batch_effect(exp_batch_idx)
        nonzero_mask = (x != 0).float()

        latent_list: list[Tensor] = []
        output_list: list[Tensor] = []
        recon = None
        for t in range(self.steps):
            if t == 0:
                input_vec = F.relu(x - batch_effect_removal)
            else:
                intermediate = F.relu(self.impute1(recon))
                imputation = (1.0 - nonzero_mask) * self.impute2(intermediate)
                input_vec = F.relu(imputation + x - batch_effect_removal)
            h = self.encoder(input_vec)
            latent = F.relu(self.to_latent(h))
            d = self.decoder(latent)
            recon = F.relu(self.to_recon(d))
            latent_list.append(latent)
            output_list.append(recon)
        return output_list, latent_list


def build_scscope() -> nn.Module:
    """Build a small scScope recurrent self-consistent autoencoder.

    Returns
    -------
    nn.Module
        Random-initialized :class:`ScScope` in eval mode.
    """
    return ScScope(input_dim=40, latent_dim=8, hidden_dim=24, steps=3).eval()


def example_input_scscope() -> list[Tensor]:
    """Return an example ``[expression (1,40), batch_idx (1,1)]`` pair.

    Returns
    -------
    list of Tensor
        Sparse (many-zero) expression vector and a single-column experimental
        batch indicator.
    """
    x = torch.rand(1, 40)
    x = x * (torch.rand(1, 40) > 0.4).float()  # sparsify like real scRNA-seq counts
    batch_idx = torch.zeros(1, 1)
    return [x, batch_idx]


# ---------------------------------------------------------------------------
# scVAE
# ---------------------------------------------------------------------------


class ScVAE(nn.Module):
    """scVAE: Gaussian-latent VAE with a zero-inflated-negative-binomial count head.

    Reimplements the MLP inference/generative networks of
    ``variational_autoencoder.py`` with the ZINB reconstruction distribution
    (mean, dispersion, zero-inflation logit) used for count (scRNA-seq) data.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, latent_dim: int = 10) -> None:
        """Initialize encoder/decoder MLPs and the ZINB reconstruction head.

        Parameters
        ----------
        input_dim:
            Number of genes.
        hidden_dim:
            Width of the single hidden layer in encoder and decoder.
        latent_dim:
            Dimension of the Gaussian latent space.
        """
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.to_mean = nn.Linear(hidden_dim, input_dim)
        self.to_disp = nn.Linear(hidden_dim, input_dim)
        self.to_zero_logits = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode to a Gaussian posterior, sample, and decode ZINB parameters.

        Parameters
        ----------
        x:
            Count matrix ``(batch, genes)``.

        Returns
        -------
        tuple of Tensor
            ``(mean, disp, zero_logits, mu, logvar)`` of the reconstruction
            distribution and the approximate posterior.
        """
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        d = self.decoder(z)
        mean = F.softplus(self.to_mean(d))
        disp = F.softplus(self.to_disp(d))
        zero_logits = self.to_zero_logits(d)
        return mean, disp, zero_logits, mu, logvar


def build_scvae() -> nn.Module:
    """Build a small scVAE count-distribution VAE.

    Returns
    -------
    nn.Module
        Random-initialized :class:`ScVAE` in eval mode.
    """
    return ScVAE(input_dim=64, hidden_dim=32, latent_dim=10).eval()


def example_input_scvae() -> Tensor:
    """Return an example scRNA-seq count-like input.

    Returns
    -------
    Tensor
        Non-negative float tensor ``(1, 64)``.
    """
    return torch.poisson(torch.rand(1, 64) * 4.0)


# ---------------------------------------------------------------------------
# Sei
# ---------------------------------------------------------------------------


def _bspline_basis(n_positions: int, n_basis: int) -> Tensor:
    """Build a fixed (non-trainable) smooth basis approximating Sei's B-spline pooling.

    Parameters
    ----------
    n_positions:
        Number of positions along the sequence axis to pool.
    n_basis:
        Number of output basis functions.

    Returns
    -------
    Tensor
        Weight matrix ``(n_positions, n_basis)`` of raised-cosine bump functions,
        each summing (numerically) to a smooth partition of the position axis --
        capturing the "soft local pooling into a small fixed basis" role that the
        B-spline transform plays in Sei, without depending on scipy.
    """
    centers = torch.linspace(0, n_positions - 1, n_basis)
    width = max(n_positions / n_basis, 1.0)
    positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    dist = (positions - centers.unsqueeze(0)) / width
    basis = torch.clamp(1.0 - dist.abs(), min=0.0)
    basis = basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return basis


class SeiResidualConvBlock(nn.Module):
    """One lconv+conv residual stage of the Sei tower."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool) -> None:
        """Initialize a pooled (or non-pooled) residual convolution stage.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        pool:
            Whether to max-pool by 4 before the two long convolutions.
        """
        super().__init__()
        pool_layers: list[nn.Module] = []
        if pool:
            pool_layers.append(nn.MaxPool1d(kernel_size=4, stride=4))
        self.lconv = nn.Sequential(
            *pool_layers,
            nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=4),
            nn.Conv1d(out_ch, out_ch, kernel_size=9, padding=4),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the pooled long-conv branch and its residual conv refinement.

        Parameters
        ----------
        x:
            Input feature map ``(batch, in_ch, length)``.

        Returns
        -------
        tuple of Tensor
            ``(lout, out)`` where ``out = conv(lout)`` and ``lout`` is kept for
            the next stage's residual sum.
        """
        lout = self.lconv(x)
        out = self.conv(lout)
        return lout, out


class SeiDilatedBlock(nn.Module):
    """One dilated residual convolution used in Sei's five-block dilation tower."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize a dilated residual convolution.

        Parameters
        ----------
        channels:
            Feature channel count (constant through the dilation tower).
        dilation:
            Dilation factor of the 1-D convolution.
        """
        super().__init__()
        padding = 2 * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, dilation=dilation, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the dilated convolution.

        Parameters
        ----------
        x:
            Input feature map ``(batch, channels, length)``.

        Returns
        -------
        Tensor
            Same-length filtered feature map.
        """
        return self.conv(x)


class Sei(nn.Module):
    """Sei: residual dilated-conv tower + B-spline pooling -> chromatin-profile head.

    Compact, reduced-width/length reimplementation of ``model/sei.py`` (three
    pooled residual conv stages, five stacked dilated residual conv blocks,
    fixed smooth-basis temporal pooling, sigmoid multi-label classifier).
    """

    def __init__(
        self, sequence_length: int = 512, n_features: int = 24, spline_df: int = 4
    ) -> None:
        """Initialize the convolutional tower and classifier head.

        Parameters
        ----------
        sequence_length:
            Length of the one-hot input DNA sequence.
        n_features:
            Number of chromatin-profile outputs.
        spline_df:
            Number of B-spline-style pooling basis functions.
        """
        super().__init__()
        self.stage1 = SeiResidualConvBlock(4, 32, pool=False)
        self.stage2 = SeiResidualConvBlock(32, 48, pool=True)
        self.stage3 = SeiResidualConvBlock(48, 64, pool=True)

        dilations = [2, 4, 8, 16, 25]
        self.dilated_blocks = nn.ModuleList([SeiDilatedBlock(64, d) for d in dilations])

        pooled_length = sequence_length // 16
        self.register_buffer("spline_basis", _bspline_basis(pooled_length, spline_df))
        self.spline_df = spline_df

        self.classifier = nn.Sequential(
            nn.Linear(64 * spline_df, n_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict multi-label chromatin-profile probabilities.

        Parameters
        ----------
        x:
            One-hot DNA sequence ``(batch, 4, sequence_length)``.

        Returns
        -------
        Tensor
            Sigmoid probabilities ``(batch, n_features)``.
        """
        lout1, out1 = self.stage1(x)
        lout2, out2 = self.stage2(out1 + lout1)
        lout3, out3 = self.stage3(out2 + lout2)

        cat_out = out3 + lout3
        for block in self.dilated_blocks:
            cat_out = cat_out + block(cat_out)

        # fixed smooth-basis temporal pooling (stand-in for Sei's BSplineTransformation)
        pooled = torch.matmul(cat_out, self.spline_basis)  # (batch, channels, spline_df)
        flat = pooled.reshape(pooled.size(0), -1)
        return self.classifier(flat)


def build_sei() -> nn.Module:
    """Build a small Sei chromatin-profile predictor.

    Returns
    -------
    nn.Module
        Random-initialized :class:`Sei` in eval mode.
    """
    return Sei(sequence_length=512, n_features=24, spline_df=4).eval()


def example_input_sei() -> Tensor:
    """Return an example one-hot DNA sequence.

    Returns
    -------
    Tensor
        Float tensor ``(1, 4, 512)``.
    """
    ids = torch.randint(0, 4, (1, 512), dtype=torch.long)
    return F.one_hot(ids, num_classes=4).float().transpose(1, 2)


# ---------------------------------------------------------------------------
# Selene model zoo: DanQ
# ---------------------------------------------------------------------------


class DanQ(nn.Module):
    """DanQ: conv+maxpool front end feeding a bidirectional LSTM (Selene model zoo).

    Reimplements ``models/danQ.py`` (Quang & Xie, 2016): a single wide 1-D
    convolution + ReLU + max-pool over one-hot DNA, followed by a bidirectional
    LSTM over the pooled positions and an MLP classifier.
    """

    def __init__(self, sequence_length: int = 200, n_features: int = 20) -> None:
        """Initialize the conv front end, BiLSTM, and classifier.

        Parameters
        ----------
        sequence_length:
            Length of the one-hot input DNA sequence.
        n_features:
            Number of genomic-feature outputs.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=13),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=6, stride=6),
        )
        self.bilstm = nn.LSTM(32, 32, num_layers=1, batch_first=True, bidirectional=True)
        pooled_length = (sequence_length - 12) // 6
        self.classifier = nn.Sequential(
            nn.Linear(pooled_length * 64, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict multi-label genomic-feature probabilities.

        Parameters
        ----------
        x:
            One-hot DNA sequence ``(batch, 4, sequence_length)``.

        Returns
        -------
        Tensor
            Sigmoid probabilities ``(batch, n_features)``.
        """
        out = self.conv(x)  # (batch, channels, pooled_length)
        out = out.transpose(1, 2)  # (batch, pooled_length, channels)
        out, _ = self.bilstm(out)  # (batch, pooled_length, 2*channels)
        flat = out.reshape(out.size(0), -1)
        return self.classifier(flat)


def build_danq() -> nn.Module:
    """Build a small DanQ conv+BiLSTM genomic-feature predictor.

    Returns
    -------
    nn.Module
        Random-initialized :class:`DanQ` in eval mode.
    """
    return DanQ(sequence_length=200, n_features=20).eval()


def example_input_danq() -> Tensor:
    """Return an example one-hot DNA sequence.

    Returns
    -------
    Tensor
        Float tensor ``(1, 4, 200)``.
    """
    ids = torch.randint(0, 4, (1, 200), dtype=torch.long)
    return F.one_hot(ids, num_classes=4).float().transpose(1, 2)


# ---------------------------------------------------------------------------
# SpaIM
# ---------------------------------------------------------------------------


class SpaIMMLP(nn.Module):
    """Linear -> (LayerNorm -> ReLU) block used throughout SpaIM's ``mlp_simple``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the linear projection and normalization.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        out_dim:
            Output feature dimension.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, use_norm: bool = True) -> Tensor:
        """Project (and optionally normalize + activate) the input.

        Parameters
        ----------
        x:
            Input tensor ``(batch, in_dim)``.
        use_norm:
            Whether to apply LayerNorm + ReLU after the linear projection
            (SpaIM disables this on the final decoder layer).

        Returns
        -------
        Tensor
            ``(batch, out_dim)``.
        """
        x = self.linear(x)
        if use_norm:
            x = F.relu(self.norm(x))
        return x


class SpaIM(nn.Module):
    """SpaIM / ReST: content-style disentangled cross-domain expression imputation.

    Reimplements ``src/model.py``'s ``Imputation`` module: separate content and
    style encoders for single-cell (sc) and spatial (st) expression, a shared
    decoder that reconstructs an expression profile from ``content * style``
    (feature modulation) at two depths, and cross-domain generation of a
    "fake" spatial profile from sc-content + st-style.
    """

    def __init__(
        self, sc_dim: int = 48, st_dim: int = 48, style_dim: int = 16, h1: int = 32, h2: int = 16
    ) -> None:
        """Initialize the content/style encoders and shared decoder.

        Parameters
        ----------
        sc_dim:
            Number of genes in the single-cell expression profile.
        st_dim:
            Number of genes in the spatial-transcriptomics expression profile.
        style_dim:
            Dimension of the style-code input.
        h1:
            Width of the first (shallower) hidden representation.
        h2:
            Width of the second (deeper) hidden representation.
        """
        super().__init__()
        self.st_enc1_cont = SpaIMMLP(st_dim, h1)
        self.st_enc2_cont = SpaIMMLP(h1, h2)
        self.st_enc1_style = SpaIMMLP(st_dim, h1)
        self.st_enc2_style = SpaIMMLP(h1, h2)

        self.st_dec2 = SpaIMMLP(h2, h1)
        self.st_dec1 = SpaIMMLP(h1, st_dim)

        self.sc_enc2_cont = SpaIMMLP(sc_dim, h2)
        self.sc_enc1_cont = SpaIMMLP(h2, h1)

        self.enc_style2 = SpaIMMLP(style_dim, h2)
        self.enc_style1 = SpaIMMLP(style_dim, h1)

    def forward(self, sc: Tensor, st: Tensor, st_style: Tensor) -> tuple[Tensor, Tensor]:
        """Reconstruct the real ST profile and a cross-domain sc-to-ST profile.

        Parameters
        ----------
        sc:
            Single-cell expression profile ``(batch, sc_dim)``.
        st:
            Spatial-transcriptomics expression profile ``(batch, st_dim)``.
        st_style:
            Style code describing the target spatial domain ``(batch, style_dim)``.

        Returns
        -------
        tuple of Tensor
            ``(st_real, st_fake)``, the ST self-reconstruction and the
            sc-content/st-style cross-domain reconstruction, each
            ``(batch, st_dim)``.
        """
        st_cont1 = self.st_enc1_cont(st)
        st_cont2 = self.st_enc2_cont(st_cont1)

        st_style1 = self.st_enc1_style(st)
        st_style2 = self.st_enc2_style(st_style1)

        sc_cont2 = self.sc_enc2_cont(sc)
        sc_cont1 = self.sc_enc1_cont(sc_cont2)

        fake_style2 = self.enc_style2(st_style)
        fake_style1 = self.enc_style1(st_style)

        real_up2 = self.st_dec2(st_cont2 * st_style2)
        real_up1 = self.st_dec1(real_up2 + st_cont1 * st_style1, use_norm=False)

        fake_up2 = self.st_dec2(sc_cont2 * fake_style2)
        fake_up1 = self.st_dec1(fake_up2 + sc_cont1 * fake_style1, use_norm=False)

        return real_up1, fake_up1


def build_spaim() -> nn.Module:
    """Build a small SpaIM content/style cross-domain imputation model.

    Returns
    -------
    nn.Module
        Random-initialized :class:`SpaIM` in eval mode.
    """
    return SpaIM(sc_dim=48, st_dim=48, style_dim=16, h1=32, h2=16).eval()


def example_input_spaim() -> list[Tensor]:
    """Return an example ``[sc_expr (1,48), st_expr (1,48), st_style (1,16)]`` triple.

    Returns
    -------
    list of Tensor
        Single-cell expression profile, spatial expression profile, and a
        style-code vector for the spatial domain.
    """
    sc = torch.rand(1, 48)
    st = torch.rand(1, 48)
    st_style = torch.randn(1, 16)
    return [sc, st, st_style]


MENAGERIE_ENTRIES = [
    ("scPRINT", "build_scprint", "example_input_scprint", "2025", "BIO"),
    ("scScope", "build_scscope", "example_input_scscope", "2019", "BIO"),
    ("scVAE", "build_scvae", "example_input_scvae", "2020", "BIO"),
    ("Sei", "build_sei", "example_input_sei", "2022", "BIO"),
    ("Selene/Selene model zoo CNNs", "build_danq", "example_input_danq", "2019", "BIO"),
    ("SpaIM", "build_spaim", "example_input_spaim", "2025", "BIO"),
]
