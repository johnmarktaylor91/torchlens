# SOURCE: vendored from tky823/DNN-based_source_separation @ main
# Files: src/models/wavesplit.py, src/utils/tasnet.py (choose_layer_norm),
#        src/models/film.py (FiLM1d), src/modules/norm.py (GlobalLayerNorm,
#        CumulativeLayerNorm1d). Only import paths were adapted (the original
#        repo uses a top-level `src/` on sys.path); no architectural code was
#        changed.
"""WaveSplit speaker-conditioned time-domain source separation (vendored).

WaveSplit estimates per-speaker embeddings with a `SpeakerStack`, clusters
them (differentiable k-means over torch ops, no sklearn), and conditions a
FiLM-modulated `SeparationStack` on the resulting speaker centroids to
separate a single-channel mixture waveform into per-source waveforms.
"""

from __future__ import annotations

import itertools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

EPS = 1e-12


# ---------------------------------------------------------------------------
# from src/modules/norm.py
# ---------------------------------------------------------------------------
class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input):
        output = self.norm(input)
        return output


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, input):
        """
        Args:
            input (batch_size, C, T)
        """
        eps = self.eps

        batch_size, C, T = input.size()

        step_sum = input.sum(dim=1)  # (batch_size, T)
        step_squared_sum = torch.pow(input, 2).sum(dim=1)  # (batch_size, T)
        cum_sum = torch.cumsum(step_sum, dim=1)  # (batch_size, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1)  # (batch_size, T)

        cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float)  # (T, ): (C, 2C, 3C, ...)
        cum_num = cum_num.to(input.device)
        cum_mean = cum_sum / cum_num  # (batch_size, T)
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean**2

        cum_mean, cum_var = cum_mean.unsqueeze(dim=1), cum_var.unsqueeze(dim=1)

        output = (input - cum_mean) / (torch.sqrt(cum_var + eps)) * self.gamma + self.beta

        return output


# ---------------------------------------------------------------------------
# from src/utils/tasnet.py
# ---------------------------------------------------------------------------
def choose_layer_norm(name, num_features, causal=False, eps=EPS, **kwargs):
    if name == "cLN":
        layer_norm = CumulativeLayerNorm1d(num_features, eps=eps)
    elif name == "gLN":
        if causal:
            raise ValueError("Global Layer Normalization is NOT causal.")
        layer_norm = GlobalLayerNorm(num_features, eps=eps)
    elif name in ["BN", "batch", "batch_norm"]:
        n_dims = kwargs.get("n_dims") or 1
        if n_dims == 1:
            layer_norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            layer_norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError("n_dims is expected 1 or 2, but give {}.".format(n_dims))
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))

    return layer_norm


# ---------------------------------------------------------------------------
# from src/models/film.py
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, gamma, beta):
        n_dims = input.dim()
        expand_dims = (1,) * (n_dims - 2)
        dims = gamma.size() + expand_dims

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta


class FiLM1d(FiLM):
    def __init__(self):
        super().__init__()

    def forward(self, input, gamma, beta):
        dims = gamma.size() + (1,)

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta


# ---------------------------------------------------------------------------
# from src/models/wavesplit.py
# ---------------------------------------------------------------------------
class WaveSplitBase(nn.Module):
    def __init__(
        self,
        speaker_stack: nn.Module,
        separation_stack: nn.Module,
        n_sources=2,
        n_training_sources=10,
        spk_criterion=None,
    ):
        super().__init__()

        assert spk_criterion is not None, "Specify spk_criterion."

        self.speaker_stack = speaker_stack
        self.separation_stack = separation_stack

        all_spk_idx = torch.arange(n_training_sources).long()
        self.all_spk_idx = nn.Parameter(all_spk_idx, requires_grad=False)

        self.spk_criterion = spk_criterion

        self.n_sources, self.n_training_sources = n_sources, n_training_sources

    def forward(
        self,
        mixture,
        spk_idx=None,
        sorted_idx=None,
        return_all_layers=False,
        return_spk_vector=False,
        stack_dim=1,
    ):
        """
        Only supports training time
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
        Returns:
            estimated_sources: (batch_size, num_blocks * num_layers, 1, T) if stack_dim=1.
            sorted_spk_vector: (batch_size, n_sources, latent_dim, T)
        """
        if self.training:
            if sorted_idx is None:
                if return_all_layers or return_spk_vector:
                    raise ValueError("Set return_all_layers=False, return_spk_vector=False.")

                sorted_idx = self.solve_permutation(
                    mixture, spk_idx=spk_idx
                )  # (batch_size, T, n_sources)

                return sorted_idx

            # If sorted_idx is given.
            estimated_sources, sorted_spk_vector = self.extract_latent(
                mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim
            )
        else:
            if spk_idx is not None:
                if sorted_idx is None:
                    if return_all_layers or return_spk_vector:
                        raise ValueError("Set return_all_layers=False, return_spk_vector=False.")

                    sorted_idx = self.solve_permutation(
                        mixture, spk_idx=spk_idx
                    )  # (batch_size, T, n_sources)

                    return sorted_idx

                # If sorted_idx is given.
                estimated_sources, sorted_spk_vector = self.extract_latent(
                    mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim
                )
            else:
                spk_vector = self.speaker_stack(mixture)  # (batch_size, n_sources, latent_dim, T)
                sorted_spk_vector = self.apply_kmeans(
                    spk_vector, feature_last=False
                )  # (batch_size, n_sources, latent_dim, T)
                spk_centroids = sorted_spk_vector.mean(
                    dim=-1
                )  # (batch_size, n_sources, latent_dim)
                estimated_sources = self.separation_stack(
                    mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim
                )

        if return_spk_vector:
            return estimated_sources, sorted_spk_vector

        return estimated_sources

    def extract_latent(self, mixture, sorted_idx, return_all_layers=False, stack_dim=1):
        spk_vector = self.speaker_stack(mixture)  # (batch_size, n_sources, latent_dim, T)

        spk_vector = spk_vector.permute(
            0, 3, 1, 2
        ).contiguous()  # (batch_size, T, n_sources, latent_dim)

        mask = torch.eye(self.n_sources)[sorted_idx]  # (batch_size, T, n_sources, n_sources)
        mask = mask.to(spk_vector.device)
        sorted_spk_vector = torch.sum(
            mask.unsqueeze(dim=4) * spk_vector.unsqueeze(dim=3), dim=2
        )  # (batch_size, T, n_sources, latent_dim)
        sorted_spk_vector = sorted_spk_vector.permute(
            0, 2, 3, 1
        ).contiguous()  # (batch_size, n_sources, latent_dim, T)
        spk_centroids = sorted_spk_vector.mean(dim=3)  # (batch_size, n_sources, latent_dim)

        estimated_sources = self.separation_stack(
            mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim
        )

        return estimated_sources, sorted_spk_vector

    def solve_permutation(self, mixture, spk_idx):
        raise NotImplementedError("Implement solve_permutation.")

    def compute_pit_speaker_loss(
        self, spk_vector, spk_embedding, all_spk_embedding, feature_last=True, batch_mean=True
    ):
        assert feature_last, "feature_last should be True."

        patterns = list(itertools.permutations(range(self.n_sources)))
        patterns = torch.Tensor(patterns).long()

        P = len(patterns)
        possible_loss = []

        for idx in range(P):
            pattern = patterns[idx]
            loss = self.spk_criterion(
                spk_vector[:, :, pattern],
                spk_embedding,
                all_spk_embedding,
                feature_last=feature_last,
                batch_mean=False,
                time_mean=False,
            )  # (batch_size, T)
            possible_loss.append(loss)

        possible_loss = torch.stack(possible_loss, dim=2)  # (batch_size, T, P)
        loss, indices = torch.min(
            possible_loss, dim=2
        )  # loss (batch_size, T), indices (batch_size, T)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss, patterns[indices]

    def apply_kmeans(self, spk_vector, feature_last=False, iter_clustering=100):
        """
        Args:
            spk_vector: (batch_size, n_sources, latent_dim, T) or (batch_size, T, n_sources, latent_dim)
        Returns:
            spk_vector: (batch_size, n_clusters, latent_dim, T), where n_clusters = n_sources
        """
        if not feature_last:
            spk_vector = spk_vector.permute(
                0, 3, 1, 2
            ).contiguous()  # (batch_size, T, n_sources, latent_dim)

        n_sources = self.n_sources

        for idx in range(iter_clustering):
            centroids = spk_vector.mean(
                dim=1, keepdim=True
            )  # (batch_size, 1, n_clusters, latent_dim)
            distance = torch.norm(
                spk_vector.unsqueeze(dim=3) - centroids.unsqueeze(dim=2), dim=4
            )  # (batch_size, T, n_sources, n_clusters)
            cluster_idx = torch.argmin(distance, dim=3)  # (batch_size, T, n_sources)
            mask = torch.eye(n_sources)[cluster_idx]  # (batch_size, T, n_sources, n_clusters)
            mask = mask.to(spk_vector.device)
            spk_vector = torch.sum(
                mask.unsqueeze(dim=4) * spk_vector.unsqueeze(dim=3), dim=2
            )  # (batch_size, T, n_clusters, latent_dim)

        if not feature_last:
            spk_vector = spk_vector.permute(
                0, 2, 3, 1
            ).contiguous()  # (batch_size, n_clusters, latent_dim, T)

        return spk_vector

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        for p in self.spk_criterion.parameters():
            if p.requires_grad:
                _num_parameters -= p.numel()

        return _num_parameters

    def get_config(self):
        config = {
            "base": {"n_sources": self.n_sources, "n_training_sources": self.n_training_sources},
            "spk_stack": self.speaker_stack.get_config(),
            "sep_stack": self.separation_stack.get_config(),
        }

        return config


class WaveSplit(WaveSplitBase):
    def __init__(
        self,
        speaker_stack: nn.Module,
        separation_stack: nn.Module,
        latent_dim: int,
        n_sources=2,
        n_training_sources=10,
        spk_criterion=None,
        eps=EPS,
    ):
        super().__init__(
            speaker_stack,
            separation_stack,
            n_sources=n_sources,
            n_training_sources=n_training_sources,
            spk_criterion=spk_criterion,
        )

        warnings.warn("Implementation of WaveSplit has not finished.", UserWarning)

        self.embedding = nn.Embedding(n_training_sources, latent_dim)

        self.eps = eps

    def forward(
        self,
        mixture,
        spk_idx=None,
        sorted_idx=None,
        return_all_layers=False,
        return_spk_vector=False,
        return_spk_embedding=False,
        return_all_spk_embedding=False,
        stack_dim=1,
    ):
        """
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
        Returns:
            sorted_idx: (batch_size, T, n_sources) if sorted_idx is None.
            estimated_sources:
                (batch_size, num_blocks * num_layers, n_sources, T) if return_all_layers=True and stack_dim=1.
                (batch_size, n_sources, T) if return_all_layers=False.
        """
        eps = self.eps

        if self.training:
            if sorted_idx is None:
                if (
                    return_all_layers
                    or return_spk_vector
                    or return_spk_embedding
                    or return_all_spk_embedding
                ):
                    raise ValueError(
                        "Set return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False."
                    )

                sorted_idx = self.solve_permutation(
                    mixture, spk_idx=spk_idx
                )  # (batch_size, T, n_sources)

                return sorted_idx

            estimated_sources, sorted_spk_vector = self.extract_latent(
                mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim
            )

            if return_spk_embedding:
                spk_embedding = self.embedding(spk_idx)  # (batch_size, n_sources, latent_dim)
                spk_embedding = spk_embedding / (
                    torch.linalg.vector_norm(spk_embedding, dim=2, keepdim=True) + eps
                )
        else:
            # Inference / eval-mode forward path used by this vendored recipe:
            # only `mixture` is given (no spk_idx, no sorted_idx). The speaker
            # stack produces per-frame speaker vectors, differentiable k-means
            # (apply_kmeans, pure torch ops) clusters them into per-source
            # centroids, and the FiLM-conditioned separation stack estimates
            # the separated sources. `spk_criterion` (needed only for the
            # training permutation-invariant loss) is never called here.
            if spk_idx is not None:
                if sorted_idx is None:
                    if return_all_layers or return_spk_vector:
                        raise ValueError("Set return_all_layers=False, return_spk_vector=False.")

                    sorted_idx = self.solve_permutation(
                        mixture, spk_idx=spk_idx
                    )  # (batch_size, T, n_sources)

                    return sorted_idx

                estimated_sources, sorted_spk_vector = self.extract_latent(
                    mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim
                )

                if return_spk_embedding:
                    spk_embedding = self.embedding(spk_idx)  # (batch_size, n_sources, latent_dim)
                    spk_embedding = spk_embedding / (
                        torch.linalg.vector_norm(spk_embedding, dim=2, keepdim=True) + eps
                    )
            else:
                spk_vector = self.speaker_stack(mixture)  # (batch_size, n_sources, latent_dim, T)
                sorted_spk_vector = self.apply_kmeans(
                    spk_vector, feature_last=False
                )  # (batch_size, n_sources, latent_dim, T)
                spk_centroids = sorted_spk_vector.mean(
                    dim=-1
                )  # (batch_size, n_sources, latent_dim)
                spk_embedding = spk_centroids / (
                    torch.linalg.vector_norm(spk_centroids, dim=2, keepdim=True) + eps
                )
                estimated_sources = self.separation_stack(
                    mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim
                )

        output = []
        output.append(estimated_sources)

        if return_spk_vector:
            output.append(sorted_spk_vector)

        if return_spk_embedding:
            output.append(spk_embedding)

        if return_all_spk_embedding:
            all_spk_embedding = self.embedding(self.all_spk_idx)  # (n_training_sources, latent_dim)
            all_spk_embedding = all_spk_embedding / (
                torch.linalg.vector_norm(all_spk_embedding, dim=1, keepdim=True) + eps
            )
            output.append(all_spk_embedding)

        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)

        return output

    def solve_permutation(self, mixture, spk_idx):
        eps = self.eps

        spk_vector = self.speaker_stack(mixture)  # (batch_size, n_sources, latent_dim, T)
        spk_vector = spk_vector.permute(
            0, 3, 1, 2
        ).contiguous()  # (batch_size, T, n_sources, latent_dim)

        spk_embedding = self.embedding(spk_idx)  # (batch_size, n_sources, latent_dim)
        all_spk_embedding = self.embedding(self.all_spk_idx)  # (n_training_sources, latent_dim)

        spk_embedding = spk_embedding / (
            torch.linalg.vector_norm(spk_embedding, dim=2, keepdim=True) + eps
        )
        all_spk_embedding = all_spk_embedding / (
            torch.linalg.vector_norm(all_spk_embedding, dim=1, keepdim=True) + eps
        )

        _, sorted_idx = self.compute_pit_speaker_loss(
            spk_vector, spk_embedding, all_spk_embedding, feature_last=True, batch_mean=False
        )  # (batch_size, T, n_sources)

        return sorted_idx


class SpeakerStack(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim=512,
        kernel_size=3,
        num_layers=14,
        dilated=True,
        separable=True,
        causal=False,
        nonlinear=None,
        norm=True,
        n_sources=2,
        eps=EPS,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.n_sources = n_sources

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.nonlinear, self.norm = nonlinear, norm

        self.eps = eps

        net = []

        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2

            if idx == 0:
                _in_channels = in_channels
                _out_channels = latent_dim
            elif idx == num_layers - 1:
                _in_channels = latent_dim
                _out_channels = n_sources * latent_dim
            else:
                _in_channels = latent_dim
                _out_channels = latent_dim

            block = ResidualBlock1d(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                separable=separable,
                causal=causal,
                nonlinear=nonlinear,
                norm=norm,
                eps=eps,
            )
            net.append(block)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, 1, T)
        Returns:
            output: (batch_size, n_sources, latent_dim, T)
        """
        n_sources = self.n_sources
        eps = self.eps
        x = input

        for idx in range(self.num_layers):
            x = self.net[idx](x)

        batch_size, _, T = x.size()
        output = x.view(batch_size, n_sources, -1, T)
        output = output / (torch.linalg.vector_norm(output, dim=2, keepdim=True) + eps)

        return output

    def get_config(self):
        config = {}

        config["in_channels"] = self.in_channels
        config["latent_dim"] = self.latent_dim
        config["kernel_size"] = self.kernel_size
        config["num_layers"] = self.num_layers

        config["dilated"], config["separable"], config["causal"] = (
            self.dilated,
            self.separable,
            self.causal,
        )
        config["nonlinear"], config["norm"] = self.nonlinear, self.norm

        config["n_sources"] = self.n_sources

        config["eps"] = self.eps

        return config


class SeparationStack(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim=512,
        kernel_size_in=4,
        kernel_size=3,
        num_blocks=4,
        num_layers=10,
        dilated=True,
        separable=True,
        causal=False,
        nonlinear=None,
        norm=True,
        n_sources=2,
        eps=EPS,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kernel_size_in, self.kernel_size = kernel_size_in, kernel_size
        self.num_blocks, self.num_layers = num_blocks, num_layers

        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.nonlinear, self.norm = nonlinear, norm

        self.n_sources = n_sources

        self.eps = eps

        self.conv1d = nn.Conv1d(in_channels, latent_dim, kernel_size=kernel_size_in, stride=1)

        net = []
        fc_weights, fc_biases = [], []

        for block_idx in range(num_blocks):
            subnet = []
            sub_fc_weights, sub_fc_biases = [], []

            for layer_idx in range(num_layers):
                if dilated:
                    dilation = 2**layer_idx
                    stride = 1
                else:
                    dilation = 1
                    stride = 2

                if block_idx == num_blocks - 1 and layer_idx == num_layers - 1:
                    dual_head = False
                else:
                    dual_head = True

                block = FiLMResidualBlock1d(
                    latent_dim,
                    latent_dim,
                    skip_channels=n_sources,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    separable=separable,
                    causal=causal,
                    nonlinear=nonlinear,
                    norm=norm,
                    dual_head=dual_head,
                    eps=eps,
                )
                subnet.append(block)

                # For FiLM
                weights = MultiSourceProjection1d(latent_dim, latent_dim, n_sources=n_sources)
                biases = MultiSourceProjection1d(latent_dim, latent_dim, n_sources=n_sources)
                sub_fc_weights.append(weights)
                sub_fc_biases.append(biases)

            subnet = nn.Sequential(*subnet)
            net.append(subnet)

            sub_fc_weights, sub_fc_biases = (
                nn.ModuleList(sub_fc_weights),
                nn.ModuleList(sub_fc_biases),
            )
            fc_weights.append(sub_fc_weights)
            fc_biases.append(sub_fc_biases)

        self.fc_weights = nn.ModuleList(fc_weights)
        self.fc_biases = nn.ModuleList(fc_biases)

        self.net = nn.Sequential(*net)

    def forward(self, input, spk_centroids, return_all=False, stack_dim=1):
        """
        Args:
            input (batch_size, in_channels, T)
            spk_centroids (batch_size, n_sources, latent_dim)
        Returns:
            output:
                (batch_size, num_blocks*num_layers, n_sources, T) if return_all
                (batch_size, n_sources, T) otherwise
        """
        padding = self.kernel_size_in - 1
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.conv1d(x)

        skip_connection = []

        for block_idx in range(self.num_blocks):
            fc_weights_block = self.fc_weights[block_idx]
            fc_biases_block = self.fc_biases[block_idx]
            net_block = self.net[block_idx]
            for layer_idx in range(self.num_layers):
                gamma = fc_weights_block[layer_idx](spk_centroids)
                beta = fc_biases_block[layer_idx](spk_centroids)
                x, skip = net_block[layer_idx](x, gamma, beta)
                skip_connection.append(skip)

        if return_all:
            output = torch.stack(skip_connection, dim=stack_dim)
        else:
            output = skip_connection[-1]

        return output

    def get_config(self):
        config = {}

        config["in_channels"] = self.in_channels
        config["latent_dim"] = self.latent_dim
        config["kernel_size_in"], config["kernel_size"] = self.kernel_size_in, self.kernel_size
        config["num_blocks"], config["num_layers"] = self.num_blocks, self.num_layers

        config["dilated"], config["separable"], config["causal"] = (
            self.dilated,
            self.separable,
            self.causal,
        )
        config["nonlinear"], config["norm"] = self.nonlinear, self.norm

        config["n_sources"] = self.n_sources
        config["eps"] = self.eps

        return config


class ResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=512,
        kernel_size=3,
        stride=1,
        dilation=1,
        separable=True,
        causal=False,
        nonlinear=None,
        norm=True,
        eps=EPS,
    ):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm

        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                causal=causal,
                nonlinear=nonlinear,
                norm=norm,
                eps=eps,
            )
        else:
            self.conv1d = ConvBlock1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                causal=causal,
                nonlinear=nonlinear,
                norm=norm,
                eps=eps,
            )

    def forward(self, input):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        separable, causal = self.separable, self.causal

        _, _, T_original = input.size()

        residual = input

        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1

        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))

        if separable:
            output = self.separable_conv1d(x)
        else:
            output = self.conv1d(x)

        if input.size(1) == output.size(1):
            output = output + residual

        return output


class FiLMResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=512,
        skip_channels=2,
        kernel_size=3,
        stride=1,
        dilation=1,
        separable=True,
        causal=False,
        nonlinear=None,
        norm=True,
        dual_head=False,
        eps=EPS,
    ):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.dual_head = dual_head
        self.norm = norm

        if separable:
            self.output_conv1d = FiLMDepthwiseSeparableConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                causal=causal,
                nonlinear=nonlinear,
                norm=norm,
                eps=eps,
            )
        else:
            self.output_conv1d = FiLMConvBlock1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                causal=causal,
                nonlinear=nonlinear,
                norm=norm,
                eps=eps,
            )

        self.skip_conv1d = nn.Conv1d(
            out_channels, skip_channels, kernel_size=1, stride=1, dilation=1
        )

    def forward(self, input, gamma, beta):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        causal = self.causal
        dual_head = self.dual_head

        _, _, T_original = input.size()

        residual = input

        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1

        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.output_conv1d(x, gamma, beta)
        x = x + residual
        skip = self.skip_conv1d(x)

        if dual_head:
            output = x
        else:
            output = None

        return output, skip


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=512,
        kernel_size=3,
        stride=1,
        dilation=1,
        causal=False,
        nonlinear=None,
        norm=True,
        eps=EPS,
    ):
        super().__init__()

        self.norm = norm
        self.eps = eps

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)

    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm

        x = self.conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        output = x

        return output


class FiLMConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=512,
        kernel_size=3,
        stride=1,
        dilation=1,
        causal=False,
        nonlinear=None,
        norm=True,
        eps=EPS,
    ):
        super().__init__()

        self.norm = norm
        self.eps = eps

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )
        self.film1d = FiLM1d()

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)

    def forward(self, input, gamma, beta):
        nonlinear, norm = self.nonlinear, self.norm

        x = self.conv1d(input)
        x = self.film1d(x, gamma, beta)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        output = x

        return output


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=512,
        kernel_size=3,
        stride=1,
        dilation=1,
        causal=False,
        nonlinear=None,
        norm=True,
        eps=EPS,
    ):
        super().__init__()

        self.norm = norm
        self.eps = eps

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)

    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm

        x = self.depthwise_conv1d(input)
        x = self.pointwise_conv1d(x)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        output = x

        return output


class FiLMDepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=512,
        kernel_size=3,
        stride=1,
        dilation=1,
        causal=False,
        nonlinear=None,
        norm=True,
        eps=EPS,
    ):
        super().__init__()

        self.norm = norm
        self.eps = eps

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.film1d = FiLM1d()

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))

            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)

    def forward(self, input, gamma, beta):
        nonlinear, norm = self.nonlinear, self.norm

        x = self.depthwise_conv1d(input)
        x = self.pointwise_conv1d(x)
        x = self.film1d(x, gamma, beta)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        output = x

        return output


class MultiSourceProjection1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_sources, channel_last=True):
        super().__init__()

        assert channel_last, "channel_last should be True."
        self.linear = nn.Linear(n_sources * in_channels, out_channels)

    def forward(self, input):
        batch_size, n_sources, in_channels = input.size()

        x = input.view(batch_size, n_sources * in_channels)
        output = self.linear(x)

        return output


class _SpeakerDistance(nn.Module):
    def __init__(self, n_sources):
        super().__init__()

        self.mask = nn.Parameter(1 - torch.eye(n_sources), requires_grad=False)

        zero_dim_size = ()
        self.scale, self.bias = (
            nn.Parameter(torch.empty(zero_dim_size), requires_grad=True),
            nn.Parameter(torch.empty(zero_dim_size), requires_grad=True),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(
        self, spk_vector, spk_embedding, _, feature_last=True, batch_mean=True, time_mean=True
    ):
        """
        Args:
            spk_vector:
                (batch_size, T, n_sources, latent_dim) if feature_last
                (batch_size, n_sources, latent_dim, T) otherwise
            spk_embedding: (batch_size, n_sources, latent_dim)
            _: All speaker embedding (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T) or (T,)
        """
        if not feature_last:
            spk_vector = spk_vector.permute(
                0, 3, 1, 2
            ).contiguous()  # (batch_size, T, n_sources, latent_dims)

        loss_euclid = self.compute_euclid_distance(
            spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1
        )  # (batch_size, T, n_sources)

        distance_table = self.compute_euclid_distance(
            spk_vector.unsqueeze(dim=3), spk_vector.unsqueeze(dim=2), dim=-1
        )  # (batch_size, T, n_sources, n_sources)
        loss_hinge = F.relu(1 - distance_table)  # (batch_size, T, n_sources, n_sources)
        loss_hinge = torch.sum(self.mask * loss_hinge, dim=2)  # (batch_size, T, n_sources)

        loss = loss_euclid + loss_hinge
        loss = loss.mean(dim=-1)

        if time_mean:
            loss = loss.mean(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)
        return loss

    def compute_euclid_distance(self, input, target, dim=-1, keepdim=False, scale=None, bias=None):
        distance = torch.sum((input - target) ** 2, dim=dim, keepdim=keepdim)

        if scale is not None or bias is not None:
            distance = torch.abs(self.scale) * distance + self.bias

        return distance


# ---------------------------------------------------------------------------
# menagerie build/example glue
# ---------------------------------------------------------------------------
def build_wavesplit() -> nn.Module:
    """Build a tiny real WaveSplit for tracing (kernel_size=3, small
    num_layers/num_blocks, non-separable for a fast small trace). Keeps
    `dilated=True` (the repo's own default, used by its `_test_wavesplit`)
    since `dilated=False` uses stride=2 per layer, which breaks the
    dual-head residual add's shape match in `FiLMResidualBlock1d`."""

    n_sources = 2
    n_training_sources = 4
    latent_dim = 8

    speaker_stack = SpeakerStack(
        in_channels=1,
        latent_dim=latent_dim,
        kernel_size=3,
        num_layers=3,
        separable=False,
        n_sources=n_sources,
    )
    separation_stack = SeparationStack(
        in_channels=1,
        latent_dim=latent_dim,
        kernel_size=3,
        num_blocks=1,
        num_layers=2,
        separable=False,
        n_sources=n_sources,
    )
    spk_criterion = _SpeakerDistance(n_sources=n_sources)

    model = WaveSplit(
        speaker_stack,
        separation_stack,
        latent_dim,
        n_sources=n_sources,
        n_training_sources=n_training_sources,
        spk_criterion=spk_criterion,
    )
    model.eval()
    return model


def example_wavesplit() -> torch.Tensor:
    """Single-channel mixture waveform, (batch, 1, T)."""

    return torch.randn(1, 1, 256)


MENAGERIE_ENTRIES = [
    ("WaveSplit", "build_wavesplit", "example_wavesplit", "2020", "speech/audio"),
]
