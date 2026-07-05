# SOURCE: vendored from uhh-pd-ml/sk_cathode @ ff0a2b2, HEPML-AnomalyDetection/CATHODE @ ecbe20e, znxlwm/pytorch-CartoonGAN @ 67a872d, nii-yamagishilab/Capsule-Forensics-v2 @ c45e55c, BAAI-DCAI/Bunny @ 5e4e736
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from timm.layers.norm_act import LayerNormAct2d
from torch import nn
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from torchvision.ops.misc import SqueezeExcitation as SELayer


class SkCathodeNeuralNetwork(nn.Module):
    """A PyTorch module implementing a simple feed-forward neural network."""

    def __init__(self, layers: list[int] | None = None, n_inputs: int = 4) -> None:
        """Initialize the vendored sk_cathode feed-forward classifier."""
        super().__init__()
        if layers is None:
            layers = [64, 64, 64]

        modules: list[nn.Module] = []
        for nodes in layers:
            modules.append(nn.Linear(n_inputs, nodes))
            modules.append(nn.ReLU())
            n_inputs = nodes
        modules.append(nn.Linear(n_inputs, 1))
        modules.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classifier forward pass."""
        return self.model_stack(x)


class SkCathodeAutoencoderModel(nn.Module):
    """A PyTorch module implementing a simple feed-forward autoencoder."""

    def __init__(self, layers: list[int] | None = None, n_inputs: int = 10) -> None:
        """Initialize the vendored sk_cathode autoencoder."""
        super().__init__()
        if layers is None:
            layers = [32, 16, 4, 16, 32]

        modules: list[nn.Module] = []
        for i, nodes in enumerate(layers):
            modules.append(nn.Linear(n_inputs, nodes))
            if i < (len(layers) - 1):
                modules.append(nn.ReLU())
            n_inputs = nodes
        self.model_stack = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the autoencoder forward pass."""
        return self.model_stack(x)


class CathodeClassifier(nn.Module):
    """Vendored CATHODE binary classifier."""

    def __init__(self, layers: list[int], n_inputs: int = 5) -> None:
        """Initialize the CATHODE classifier."""
        super().__init__()
        modules: list[nn.Module] = []
        for nodes in layers:
            modules.append(nn.Linear(n_inputs, nodes))
            modules.append(nn.ReLU())
            n_inputs = nodes
        modules.append(nn.Linear(n_inputs, 1))
        modules.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classifier forward pass."""
        return self.model_stack(x)


def initialize_weights(net: nn.Module) -> None:
    """Initialize CartoonGAN modules as in the source utility."""
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ResnetBlock(nn.Module):
    """Vendored CartoonGAN residual block."""

    def __init__(self, channel: int, kernel: int, stride: int, padding: int) -> None:
        """Initialize the residual block."""
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)
        initialize_weights(self)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""
        x = F.relu(self.conv1_norm(self.conv1(input_tensor)), True)
        x = self.conv2_norm(self.conv2(x))
        return input_tensor + x


class CartoonGenerator(nn.Module):
    """Vendored CartoonGAN generator."""

    def __init__(self, in_nc: int, out_nc: int, nf: int = 32, nb: int = 6) -> None:
        """Initialize the CartoonGAN generator."""
        super().__init__()
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1),
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )
        self.resnet_blocks = nn.Sequential(*[ResnetBlock(nf * 4, 3, 1, 1) for _ in range(nb)])
        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the generator forward pass."""
        x = self.down_convs(input_tensor)
        x = self.resnet_blocks(x)
        return self.up_convs(x)


class StatsNet(nn.Module):
    """Vendored Capsule-Forensics statistics block."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-channel mean and standard deviation statistics."""
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2] * x.data.shape[3])
        mean = torch.mean(x, 2)
        std = torch.std(x, 2)
        return torch.stack((mean, std), dim=1)


class View(nn.Module):
    """Vendored Capsule-Forensics view helper."""

    def __init__(self, *shape: int) -> None:
        """Initialize the target shape."""
        super().__init__()
        self.shape = shape

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor."""
        return input_tensor.view(self.shape)


class CapsuleFeatureExtractor(nn.Module):
    """Vendored Capsule-Forensics feature extractor."""

    def __init__(self, no_caps: int = 10) -> None:
        """Initialize the capsule feature extractor."""
        super().__init__()
        self.capsules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    StatsNet(),
                    nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(8),
                    nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(1),
                    View(-1, 8),
                )
                for _ in range(no_caps)
            ]
        )

    def squash(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Apply capsule squash nonlinearity."""
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the feature extractor."""
        outputs = [capsule(x) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)
        return self.squash(output, dim=-1)


class RoutingLayer(nn.Module):
    """Vendored Capsule-Forensics dynamic routing layer."""

    def __init__(
        self,
        num_input_capsules: int,
        num_output_capsules: int,
        data_in: int,
        data_out: int,
        num_iterations: int,
    ) -> None:
        """Initialize the routing layer."""
        super().__init__()
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(
            torch.randn(num_output_capsules, num_input_capsules, data_out, data_in)
        )

    def squash(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Apply capsule squash nonlinearity."""
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x: torch.Tensor, random: bool = False, dropout: float = 0.0) -> torch.Tensor:
        """Run the routing layer."""
        x = x.transpose(2, 1)
        route_weights = self.route_weights
        if random:
            route_weights = route_weights + 0.01 * torch.randn_like(route_weights)
        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]
        priors = priors.transpose(1, 0)
        if dropout > 0.0:
            drop = torch.empty_like(priors).bernoulli_(1.0 - dropout)
            priors = priors * drop
        logits = torch.zeros_like(priors)
        for i in range(self.num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)
            if i != self.num_iterations - 1:
                logits = logits + priors * outputs
        outputs = outputs.squeeze()
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        return outputs


class CapsuleNet(nn.Module):
    """Vendored Capsule-Forensics capsule classifier."""

    def __init__(self, num_class: int, no_caps: int = 10) -> None:
        """Initialize the capsule classifier."""
        super().__init__()
        self.num_class = num_class
        self.fea_ext = CapsuleFeatureExtractor(no_caps=no_caps)
        self.fea_ext.apply(self.weights_init)
        self.routing_stats = RoutingLayer(
            num_input_capsules=no_caps,
            num_output_capsules=num_class,
            data_in=8,
            data_out=4,
            num_iterations=2,
        )

    def weights_init(self, module: nn.Module) -> None:
        """Initialize convolution and batch-normalization modules."""
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, random: bool = False, dropout: float = 0.0) -> torch.Tensor:
        """Run the capsule classifier and return logits."""
        z = self.fea_ext(x)
        z = self.routing_stats(z, random, dropout=dropout)
        return z


class BunnyMinigpt(nn.Module):
    """Vendored Bunny Minigpt vision projector."""

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the projector."""
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the projector forward pass."""
        b, num_tokens, c = x.shape
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")
        x = x.view(b, num_tokens // 4, c * 4)
        return self.linear(x)


class BunnyLDPBlock(nn.Module):
    """Vendored Bunny lightweight downsample projector block."""

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the projector block."""
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size

        def layer_norm(channels: int) -> LayerNormAct2d:
            """Build the source LayerNormAct2d factory."""
            return LayerNormAct2d(channels, act_layer=None)

        def se_layer(channels: int, squeeze_channels: int) -> SELayer:
            """Build the source squeeze-excitation factory."""
            return SELayer(channels, squeeze_channels, scale_activation=nn.Hardsigmoid)

        self.mlp = nn.Sequential(nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc))
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(
                InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer
            ),
            InvertedResidual(
                InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the projector block."""
        b, num_tokens, _ = x.shape
        h = int(num_tokens**0.5)
        x = self.mlp(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        return x.flatten(2).permute(0, 2, 1)


class MlpChannels:
    """Vendored BuildingNet MLP channel metadata."""

    def __init__(
        self,
        in_channel: int = 10,
        hidden_channel: int = 32,
        out_channel: int = 32,
        num_hidden: int = 32,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the channel metadata."""
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.num_hidden = num_hidden
        self.dropout = dropout


class GnnModelMeta:
    """Vendored BuildingNet model metadata."""

    def __init__(self, mlp_channels: list[MlpChannels], normalization: str = "BN") -> None:
        """Initialize the model metadata."""
        self.normalization = normalization
        self.mlp_channels = mlp_channels
        self.num_updates = len(mlp_channels)


class BuildingMlp(nn.Module):
    """Vendored BuildingNet MLP."""

    def __init__(self, channels: list[int], num_layers: int, dropout_prob: float = 0.5) -> None:
        """Initialize the MLP."""
        super().__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.mlpmodules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.channels[i], self.channels[i + 1]),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(self.channels[i + 1]),
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.mlpNoBNmodules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.channels[i], self.channels[i + 1]),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.dropout_prob = dropout_prob
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize linear and batch-normalization weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight.data)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run the BuildingNet MLP."""
        x = input_features
        modules = self.mlpNoBNmodules if len(x) == 1 else self.mlpmodules
        for mlpmod in modules:
            x = mlpmod(x)
        return x


class GnnNode(nn.Module):
    """Vendored BuildingNet node update layer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_hidden: int,
        normalization: str = "BN",
        is_decoder: bool = False,
    ) -> None:
        """Initialize the node update layer."""
        super().__init__()
        del normalization, is_decoder
        channels = [in_channels]
        for _ in range(num_hidden):
            channels.append(hidden_channels)
        channels.append(out_channels)
        self.mlp = BuildingMlp(channels, len(channels))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run the node update."""
        return self.mlp(input_features)


class BuildnetEncNode(nn.Module):
    """Vendored BuildingNet node encoder."""

    def __init__(self, modelmeta: GnnModelMeta) -> None:
        """Initialize the node encoder."""
        super().__init__()
        self.channels = modelmeta.mlp_channels
        self.normalization = modelmeta.normalization
        self.num_updates = len(self.channels)
        self.GNNModule = nn.ModuleList(
            [
                GnnNode(
                    self.channels[i].in_channel,
                    self.channels[i].hidden_channel,
                    self.channels[i].out_channel,
                    self.channels[i].num_hidden,
                    self.normalization,
                )
                for i in range(self.num_updates)
            ]
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run the node encoder."""
        x = input_features
        for module in self.GNNModule:
            x = module(x)
        return x


class BuildnetDecNode(nn.Module):
    """Vendored BuildingNet node decoder."""

    def __init__(self, modelmeta: GnnModelMeta) -> None:
        """Initialize the node decoder."""
        super().__init__()
        self.channels = modelmeta.mlp_channels
        self.normalization = modelmeta.normalization
        self.num_updates = len(self.channels)
        self.GNNModule = nn.ModuleList(
            [
                GnnNode(
                    self.channels[i].in_channel,
                    self.channels[i].hidden_channel,
                    self.channels[i].out_channel,
                    self.channels[i].num_hidden,
                    self.normalization,
                    True,
                )
                for i in range(self.num_updates)
            ]
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Run the node decoder."""
        for module in self.GNNModule:
            node_features = module(node_features)
        return node_features


class BuildingNetNodeSegmentation(nn.Module):
    """Vendored BuildingNet node segmentation encoder-decoder."""

    def __init__(self) -> None:
        """Initialize a tiny node encoder-decoder using the source channel pattern."""
        super().__init__()
        encmeta = GnnModelMeta(
            [
                MlpChannels(6, 16, 8, 1, 0),
                MlpChannels(8, 16, 8, 1, 0),
                MlpChannels(8, 16, 8, 1, 0),
            ]
        )
        decmeta = GnnModelMeta([MlpChannels(8, 8, 4, 1)])
        self.buildnet_enc = BuildnetEncNode(encmeta)
        self.buildnet_dec = BuildnetDecNode(decmeta)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Run the node segmentation model."""
        return self.buildnet_dec(self.buildnet_enc(node_features))


class GPTConfig:
    """Vendored C-BeT GPT config."""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    discrete_input = False
    input_size = 10
    n_embd = 768
    n_layer = 12

    def __init__(self, vocab_size: int, block_size: int, **kwargs: Any) -> None:
        """Initialize the config."""
        self.vocab_size = vocab_size
        self.block_size = block_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class CausalSelfAttention(nn.Module):
    """Vendored C-BeT masked self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize masked self-attention."""
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run masked self-attention."""
        batch, steps, channels = x.size()
        k = self.key(x).view(batch, steps, self.n_head, channels // self.n_head).transpose(1, 2)
        q = self.query(x).view(batch, steps, self.n_head, channels // self.n_head).transpose(1, 2)
        v = self.value(x).view(batch, steps, self.n_head, channels // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_drop(self.proj(y))


class GPTBlock(nn.Module):
    """Vendored C-BeT Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the block."""
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Transformer block."""
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Vendored C-BeT GPT model."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the GPT model."""
        super().__init__()
        if config.discrete_input:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        else:
            self.tok_emb = nn.Linear(config.input_size, config.n_embd)
        self.discrete_input = config.discrete_input
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize GPT weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the GPT forward pass and return logits."""
        if self.discrete_input:
            _, steps = idx.size()
        else:
            _, steps, _ = idx.size()
        assert steps <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :steps, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


class Flatten(nn.Module):
    """Vendored BurstCCN flatten layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten all dimensions after the batch."""
        return x.view(x.size(0), -1)


class BurstCCNOutputLayer(nn.Module):
    """Vendored BurstCCN output layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_baseline: float,
        weight_y_learning: bool,
        weight_q_learning: bool,
        device: torch.device,
    ) -> None:
        """Initialize the output layer."""
        super().__init__()
        del weight_y_learning, weight_q_learning
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.forward_noise = None
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features).to(device), requires_grad=False
        )
        self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=False)
        self.delta_weight = nn.Parameter(
            torch.zeros(out_features, in_features).to(device), requires_grad=False
        )
        self.delta_bias = nn.Parameter(torch.zeros(out_features).to(device), requires_grad=False)
        self.p = self.p_baseline * torch.ones(self.out_features).to(device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the output layer."""
        if self.forward_noise is not None:
            self.input = input_tensor + self.forward_noise * torch.randn(
                input_tensor.shape, device=input_tensor.device
            )
        else:
            self.input = input_tensor
        self.e = torch.sigmoid(F.linear(self.input, self.weight, self.bias))
        return self.e


class BurstCCNHiddenLayer(nn.Module):
    """Vendored BurstCCN hidden layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        next_features: int,
        p_baseline: float,
        weight_y_learning: bool,
        weight_q_learning: bool,
        device: torch.device,
    ) -> None:
        """Initialize the hidden layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.forward_noise = None
        self.weight_Y_learning = weight_y_learning
        self.weight_Q_learning = weight_q_learning
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features).to(device), requires_grad=False
        )
        self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=False)
        self.weight_Y = nn.Parameter(
            torch.Tensor(next_features, out_features).to(device), requires_grad=False
        )
        self.weight_Q = nn.Parameter(
            torch.Tensor(next_features, out_features).to(device), requires_grad=False
        )
        self.delta_weight = nn.Parameter(
            torch.zeros(out_features, in_features).to(device), requires_grad=False
        )
        self.delta_bias = nn.Parameter(torch.zeros(out_features).to(device), requires_grad=False)
        if self.weight_Y_learning:
            self.delta_weight_Y = nn.Parameter(
                torch.zeros(next_features, out_features).to(device), requires_grad=False
            )
        if self.weight_Q_learning:
            self.delta_weight_Q = nn.Parameter(
                torch.zeros(next_features, out_features).to(device), requires_grad=False
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the hidden layer."""
        if self.forward_noise is not None:
            self.input = input_tensor + self.forward_noise * torch.randn(
                input_tensor.shape, device=input_tensor.device
            )
        else:
            self.input = input_tensor
        self.e = torch.sigmoid(F.linear(self.input, self.weight, self.bias))
        return self.e


class BurstCCN(nn.Module):
    """Vendored BurstCCN feed-forward network."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        p_baseline: float,
        n_hidden_layers: int,
        n_hidden_units: int,
        y_mode: str,
        q_mode: str,
        y_scale: float,
        q_scale: float,
        y_learning: bool,
        q_learning: bool,
        device: torch.device,
    ) -> None:
        """Initialize BurstCCN."""
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.p_baseline = p_baseline
        self.Y_mode = y_mode
        self.Y_scale = y_scale
        self.Q_mode = q_mode
        self.Q_scale = q_scale
        self.Y_learning = y_learning
        self.Q_learning = q_learning
        self.device = device
        self.feature_layers = [Flatten()]
        self.classification_layers: list[nn.Module] = []
        if n_hidden_layers == 0:
            self.classification_layers.append(
                BurstCCNOutputLayer(n_inputs, n_outputs, p_baseline, y_learning, q_learning, device)
            )
        elif n_hidden_layers == 1:
            self.classification_layers.append(
                BurstCCNHiddenLayer(
                    n_inputs, n_hidden_units, n_outputs, p_baseline, y_learning, q_learning, device
                )
            )
            self.classification_layers.append(
                BurstCCNOutputLayer(
                    n_hidden_units, n_outputs, p_baseline, y_learning, q_learning, device
                )
            )
        else:
            self.classification_layers.append(
                BurstCCNHiddenLayer(
                    n_inputs,
                    n_hidden_units,
                    n_hidden_units,
                    p_baseline,
                    y_learning,
                    q_learning,
                    device,
                )
            )
            for _ in range(1, n_hidden_layers - 1):
                self.classification_layers.append(
                    BurstCCNHiddenLayer(
                        n_hidden_units,
                        n_hidden_units,
                        n_hidden_units,
                        p_baseline,
                        y_learning,
                        q_learning,
                        device,
                    )
                )
            self.classification_layers.append(
                BurstCCNHiddenLayer(
                    n_hidden_units,
                    n_hidden_units,
                    n_outputs,
                    p_baseline,
                    y_learning,
                    q_learning,
                    device,
                )
            )
            self.classification_layers.append(
                BurstCCNOutputLayer(
                    n_hidden_units, n_outputs, p_baseline, y_learning, q_learning, device
                )
            )
        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize BurstCCN weights."""
        for module in self.modules():
            if isinstance(module, (BurstCCNHiddenLayer, BurstCCNOutputLayer)):
                nn.init.xavier_normal_(module.weight, gain=3.6)
                nn.init.constant_(module.bias, 0)
                if isinstance(module, BurstCCNHiddenLayer):
                    if self.Y_mode == "tied":
                        module.weight_Y = module.weight
                    else:
                        nn.init.normal_(module.weight_Y, 0, self.Y_scale)
                    if self.Q_mode == "tied":
                        module.weight_Q = module.weight
                    else:
                        nn.init.normal_(module.weight_Q, 0, self.Q_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the BurstCCN forward pass."""
        return self.out(x)


def build_bump() -> nn.Module:
    """Build the BUMP/sk_cathode neural classifier."""
    return SkCathodeNeuralNetwork(layers=[8, 8], n_inputs=4).eval()


def example_input_bump() -> torch.Tensor:
    """Return an example BUMP/sk_cathode input."""
    return torch.randn(2, 4)


def build_sk_cathode_autoencoder() -> nn.Module:
    """Build the sk_cathode autoencoder."""
    return SkCathodeAutoencoderModel(layers=[8, 3, 8], n_inputs=8).eval()


def example_input_sk_cathode_autoencoder() -> torch.Tensor:
    """Return an example sk_cathode autoencoder input."""
    return torch.randn(2, 8)


def build_cartoon_gan() -> nn.Module:
    """Build the CartoonGAN generator."""
    return CartoonGenerator(3, 3, nf=8, nb=1).eval()


def example_input_cartoon_gan() -> torch.Tensor:
    """Return an example CartoonGAN image input."""
    return torch.randn(1, 3, 32, 32)


def build_capsule_forensics() -> nn.Module:
    """Build the Capsule-Forensics capsule classifier."""
    return CapsuleNet(num_class=2, no_caps=4).eval()


def example_input_capsule_forensics() -> torch.Tensor:
    """Return an example Capsule-Forensics feature-map input."""
    return torch.randn(2, 256, 8, 8)


def build_cathode_classifier() -> nn.Module:
    """Build the CATHODE classifier."""
    return CathodeClassifier(layers=[8, 8], n_inputs=5).eval()


def example_input_cathode_classifier() -> torch.Tensor:
    """Return an example CATHODE classifier input."""
    return torch.randn(2, 5)


def build_bunny_projector() -> nn.Module:
    """Build the Bunny lightweight downsample projector."""
    return BunnyLDPBlock(SimpleNamespace(mm_hidden_size=8, hidden_size=8)).eval()


def example_input_bunny_projector() -> torch.Tensor:
    """Return an example Bunny projector input."""
    return torch.randn(1, 16, 8)


def build_buildingnet() -> nn.Module:
    """Build the BuildingNet node segmentation model."""
    return BuildingNetNodeSegmentation().eval()


def example_input_buildingnet() -> torch.Tensor:
    """Return an example BuildingNet node-feature input."""
    return torch.randn(4, 6)


def build_cbet() -> nn.Module:
    """Build the C-BeT Transformer."""
    config = GPTConfig(
        vocab_size=7,
        block_size=5,
        input_size=3,
        n_embd=16,
        n_layer=1,
        n_head=2,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT(config).eval()


def example_input_cbet() -> torch.Tensor:
    """Return an example C-BeT continuous-token input."""
    return torch.randn(2, 5, 3)


def build_burstccn() -> nn.Module:
    """Build the BurstCCN model."""
    return BurstCCN(
        n_inputs=4,
        n_outputs=3,
        p_baseline=0.5,
        n_hidden_layers=1,
        n_hidden_units=5,
        y_mode="random_init",
        q_mode="random_init",
        y_scale=0.1,
        q_scale=0.1,
        y_learning=False,
        q_learning=False,
        device=torch.device("cpu"),
    ).eval()


def example_input_burstccn() -> torch.Tensor:
    """Return an example BurstCCN input."""
    return torch.randn(2, 4)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "BuildingNet (3D part segmentation)",
        "build_buildingnet",
        "example_input_buildingnet",
        2021,
        "CV6-BUILDINGNET",
    ),
    ("BUMP (Bump Hunter neural variant)", "build_bump", "example_input_bump", 2023, "CV6-BUMP"),
    ("CartoonGAN", "build_cartoon_gan", "example_input_cartoon_gan", 2018, "CV6-CARTOONGAN"),
    (
        "Capsule-Forensics",
        "build_capsule_forensics",
        "example_input_capsule_forensics",
        2019,
        "CV6-CAPSULEFORENSICS",
    ),
    (
        "CATHODE",
        "build_cathode_classifier",
        "example_input_cathode_classifier",
        2022,
        "CV6-CATHODE",
    ),
    ("Bunny", "build_bunny_projector", "example_input_bunny_projector", 2024, "CV6-BUNNY"),
    (
        "C-BeT (Continuous Behavior Transformer)",
        "build_cbet",
        "example_input_cbet",
        2022,
        "CV6-CBET",
    ),
    ("BurstCCN", "build_burstccn", "example_input_burstccn", 2022, "CV6-BURSTCCN"),
]
