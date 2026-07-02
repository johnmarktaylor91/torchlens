# SOURCE: vendored from raunaqbhirangi/visuoskin @ 66519d2ffc09df7df33f1f0807c88f29f849be23
# https://github.com/raunaqbhirangi/visuoskin/blob/main/agent/networks/rgb_modules.py
# https://github.com/raunaqbhirangi/visuoskin/blob/main/agent/networks/mlp.py
# https://github.com/raunaqbhirangi/visuoskin/blob/main/agent/networks/gpt.py
# https://github.com/raunaqbhirangi/visuoskin/blob/main/agent/networks/policy_head.py
# https://github.com/raunaqbhirangi/visuoskin/blob/main/agent/bc.py
# https://github.com/raunaqbhirangi/visuoskin/blob/main/utils.py
# VISK (Visuo-Skin): ResNet-18-based `ResnetEncoder` (torchvision resnet18 backbone,
# truncated + a `SpatialSoftmax`/`SpatialProjection` head) used identically for BOTH the
# RGB camera stream and each AnySkin tactile-image stream (per `BCAgent.__init__`, any
# aux key starting with "digit" routes through the same `ResnetEncoder`), fused with the
# proprioceptive/tactile-vector `MLP` projector, and consumed by the BAKU policy: a
# nanoGPT-style causal-Transformer decoder (`GPT`) with a per-timestep action token,
# followed by a `DeterministicHead` action head. The `ResnetEncoder`, `SpatialSoftmax`,
# `SpatialProjection`, `MLP`, `GPT`/`GPTConfig`/`CausalSelfAttention`/`MLP`(gpt)/`Block`,
# `DeterministicHead`, `Actor`, `weight_init`, and `TruncatedNormal` classes/functions are
# transcribed VERBATIM from the files above. Only changes: (1) the RL-training scaffolding
# in `BCAgent` (optimizers, EMA, augmentation, hydra config loading, dataset/env I/O) is
# stripped -- only the encoder/aux_projector/actor CONSTRUCTION and the Actor forward pass
# that BCAgent.__init__ and BCAgent's act() path actually build are kept; (2)
# `DeterministicHead.forward`'s stddev sampling path is exercised with a concrete
# `stddev` float instead of the training-time schedule float that hydra would normally
# supply -- same code path, just called directly; (3) `torchvision.models.resnet18`'s
# deprecated `pretrained=False` kwarg is used exactly as upstream does (random init, no
# network fetch). No architectural layer, head, or fusion mechanism was added, removed,
# or altered.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

MENAGERIE_ZOO = "vendored-pytorch"


# --- utils.py (verbatim, only what the forward path needs) ---
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


# --- agent/networks/rgb_modules.py (verbatim, visual/tactile encoder) ---
class SpatialSoftmax(nn.Module):
    """The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)"""

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        keypoint_y = (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert len(input_shape) == 3, "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ResnetEncoder(nn.Module):
    """
    A Resnet-18-based encoder for mapping an image to a latent vector.
    Used for BOTH the RGB camera stream and the AnySkin tactile-image streams.
    """

    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=False,
        freeze=False,
        remove_layer_num=2,
        no_stride=False,
        cond_dim=768,
        cond_fusion="film",
    ):
        super().__init__()

        ### 1. encode input (images) using convolutional layers
        assert remove_layer_num <= 5, "[error] please only remove <=5 layers"
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[
            :-remove_layer_num
        ]
        self.remove_layer_num = remove_layer_num

        assert len(input_shape) == 3, "[error] input shape of resnet should be (C, H, W)"

        in_channels = input_shape[0]
        if in_channels != 3:  # has eye_in_hand, increase channel size
            conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.resnet18_base = nn.Sequential(*layers[:4])
        self.block_1 = layers[4][0]
        self.block_2 = layers[4][1]
        self.block_3 = layers[5][0]
        self.block_4 = layers[5][1]

        self.cond_fusion = cond_fusion
        if cond_fusion == "film":
            self.lang_proj1 = nn.Linear(cond_dim, 64 * 2)
            self.lang_proj2 = nn.Linear(cond_dim, 64 * 2)
            self.lang_proj3 = nn.Linear(cond_dim, 128 * 2)
            self.lang_proj4 = nn.Linear(cond_dim, 128 * 2)

        if freeze:
            if in_channels != 3:
                raise Exception(
                    "[error] cannot freeze pretrained " + "resnet with the extra eye_in_hand input"
                )
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        ### 2. project the encoded input to a latent space
        x = torch.zeros(1, *input_shape)
        y = self.block_4(self.block_3(self.block_2(self.block_1(self.resnet18_base(x)))))
        output_shape = y.shape  # compute the out dim
        self.projection_layer = SpatialProjection(output_shape[1:], output_size)
        self.output_shape = self.projection_layer(y).shape

    def forward(self, x, langs=None):
        h = self.resnet18_base(x)
        h = self.block_1(h)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)
        h = self.projection_layer(h)
        return h


# --- agent/networks/mlp.py (verbatim, tactile/proprio-vector projector) ---
class MLP(torch.nn.Sequential):
    """Multi-layer perceptron module (adapted from torchvision.ops.MLP)."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        activation_layer=torch.nn.ReLU,
        inplace=None,
        bias=True,
        dropout=0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


# --- agent/networks/gpt.py (verbatim, nanoGPT-derived BAKU policy backbone) ---
def new_gelu(x):
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class CrossAttention(nn.Module):
    def __init__(self, repr_dim, nhead=4, nlayers=4, use_buffer_token=False):
        super().__init__()
        self.tf_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=repr_dim, nhead=nhead, dim_feedforward=repr_dim * 4, batch_first=True
            ),
            num_layers=nlayers,
        )
        if use_buffer_token:
            self.buffer_token = nn.Parameter(torch.randn(1, 1, repr_dim))
        self.use_buffer_token = use_buffer_token

    def forward(self, feat, cond):
        if self.use_buffer_token:
            batch_size = feat.size(0)
            buffer_token = self.buffer_token.expand(batch_size, 1, -1)
            cond_with_buffer = torch.cat([buffer_token, cond], dim=1)
            return self.tf_decoder(feat, cond_with_buffer)
        else:
            return self.tf_decoder(feat, cond)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:, :, :T, :T]
        if attn_mask is not None:
            mask = mask * attn_mask
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPTMLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.input_dim is not None
        assert config.output_dim is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.input_dim, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, input, targets=None, mask=None):
        device = input.device
        b, t, d = input.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, mask=mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


# --- agent/networks/policy_head.py (verbatim, deterministic action head) ---
class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


# --- agent/bc.py (verbatim Actor; the GPT-policy "BAKU" decoder over fused tokens) ---
class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_type="gpt",
        policy_head="deterministic",
        num_feat_per_step=1,
    ):
        super().__init__()

        self._policy_type = policy_type
        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        if policy_type == "gpt":
            self._policy = GPT(
                GPTConfig(
                    block_size=65,
                    input_dim=repr_dim,
                    output_dim=hidden_dim,
                    n_layer=2,
                    n_head=2,
                    n_embd=hidden_dim,
                    dropout=0.1,
                )
            )
        else:
            raise NotImplementedError
        self._action_head = DeterministicHead(
            hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
        )
        self.apply(weight_init)

    def forward(self, obs, num_prompt_feats, stddev, action=None, cluster_centers=None, mask=None):
        B, T, D = obs.shape
        if self._policy_type == "gpt":
            prompt = obs[:, :num_prompt_feats]
            obs = obs[:, num_prompt_feats:]
            obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
            action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
            obs = torch.cat([prompt, obs], dim=1)

            base_mask = None
            if mask is not None:
                mask = torch.cat([mask, torch.ones(B, 1).to(mask.device)], dim=1)
                mask = mask.view(B, -1, 1, self._num_feat_per_step + 1)
                base_mask = torch.ones(
                    B, mask.shape[1], self._num_feat_per_step + 1, self._num_feat_per_step + 1
                ).to(mask.device)
                base_mask[:, :, -1:] = mask

            features = self._policy(obs, mask=base_mask)
            features = features[:, num_prompt_feats:]
            num_feat_per_step = self._num_feat_per_step + 1
            features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        pred_action = self._action_head(
            features, stddev, **{"cluster_centers": cluster_centers, "action_seq": action}
        )
        return pred_action


# --- staging construction, mirroring BCAgent.__init__'s encoder/actor wiring ---
class VISKPolicy(nn.Module):
    """
    Wires together the real ResNet-18 vision/tactile encoders + MLP proprio projector +
    GPT-based BAKU Actor exactly as `BCAgent.__init__` constructs them for the
    encoder_type="resnet" branch with a "digitv1" (AnySkin tactile) aux key.
    """

    def __init__(
        self, obs_shape=(3, 84, 84), tactile_dim=15, act_dim=7, hidden_dim=128, num_prompt_feats=0
    ):
        super().__init__()
        self.repr_dim = 64
        self.num_prompt_feats = num_prompt_feats

        # visual encoder (RGB camera stream)
        self.encoder = ResnetEncoder(obs_shape, self.repr_dim, cond_dim=None, cond_fusion="none")

        # tactile-vector proprio projector (AnySkin sensor readout -> repr_dim token)
        self.aux_projector = MLP(tactile_dim, hidden_channels=[self.repr_dim, self.repr_dim])
        self.aux_projector.apply(weight_init)

        # BAKU policy: GPT decoder fusing [visual token, tactile token] per timestep
        num_feat_per_step = 2  # 1 pixel key + 1 aux (tactile) key
        self.actor = Actor(
            self.repr_dim, act_dim, hidden_dim, "gpt", "deterministic", num_feat_per_step
        )

    def forward(self, pixels, tactile, stddev=0.1):
        # pixels: (B, T, C, H, W), tactile: (B, T, tactile_dim)
        B, T = pixels.shape[:2]
        vis_feat = self.encoder(pixels.reshape(B * T, *pixels.shape[2:])).view(
            B, T, 1, self.repr_dim
        )
        tac_feat = self.aux_projector(tactile).view(B, T, 1, self.repr_dim)
        obs = torch.cat([vis_feat, tac_feat], dim=2).view(B, T * 2, self.repr_dim)
        dist = self.actor(obs, self.num_prompt_feats, stddev)
        return dist.loc  # deterministic mean action, for a concrete traceable tensor output


# --- staging entry points ---
def build_visk():
    return VISKPolicy(
        obs_shape=(3, 84, 84), tactile_dim=15, act_dim=7, hidden_dim=64, num_prompt_feats=0
    )


def example_input_visk():
    T = 2
    pixels = torch.randn(1, T, 3, 84, 84)
    tactile = torch.randn(1, T, 15)
    return (pixels, tactile)


MENAGERIE_ENTRIES = [
    ("VISK (Visuo-Skin)", "build_visk", "example_input_visk", 2024, MENAGERIE_ZOO),
]
