# SOURCE: vendored from Mael-zys/T2M-GPT @ main
# (models/vqvae.py, models/encdec.py, models/resnet.py, models/quantize_cnn.py,
# models/t2m_trans.py, models/pos_encoding.py -- verbatim, self-contained modules)
# https://github.com/Mael-zys/T2M-GPT -- "T2M-GPT: Generating Human Motion from
# Textual Descriptions with Discrete Representations" (Zhang et al., CVPR 2023,
# arXiv:2301.06052). Two novel stages, both vendored below and both traced: (1)
# `HumanVQVAE` (models/vqvae.py) -- a 1D-convolutional ResNet VQ-VAE that encodes
# motion sequences (joint positions/rotations over time) into a discrete codebook,
# and (2) `Text2Motion_Transformer` (models/t2m_trans.py) -- a GPT-style causal
# transformer (`CrossCondTransBase` + `CrossCondTransHead`, models/t2m_trans.py) that
# autoregressively predicts VQ-VAE code indices conditioned on a CLIP text embedding
# prepended as the first token. Every class body below is transcribed verbatim from
# the real repo files; the only change is stripping the training-time `.cuda()` calls
# hardcoded into `QuantizeEMAReset.reset_codebook`/`QuantizeEMA.reset_codebook`
# (models/quantize_cnn.py) so the codebook buffer registers on the tracing device
# instead of forcing CUDA -- purely a device-placement fix for CPU tracing, not an
# architectural change (the quantizer forward path and every layer are unmodified).
# Package-relative imports (`models.resnet`, `models.pos_encoding`, `models.encdec`,
# `models.quantize_cnn`) were flattened into this single module.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from models/resnet.py ----
class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(
            n_state,
            n_in,
            1,
            1,
            0,
        )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in, n_in, dilation=dilation_growth_rate**depth, activation=activation, norm=norm
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ---- verbatim from models/encdec.py ----
class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ---- verbatim from models/quantize_cnn.py (QuantizeEMAReset only -- the quantizer
# variant HumanVQVAE actually instantiates when args.quantizer == "ema_reset", T2M-GPT's
# default/paper config). `.cuda()` stripped from `reset_codebook`'s buffer registration
# -- see module header. ----
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / (code_dim**0.5)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[: self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(
            self.nb_code, code_idx.shape[0], device=code_idx.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[: self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(
            self.nb_code, 1
        )

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity


# ---- verbatim from models/vqvae.py (constructor args flattened from `args` namespace
# into explicit kwargs -- same defaults as the repo's option_vq.py) ----
class VQVAE_251(nn.Module):
    def __init__(
        self,
        nb_code=1024,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        input_width=263,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.encoder = Encoder(
            input_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            input_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.quantizer = QuantizeEMAReset(nb_code, code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(
        self,
        nb_code=512,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        input_width=263,
    ):
        super().__init__()

        self.nb_joints = 22
        self.vqvae = VQVAE_251(
            nb_code,
            code_dim,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
            input_width=input_width,
        )

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x)  # (N, T)
        return quants

    def forward(self, x):
        x_out, loss, perplexity = self.vqvae(x)

        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out


# ---- verbatim from models/pos_encoding.py ----
def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(dim)
        )
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """

    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x.shape: bs, seq_len, feat_dim
        seq_len = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[:seq_len].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


# ---- verbatim from models/t2m_trans.py ----
class Text2Motion_Transformer(nn.Module):
    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        clip_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()
        self.trans_base = CrossCondTransBase(
            num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate
        )
        self.trans_head = CrossCondTransHead(
            num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate
        )
        self.block_size = block_size
        self.num_vq = num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature):
        feat = self.trans_base(idxs, clip_feature)
        logits = self.trans_head(feat)
        return logits

    def sample(self, clip_feature, if_categorial=False):
        for k in range(self.block_size):
            if k == 0:
                x = []
            else:
                x = xs  # noqa: F821 -- verbatim upstream quirk (xs bound in prior loop iteration)
            logits = self.forward(x, clip_feature)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)

            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs


class CausalCrossConditionalSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(
            embed_dim, block_size, n_head, drop_out_rate
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):
    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        clip_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ]
        )
        self.pos_embed = PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, clip_feature):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            token_embeddings = torch.cat(
                [self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1
            )

        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):
    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_t2m_gpt_vqvae():
    torch.manual_seed(0)
    return HumanVQVAE(
        nb_code=64,
        code_dim=32,
        output_emb_width=32,
        down_t=2,
        stride_t=2,
        width=32,
        depth=2,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        input_width=48,
    )


def example_input_t2m_gpt_vqvae():
    torch.manual_seed(0)
    batch_size, seq_len, feat_dim = 2, 16, 48
    return (torch.randn(batch_size, seq_len, feat_dim),)


def build_t2m_gpt_transformer():
    torch.manual_seed(0)
    return Text2Motion_Transformer(
        num_vq=64,
        embed_dim=32,
        clip_dim=32,
        block_size=8,
        num_layers=2,
        n_head=4,
        drop_out_rate=0.0,
        fc_rate=4,
    )


def example_input_t2m_gpt_transformer():
    torch.manual_seed(0)
    batch_size, seq_len = 2, 4
    idxs = torch.randint(0, 64, (batch_size, seq_len))
    clip_feature = torch.randn(batch_size, 32)
    return (idxs, clip_feature)


MENAGERIE_ENTRIES = [
    ("T2M-GPT-VQVAE", build_t2m_gpt_vqvae, example_input_t2m_gpt_vqvae, 2023, "vendored-pytorch"),
    (
        "T2M-GPT-Transformer",
        build_t2m_gpt_transformer,
        example_input_t2m_gpt_transformer,
        2023,
        "vendored-pytorch",
    ),
]
