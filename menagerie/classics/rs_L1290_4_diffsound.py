# SOURCE: vendored from https://github.com/yangdongchao/Text-to-sound-Synthesis @ master
#   Vendored files: Diffsound/sound_synthesis/modeling/transformers/transformer_utils.py
#   (FullAttention, CrossAttention, GELU2, SinusoidalPosEmb, AdaLayerNorm, AdaInsNorm,
#   Block, Conv_MLP, Text2ImageTransformer -- the `Condition2ImageTransformer` and
#   `UnCondition2ImageTransformer` variants are omitted, they are unused siblings built
#   from the same vendored `Block`) and
#   Diffsound/sound_synthesis/modeling/embeddings/dalle_mask_image_embedding.py
#   (DalleMaskImageEmbedding + its BaseEmbedding parent from base_embedding.py).
#   The ONLY functional change: `Text2ImageTransformer.forward`/`Block.forward` called
#   `t.cuda()` unconditionally on the diffusion-timestep tensor (hardcoding a CUDA
#   device); this is a portability fix (matches the "fix only imports/relative-paths
#   minimally" rung-2 allowance), not an architectural change -- we drop `.cuda()` and
#   pass `t` through as-is so the model runs on whatever device its parameters are on.
#   `content_emb` is constructed directly (DalleMaskImageEmbedding(...)) instead of via
#   the repo's `instantiate_from_config(config)` YAML-dict indirection, since that
#   indirection is training-config plumbing, not part of the architecture.
#
# Diffsound (Yang, Yu, et al., "Diffsound: Discrete Diffusion Model for Text-to-sound
# Generation", 2022) generates audio spectrogram-codebook tokens from CLIP text
# embeddings via a VQ-Diffusion-style (Gu et al. 2022) discrete denoising diffusion
# transformer. `Text2ImageTransformer` (named for its VQ-Diffusion lineage; here it
# denoises SpecVQGAN spectrogram tokens conditioned on CLIP text features rather than
# image tokens) is the trainable denoising network at the heart of the pipeline: a
# masked-token content embedding (`DalleMaskImageEmbedding`, adding learned height/width
# positional embeddings) is refined through `n_layer` transformer blocks that alternate
# AdaLayerNorm-conditioned (on the diffusion timestep) self-attention with cross-
# attention to the text-condition sequence, before a final LayerNorm+Linear head
# predicts per-token logits over the codebook. We trace `Text2ImageTransformer.forward`
# with a small 2-layer / 64-dim / 4-head config, a short content sequence, and a short
# random condition (text) embedding sequence.

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- from modeling/embeddings/base_embedding.py, unmodified ----
class BaseEmbedding(nn.Module):
    def get_loss(self):
        return None

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            super().train()
        return self

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()


# ---- from modeling/embeddings/dalle_mask_image_embedding.py, unmodified ----
class DalleMaskImageEmbedding(BaseEmbedding):
    def __init__(
        self,
        num_embed=8192,
        spatial_size=[32, 32],
        embed_dim=3968,
        trainable=True,
        pos_emb_type="embedding",
    ):
        super().__init__()

        if isinstance(spatial_size, int):
            spatial_size = [spatial_size, spatial_size]

        self.spatial_size = spatial_size
        self.num_embed = num_embed + 1  # add a mask token
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type

        assert self.pos_emb_type in ["embedding", "parameter"]

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        if self.pos_emb_type == "embedding":
            self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim)
            self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim)
        else:
            self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim))
            self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim))

        self._set_trainable()

    def forward(self, index, **kwargs):
        assert index.dim() == 2  # B x L
        try:
            index[index < 0] = 0
            emb = self.emb(index)
        except Exception:
            raise RuntimeError(
                "IndexError: index out of range in self, max index {}, num embed {}".format(
                    index.max(), self.num_embed
                )
            )

        if emb.shape[1] > 0:
            if self.pos_emb_type == "embedding":
                height_emb = self.height_emb(
                    torch.arange(self.spatial_size[0], device=index.device).view(
                        1, self.spatial_size[0]
                    )
                ).unsqueeze(2)
                width_emb = self.width_emb(
                    torch.arange(self.spatial_size[1], device=index.device).view(
                        1, self.spatial_size[1]
                    )
                ).unsqueeze(1)
            else:
                height_emb = self.height_emb.unsqueeze(2)
                width_emb = self.width_emb.unsqueeze(1)
            pos_emb = (height_emb + width_emb).view(
                1, self.spatial_size[0] * self.spatial_size[1], -1
            )
            emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


# ---- from modeling/transformers/transformer_utils.py, unmodified except
# .cuda() -> passthrough on the diffusion timestep tensor `t` (see header) ----
class FullAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len=None, attn_pdrop=0.1, resid_pdrop=0.1, causal=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        condition_seq_len,
        n_embd,
        condition_embd,
        n_head,
        seq_len=None,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        if self.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
            )

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        condition_seq_len=77,
        n_embd=1024,
        n_head=16,
        seq_len=256,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
        attn_type="selfcross",
        content_spatial_size=None,
        condition_dim=1024,
        diffusion_step=100,
        timestep_type="adalayernorm",
        mlp_type="fc",
    ):
        super().__init__()
        self.attn_type = attn_type

        assert attn_type == "selfcross"
        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            condition_seq_len,
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        assert "adalayernorm" in timestep_type
        self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln2 = nn.LayerNorm(n_embd)

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, encoder_output, timestep, mask=None):
        a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a

        x = x + self.mlp(self.ln2(x))

        return x, att


class Text2ImageTransformer(nn.Module):
    def __init__(
        self,
        condition_seq_len=77,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        content_seq_len=1024,
        attn_pdrop=0,
        resid_pdrop=0,
        mlp_hidden_times=4,
        block_activate=None,
        attn_type="selfcross",
        content_spatial_size=[32, 32],
        condition_dim=512,
        diffusion_step=1000,
        timestep_type="adalayernorm",
        content_emb=None,
        mlp_type="fc",
        checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = checkpoint
        self.content_emb = content_emb

        assert attn_type == "selfcross"
        all_attn_type = [attn_type] * n_layer

        if content_spatial_size is None:
            s = int(math.sqrt(content_seq_len))
            assert s * s == content_seq_len
            content_spatial_size = (s, s)

        self.blocks = nn.Sequential(
            *[
                Block(
                    condition_seq_len,
                    n_embd=n_embd,
                    n_head=n_head,
                    seq_len=content_seq_len,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate if block_activate is not None else "GELU",
                    attn_type=all_attn_type[n],
                    content_spatial_size=content_spatial_size,
                    condition_dim=condition_dim,
                    diffusion_step=diffusion_step,
                    timestep_type=timestep_type,
                    mlp_type=mlp_type,
                )
                for n in range(n_layer)
            ]
        )

        out_cls = self.content_emb.num_embed - 1
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_cls),
        )

        self.condition_seq_len = condition_seq_len
        self.content_seq_len = content_seq_len

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input, cond_emb, t):
        cont_emb = self.content_emb(input)
        emb = cont_emb

        for block_idx in range(len(self.blocks)):
            # NOTE: original repo called `t.cuda()` here unconditionally (portability
            # fix -- see module header); we pass `t` through unmodified instead.
            emb, att_weight = self.blocks[block_idx](emb, cond_emb, t)
        logits = self.to_logits(emb)
        out = rearrange(logits, "b l c -> b c l")
        return out


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_diffsound():
    """Diffsound Text2ImageTransformer denoiser, small (2-layer, 64-dim, 4-head)
    config with a short 16-token content sequence and 512-dim text condition."""
    torch.manual_seed(0)
    content_emb = DalleMaskImageEmbedding(
        num_embed=64,
        spatial_size=[4, 4],  # 4*4 = 16 = content_seq_len
        embed_dim=64,
        trainable=True,
        pos_emb_type="embedding",
    )
    return Text2ImageTransformer(
        condition_seq_len=8,
        n_layer=2,
        n_embd=64,
        n_head=4,
        content_seq_len=16,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_hidden_times=2,
        block_activate="GELU",
        attn_type="selfcross",
        content_spatial_size=[4, 4],
        condition_dim=32,
        diffusion_step=100,
        timestep_type="adalayernorm",
        content_emb=content_emb,
        mlp_type="fc",
        checkpoint=False,
    )


def example_input_diffsound():
    torch.manual_seed(0)
    input_tokens = torch.randint(0, 64, (1, 16))  # B x L content token indices
    cond_emb = torch.randn(1, 8, 32)  # B x T_E x condition_dim text embedding
    t = torch.randint(0, 100, (1,))  # B diffusion timestep
    return (input_tokens, cond_emb, t)


MENAGERIE_ENTRIES = [
    (
        "Diffsound (Discrete Diffusion Text-to-Sound)",
        "build_diffsound",
        "example_input_diffsound",
        2022,
        "vendored-pytorch",
    ),
]
