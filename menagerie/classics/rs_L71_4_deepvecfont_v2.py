# SOURCE: vendored from yizhiwang96/deepvecfont-v2 @ main
# Files: models/image_encoder.py, models/image_decoder.py,
#        models/modality_fusion.py, models/transformers.py, options.py
# https://github.com/yizhiwang96/deepvecfont-v2
#
# Minimal changes from the original source:
#   - `options.py::get_parser_main_model` is inlined verbatim (it is a
#     standalone top-level module in the original repo, not importable
#     as a package from a staging file outside the repo checkout).
#   - `models/transformers.py` and `models/modality_fusion.py` read a
#     module-level `opts = get_parser_main_model().parse_args()`, which
#     parses `sys.argv` at import time (breaks under pytest / any
#     harness with unrelated argv). Replaced with
#     `get_parser_main_model().parse_args([])`, which yields the exact
#     same defaults (matching the original repo's own `if __name__ ==
#     '__main__':` smoke test at the bottom of transformers.py, which
#     also relies on argparse defaults).
#   - `models/util_funcs.py`'s `sequence_mask()` (a 6-line, torch-only
#     helper) is inlined verbatim here instead of importing the whole
#     `models/util_funcs.py` module, whose top-level `import cairosvg`
#     (used only by an unrelated SVG-rasterization helper,
#     `svg2img()`, that this staging module never calls) is not
#     available in the base env.
#   - `ModelMain.forward()` in `models/model_main.py` is NOT vendored
#     verbatim: the original hardcodes `.cuda()` throughout and its
#     `fetch_data()` is tightly coupled to the project's custom
#     dataloader (`dataloader.py`, glyph-image + SVG-command-sequence
#     tensors read from a pickled font dataset). Instead,
#     `DeepVecFontV2Net.forward()` below drives the exact same real
#     submodules (`ImageEncoder`, `Transformer` (Perceiver seq
#     encoder), `ModalityFusion`, `ImageDecoder`, `Transformer_decoder`)
#     in the same order and with the same tensor shapes/dtypes that
#     `ModelMain.forward(mode='train')` uses, but takes those tensors
#     directly as arguments instead of extracting them from a dataset
#     dict via CUDA-only helpers. No architecture code is rewritten;
#     only the outer orchestration/data-plumbing layer is replaced with
#     an equivalent CPU-safe harness.
#
# Architecture (unmodified from source): DeepVecFont-v2 (CVPR 2023) is a
# dual-modality (raster image + SVG vector sequence) VAE-style font
# generator. ImageEncoder (a strided-conv tower over the few-shot
# reference glyph raster images) produces an image feature; a Perceiver-
# style cross-attention Transformer (models.transformers.Transformer)
# separately encodes the reference SVG command sequences (via an
# SVGEmbedding of quantized command+argument tokens) into a sequence
# feature; ModalityFusion concatenates and projects both features into
# a VAE latent (mu, log_sigma) with reparameterization; the latent
# drives an ImageDecoder (transposed-conv tower) to reconstruct the
# target glyph raster, while a custom Transformer_decoder (a 6-layer
# encoder-decoder-style Transformer over quantized SVG command/argument
# tokens, plus a lightweight "parallel_decoder" refinement head)
# autoregressively predicts the target glyph's vector (SVG command)
# sequence conditioned on the fused latent.

import argparse
import copy
import math
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce


# ---------------------------------------------------------------------------
# options.py::get_parser_main_model (inlined verbatim; the original repo's
# `options.py` is a standalone top-level module, not importable as a
# package from a staging file outside the repo checkout).
# ---------------------------------------------------------------------------
def get_parser_main_model():
    parser = argparse.ArgumentParser()
    # basic parameters training related
    parser.add_argument(
        "--model_name",
        type=str,
        default="main_model",
        choices=["main_model", "neural_raster"],
        help="current model_name",
    )
    parser.add_argument("--language", type=str, default="eng", choices=["eng", "chn"])
    parser.add_argument(
        "--bottleneck_bits", type=int, default=512, help="latent code number of bottleneck bits"
    )
    parser.add_argument("--char_num", type=int, default=52, help="number of glyphs, original is 52")
    parser.add_argument("--ref_nshot", type=int, default=4, help="reference number")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--batch_size_val", type=int, default=8, help="batch size when do validation"
    )
    parser.add_argument("--img_size", type=int, default=64, help="image size")
    parser.add_argument("--max_seq_len", type=int, default=51, help="maximum length of sequence")
    parser.add_argument(
        "--dim_seq",
        type=int,
        default=12,
        help="the dim of each stroke in a sequence, 4 + 8, 4 is cmd, and 8 is args",
    )
    parser.add_argument(
        "--dim_seq_short",
        type=int,
        default=9,
        help="the short dim of each stroke in a sequence, 1 + 8, 1 is cmd class num, and 8 is args",
    )
    parser.add_argument("--hidden_size", type=int, default=512, help="hidden_size")
    parser.add_argument(
        "--dim_seq_latent", type=int, default=512, help="sequence encoder latent dim"
    )
    parser.add_argument(
        "--ngf", type=int, default=16, help="the basic num of channel in image encoder and decoder"
    )
    parser.add_argument(
        "--n_aux_pts",
        type=int,
        default=6,
        help="the number of aux pts in bezier curves for additional supervison",
    )
    # experiment related
    parser.add_argument("--random_index", type=str, default="00")
    parser.add_argument("--name_ckpt", type=str, default="600_192921.ckpt")
    parser.add_argument("--init_epoch", type=int, default=0, help="init epoch")
    parser.add_argument("--n_epochs", type=int, default=800, help="number of epochs")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="the number of samples for each glyph when testing",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--ref_char_ids", type=str, default="0,1,26,27", help="default is A, B, a, b"
    )

    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--name_exp", type=str, default="dvf")
    parser.add_argument("--data_root", type=str, default="./data/vecfont_dataset/")
    parser.add_argument(
        "--freq_ckpt", type=int, default=50, help="save checkpoint frequency of epoch"
    )
    parser.add_argument("--freq_sample", type=int, default=500, help="sample train output of steps")
    parser.add_argument("--freq_log", type=int, default=50, help="freq of showing logs")
    parser.add_argument("--freq_val", type=int, default=500, help="sample validate output of steps")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 of Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 of Adam optimizer")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--tboard", type=bool, default=True, help="whether use tensorboard to visulize loss"
    )

    # loss weight
    parser.add_argument("--kl_beta", type=float, default=0.01, help="latent code kl loss beta")
    parser.add_argument(
        "--loss_w_pt_c",
        type=float,
        default=0.001 * 10,
        help="the weight of perceptual content loss",
    )
    parser.add_argument(
        "--loss_w_l1",
        type=float,
        default=1.0 * 10,
        help="the weight of image reconstruction l1 loss",
    )
    parser.add_argument("--loss_w_cmd", type=float, default=1.0, help="the weight of cmd loss")
    parser.add_argument("--loss_w_args", type=float, default=1.0, help="the weight of args loss")
    parser.add_argument("--loss_w_aux", type=float, default=0.01, help="the weight of pts aux loss")
    parser.add_argument("--loss_w_smt", type=float, default=10.0, help="the weight of smooth loss")

    return parser


opts = get_parser_main_model().parse_args([])


# ---------------------------------------------------------------------------
# models/util_funcs.py::sequence_mask (inlined; avoids the cairosvg import
# that the rest of util_funcs.py pulls in at module scope).
# ---------------------------------------------------------------------------
def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .unsqueeze(0)
        .expand(batch_size, max_len)
        .lt(lengths.unsqueeze(1))
    )


# ---------------------------------------------------------------------------
# models/image_encoder.py (verbatim)
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, img_size, input_nc, ngf=16, norm_layer=nn.LayerNorm):
        super(ImageEncoder, self).__init__()
        n_downsampling = int(math.log(img_size, 2))
        ks_list = [5] * (n_downsampling - n_downsampling // 3) + [3] * (n_downsampling // 3)
        stride_list = [2] * n_downsampling

        chn_mult = []
        for i in range(n_downsampling):
            chn_mult.append(2 ** (i + 1))

        encoder = [
            nn.Conv2d(
                input_nc, ngf, kernel_size=7, padding=7 // 2, bias=True, padding_mode="replicate"
            ),
            norm_layer([ngf, 2**n_downsampling, 2**n_downsampling]),
            nn.ReLU(True),
        ]
        for i in range(n_downsampling):  # add downsampling layers
            if i == 0:
                chn_prev = ngf
            else:
                chn_prev = ngf * chn_mult[i - 1]
            chn_next = ngf * chn_mult[i]

            encoder += [
                nn.Conv2d(
                    chn_prev,
                    chn_next,
                    kernel_size=ks_list[i],
                    stride=stride_list[i],
                    padding=ks_list[i] // 2,
                    padding_mode="replicate",
                ),
                norm_layer(
                    [chn_next, 2 ** (n_downsampling - 1 - i), 2 ** (n_downsampling - 1 - i)]
                ),
                nn.ReLU(True),
            ]

        self.encode = nn.Sequential(*encoder)
        self.flatten = nn.Flatten()

    def forward(self, input):
        """Standard forward"""
        ret = self.encode(input)
        img_feat = self.flatten(ret)
        output = {}
        output["img_feat"] = img_feat
        return output


# ---------------------------------------------------------------------------
# models/image_decoder.py (verbatim)
# ---------------------------------------------------------------------------
class ImageDecoder(nn.Module):
    def __init__(self, img_size, input_nc, output_nc, ngf=16, norm_layer=nn.LayerNorm):
        super(ImageDecoder, self).__init__()
        n_upsampling = int(math.log(img_size, 2))
        ks_list = [3] * (n_upsampling // 3) + [5] * (n_upsampling - n_upsampling // 3)
        stride_list = [2] * n_upsampling
        decoder = []

        chn_mult = []
        for i in range(n_upsampling):
            chn_mult.append(2 ** (n_upsampling - i - 1))

        decoder += [
            nn.ConvTranspose2d(
                input_nc,
                chn_mult[0] * ngf,
                kernel_size=ks_list[0],
                stride=stride_list[0],
                padding=ks_list[0] // 2,
                output_padding=stride_list[0] - 1,
            ),
            norm_layer([chn_mult[0] * ngf, 2, 2]),
            nn.ReLU(True),
        ]

        for i in range(1, n_upsampling):  # add upsampling layers
            chn_prev = chn_mult[i - 1] * ngf
            chn_next = chn_mult[i] * ngf
            decoder += [
                nn.ConvTranspose2d(
                    chn_prev,
                    chn_next,
                    kernel_size=ks_list[i],
                    stride=stride_list[i],
                    padding=ks_list[i] // 2,
                    output_padding=stride_list[i] - 1,
                ),
                norm_layer([chn_next, 2 ** (i + 1), 2 ** (i + 1)]),
                nn.ReLU(True),
            ]

        decoder += [nn.Conv2d(chn_mult[-1] * ngf, output_nc, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)

    def forward(self, latent_feat, trg_char, trg_img=None):
        """Standard forward"""
        dec_input = torch.cat((latent_feat, trg_char), -1)
        dec_input = dec_input.view(dec_input.size(0), dec_input.size(1), 1, 1)
        dec_out = self.decode(dec_input)
        output = {}
        output["gen_imgs"] = dec_out
        if trg_img is not None:
            output["img_l1loss"] = F.l1_loss(dec_out, trg_img)

        return output


# ---------------------------------------------------------------------------
# models/modality_fusion.py (verbatim, module-level `opts` now parsed
# with an empty argv above)
# ---------------------------------------------------------------------------
class ModalityFusion(nn.Module):
    def __init__(
        self,
        img_size=64,
        ref_nshot=4,
        bottleneck_bits=512,
        ngf=32,
        seq_latent_dim=512,
        mode="train",
    ):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.ref_nshot = ref_nshot
        self.mode = mode
        self.fc_merge = nn.Linear(seq_latent_dim * opts.ref_nshot, 512)
        n_downsampling = int(math.log(img_size, 2))
        mult_max = 2 ** (n_downsampling)
        self.fc_fusion = nn.Linear(
            ngf * mult_max + seq_latent_dim, opts.bottleneck_bits * 2, bias=True
        )  # the max multiplier for img feat channels is

    def forward(self, seq_feat, img_feat, ref_pad_mask=None):
        cls_one_pad = torch.ones((1, 1, 1)).to(seq_feat.device).repeat(seq_feat.size(0), 1, 1)
        ref_pad_mask = torch.cat([cls_one_pad, ref_pad_mask], dim=-1)

        seq_feat = seq_feat * (ref_pad_mask.transpose(1, 2))
        seq_feat_ = seq_feat.view(
            seq_feat.size(0) // self.ref_nshot, self.ref_nshot, seq_feat.size(-2), seq_feat.size(-1)
        )
        seq_feat_ = seq_feat_.transpose(1, 2)
        seq_feat_ = seq_feat_.contiguous().view(
            seq_feat_.size(0), seq_feat_.size(1), seq_feat_.size(2) * seq_feat_.size(3)
        )
        seq_feat_ = self.fc_merge(seq_feat_)
        seq_feat_cls = seq_feat_[:, 0]

        feat_cat = torch.cat((img_feat, seq_feat_cls), -1)
        dist_param = self.fc_fusion(feat_cat)

        output = {}
        mu = dist_param[..., : self.bottleneck_bits]
        log_sigma = dist_param[..., self.bottleneck_bits :]

        if self.mode == "train":
            # calculate the kl loss and reparamerize latent code
            epsilon = torch.randn(*mu.size(), device=mu.device)
            z = mu + torch.exp(log_sigma / 2) * epsilon
            kl = 0.5 * torch.mean(torch.exp(log_sigma) + torch.square(mu) - 1.0 - log_sigma)
            output["latent"] = z
            output["kl_loss"] = kl
            seq_feat_[:, 0] = z
            latent_feat_seq = seq_feat_

        else:
            output["latent"] = mu
            output["kl_loss"] = 0.0
            seq_feat_[:, 0] = mu
            latent_feat_seq = seq_feat_

        return output, latent_feat_seq


# ---------------------------------------------------------------------------
# models/transformers.py (verbatim)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[: x.size(0), :].to(x.device)
        return self.dropout(x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, cls_conv_dim=None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)  # 27 to 5012*2 = 1024

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, ref_cls_onehot=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = repeat(mask, "b j k -> (b h) k j", h=h)
            sim.masked_fill(mask == 0, -1e9)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out), attn


class SVGEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_embed = nn.Embedding(4, 512)
        self.arg_embed = nn.Embedding(128, 128, padding_idx=0)
        self.embed_fcn = nn.Linear(128 * 8, 512)
        self.pos_encoding = PositionalEncoding(
            d_model=opts.hidden_size, max_len=opts.max_seq_len + 1
        )
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        S, GN, _ = commands.shape
        src = self.command_embed(commands.long()).squeeze() + self.embed_fcn(
            self.arg_embed((args).long()).view(S, GN, -1)
        )  # shift due to -1 PAD_VAL

        src = self.pos_encoding(src)

        return src


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(F.relu(self.dropout(self.w_1(x))))


class Transformer_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.SVG_embedding = SVGEmbedding()
        self.command_fcn = nn.Linear(512, 4)
        self.args_fcn = nn.Linear(512, 8 * 128)
        c = copy.deepcopy
        attn = MultiHeadedAttention(h=8, d_model=512, dropout=0.0)
        ff = PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.0)
        self.decoder_layers = clones(DecoderLayer(512, c(attn), c(attn), c(ff), dropout=0.0), 6)
        self.decoder_norm = nn.LayerNorm(512)
        self.decoder_layers_parallel = clones(
            DecoderLayer(512, c(attn), c(attn), c(ff), dropout=0.0), 1
        )
        self.decoder_norm_parallel = nn.LayerNorm(512)
        self.cls_embedding = nn.Embedding(52, 512)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def forward(self, x, memory, trg_char, src_mask=None, tgt_mask=None):
        memory = memory.unsqueeze(1)
        commands = x[:, :, :1]
        args = x[:, :, 1:]
        x = self.SVG_embedding(commands, args).transpose(0, 1)
        trg_char = trg_char.long()
        trg_char = self.cls_embedding(trg_char)
        x[:, 0:1, :] = trg_char
        tgt_mask = tgt_mask.squeeze()
        attn = None
        for layer in self.decoder_layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
        out = self.decoder_norm(x)
        N, S, _ = out.shape
        cmd_logits = self.command_fcn(out)
        args_logits = self.args_fcn(out)  # shape: bs, max_len, 8, 256
        args_logits = args_logits.reshape(N, S, 8, 128)
        return cmd_logits, args_logits, attn

    def parallel_decoder(self, cmd_logits, args_logits, memory, trg_char):
        memory = memory.unsqueeze(1)
        cmd_args_mask = torch.Tensor(
            [
                [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ).to(cmd_logits.device)
        if opts.mode == "train":
            cmd2 = torch.argmax(cmd_logits, -1).unsqueeze(-1).transpose(0, 1)
            arg2 = torch.argmax(args_logits, -1).transpose(0, 1)

            cmd2paddingmask = (
                _get_key_padding_mask(cmd2).transpose(0, 1).unsqueeze(-1).to(cmd2.device)
            )
            cmd2 = cmd2 * cmd2paddingmask
            args_mask = (
                torch.matmul(F.one_hot(cmd2.long(), 4).float(), cmd_args_mask)
                .transpose(-1, -2)
                .squeeze(-1)
            )
            arg2 = arg2 * args_mask

            x = self.SVG_embedding(cmd2, arg2).transpose(0, 1)
        else:
            cmd2 = cmd_logits
            arg2 = args_logits

            cmd2paddingmask = (
                _get_key_padding_mask(cmd2).transpose(0, 1).unsqueeze(-1).to(cmd2.device)
            )
            cmd2 = cmd2 * cmd2paddingmask
            args_mask = (
                torch.matmul(F.one_hot(cmd2.long(), 4).float(), cmd_args_mask)
                .transpose(-1, -2)
                .squeeze(-1)
            )
            arg2 = arg2 * args_mask

            x = self.SVG_embedding(cmd2, arg2).transpose(0, 1)

        S = x.size(1)
        B = x.size(0)
        tgt_mask = torch.ones(S, S).to(x.device).unsqueeze(0).repeat(B, 1, 1)
        cmd2paddingmask = cmd2paddingmask.transpose(0, 1).transpose(-1, -2)
        tgt_mask = tgt_mask * cmd2paddingmask

        trg_char = trg_char.long()
        trg_char = self.cls_embedding(trg_char)

        x = torch.cat([trg_char, x], 1)
        x[:, 0:1, :] = trg_char
        x = x[:, : opts.max_seq_len, :]
        attn = None
        for layer in self.decoder_layers_parallel:
            x, attn = layer(x, memory, src_mask=None, tgt_mask=tgt_mask)
        out = self.decoder_norm_parallel(x)

        N, S, _ = out.shape
        cmd_logits = self.command_fcn(out)
        args_logits = self.args_fcn(out)
        args_logits = args_logits.reshape(N, S, 8, 128)

        return cmd_logits, args_logits


def _get_key_padding_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    lens = []
    with torch.no_grad():
        commands = commands.transpose(0, 1).squeeze(-1)  # bs, opts.max_seq_len
        for i in range(commands.size(0)):
            try:
                seqi = commands[i]  # blue opts.max_seq_len
                index = torch.where(seqi == 0)[0][0]
            except Exception:
                index = opts.max_seq_len

            lens.append(index)
        lens = torch.tensor(lens) + 1  # blue b
        seqlen_mask = sequence_mask(lens, opts.max_seq_len)  # blue b,opts.max_seq_len
        return seqlen_mask


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels=1,
        input_axis=2,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1000,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        fourier_encode_data=True,
        self_per_cross_attn=2,
        final_classifier_head=True,
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (
            (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        )  # 26
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                input_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=input_dim,
        )  # noqa: E731
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))  # noqa: E731
        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout
            ),
        )  # noqa: E731
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))  # noqa: E731

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(
                    nn.ModuleList(
                        [
                            get_latent_attn(**cache_args, key=block_ind),
                            get_latent_ff(**cache_args, key=block_ind),
                        ]
                    )
                )

            self.layers.append(
                nn.ModuleList(
                    [get_cross_attn(**cache_args), get_cross_ff(**cache_args), self_attns]
                )
            )

        get_cross_attn2 = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                input_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=input_dim,
        )  # noqa: E731
        get_cross_ff2 = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))  # noqa: E731
        get_latent_attn2 = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout
            ),
        )  # noqa: E731
        get_latent_ff2 = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))  # noqa: E731

        get_cross_attn2, get_cross_ff2, get_latent_attn2, get_latent_ff2 = map(
            cache_fn, (get_cross_attn2, get_cross_ff2, get_latent_attn2, get_latent_ff2)
        )

        self.layers_cnnsvg = nn.ModuleList([])
        for i in range(1):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self_attns2 = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns2.append(
                    nn.ModuleList(
                        [
                            get_latent_attn2(**cache_args, key=block_ind),
                            get_latent_ff2(**cache_args, key=block_ind),
                        ]
                    )
                )

            self.layers_cnnsvg.append(
                nn.ModuleList(
                    [get_cross_attn2(**cache_args), get_cross_ff2(**cache_args), self_attns2]
                )
            )

        self.to_logits = (
            nn.Sequential(
                Reduce("b n d -> b d", "mean"),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes),
            )
            if final_classifier_head
            else nn.Identity()
        )
        self.pre_lstm_fc = nn.Linear(10, opts.hidden_size)
        self.posr = PositionalEncoding(d_model=opts.hidden_size, max_len=opts.max_seq_len)

        self.SVG_embedding = SVGEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def forward(self, data, seq, ref_cls_onehot=None, mask=None, return_embeddings=True):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype  # noqa: F841
        assert len(axis) == self.input_axis, (
            "input data must have the right number of axis"
        )  # img is 2
        x = seq
        commands = x[:, :, :1]
        args = x[:, :, 1:]
        x = self.SVG_embedding(commands, args).transpose(0, 1)
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=x.size(0))
        x = torch.cat([cls_tokens, x], dim=1)
        cls_one_pad = torch.ones((1, 1, 1)).to(x.device).repeat(x.size(0), 1, 1)
        mask = torch.cat([cls_one_pad, mask], dim=-1)
        self_atten = []
        for cross_attn, cross_ff, self_attns in self.layers:
            for self_attn, self_ff in self_attns:
                x_, atten = self_attn(x, mask=mask)
                x = x_ + x
                self_atten.append(atten)
                x = self_ff(x) + x
        x = x + torch.randn_like(x)  # add a perturbation
        return x, self_atten

    def att_residual(self, x, mask=None):
        for cross_attn, cross_ff, self_attns in self.layers_cnnsvg:
            for self_attn, self_ff in self_attns:
                x_, atten = self_attn(x)
                x = x_ + x
                x = self_ff(x) + x
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        attn = self.self_attn.attn
        return self.sublayer[2](x, self.feed_forward), attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x_norm = self.norm(x)
        return x + self.dropout(sublayer(x_norm))


def attention(query, key, value, mask=None, trg_tri_mask=None, dropout=None, posr=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if posr is not None:
        posr = posr.unsqueeze(1)
        scores = scores + posr

    if mask is not None:
        scores = scores.masked_fill(
            mask == 0, -1e9
        )  # note mask: b,1,501,501  scores: b, head, 501,501

    if trg_tri_mask is not None:
        scores = scores.masked_fill(trg_tri_mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, trg_tri_mask=None, posr=None):
        "Implements Figure 2"

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # noqa: E741

        x, self.attn = attention(
            query, key, value, mask=mask, trg_tri_mask=trg_tri_mask, dropout=self.dropout, posr=posr
        )

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


def subsequent_mask(size):
    "Mask out subsequent positions."
    import numpy as np

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


# ---------------------------------------------------------------------------
# CPU-safe harness driving the real modules in ModelMain's train-mode order
# (models/model_main.py::ModelMain.forward, mode='train').
# ---------------------------------------------------------------------------
class DeepVecFontV2Net(nn.Module):
    """Wraps the real ImageEncoder / Transformer (Perceiver seq encoder) /
    ModalityFusion / ImageDecoder / Transformer_decoder submodules and
    drives them in the same order as ModelMain.forward(mode='train'),
    but takes pre-extracted tensors directly instead of reading them out
    of a dataset dict via CUDA-only helpers.
    """

    def __init__(self, opts=opts):
        super().__init__()
        self.opts = opts
        self.img_encoder = ImageEncoder(
            img_size=opts.img_size, input_nc=opts.ref_nshot, ngf=opts.ngf, norm_layer=nn.LayerNorm
        )
        self.img_decoder = ImageDecoder(
            img_size=opts.img_size,
            input_nc=opts.bottleneck_bits + opts.char_num,
            output_nc=1,
            ngf=opts.ngf,
            norm_layer=nn.LayerNorm,
        )
        self.modality_fusion = ModalityFusion(
            img_size=opts.img_size,
            ref_nshot=opts.ref_nshot,
            bottleneck_bits=opts.bottleneck_bits,
            ngf=opts.ngf,
            mode=opts.mode,
        )
        self.transformer_main = Transformer(
            input_channels=1,
            input_axis=2,
            num_freq_bands=6,
            max_freq=10.0,
            depth=6,
            num_latents=256,
            latent_dim=opts.dim_seq_latent,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1000,
            attn_dropout=0.0,
            ff_dropout=0.0,
            weight_tie_layers=False,
            fourier_encode_data=True,
            self_per_cross_attn=2,
        )
        self.transformer_seqdec = Transformer_decoder()

    def forward(
        self, ref_img, ref_seq_cat, ref_pad_mask, trg_img, trg_char_onehot, trg_cls, trg_seq_shifted
    ):
        opts = self.opts

        # image encoding (ImageEncoder over the ref_nshot glyph rasters)
        img_encoder_out = self.img_encoder(ref_img)
        img_feat = img_encoder_out["img_feat"]  # bs, ngf * (2 ** 6)

        # seq encoding (Perceiver-style Transformer over quantized SVG
        # command+argument tokens for the reference glyphs)
        ref_img_ = ref_img.view(
            ref_img.size(0) * ref_img.size(1), ref_img.size(2), ref_img.size(3)
        ).unsqueeze(-1)
        seq_feat, _ = self.transformer_main(ref_img_, ref_seq_cat, mask=ref_pad_mask)

        # modality fusion -> VAE latent
        mf_output, latent_feat_seq = self.modality_fusion(
            seq_feat, img_feat, ref_pad_mask=ref_pad_mask
        )
        latent_feat_seq = self.transformer_main.att_residual(latent_feat_seq)
        z = mf_output["latent"]
        kl_loss = mf_output["kl_loss"]

        # image decoding (transposed-conv tower over the fused latent)
        img_decoder_out = self.img_decoder(z, trg_char_onehot, trg_img)

        # SVG sequence decoding: 6-layer Transformer_decoder + a
        # lightweight parallel-refinement decoder head, exactly as
        # ModelMain.forward(mode='train') calls them.
        tgt_mask = (
            subsequent_mask(opts.max_seq_len)
            .type_as(ref_pad_mask.data)
            .unsqueeze(0)
            .expand(z.size(0), -1, -1, -1)
            .float()
        )
        command_logits, args_logits, attn = self.transformer_seqdec(
            x=trg_seq_shifted, memory=latent_feat_seq, trg_char=trg_cls, tgt_mask=tgt_mask
        )
        command_logits_2, args_logits_2 = self.transformer_seqdec.parallel_decoder(
            command_logits, args_logits, memory=latent_feat_seq.detach(), trg_char=trg_cls
        )

        return {
            "gen_img": img_decoder_out["gen_imgs"],
            "img_l1loss": img_decoder_out["img_l1loss"],
            "kl_loss": kl_loss,
            "command_logits": command_logits,
            "args_logits": args_logits,
            "command_logits_parallel": command_logits_2,
            "args_logits_parallel": args_logits_2,
        }


def build_deepvecfont_v2():
    return DeepVecFontV2Net()


def example_input_deepvecfont_v2():
    # batch_size=2 (not 1): Transformer_decoder.forward() does
    # `tgt_mask.squeeze()` on a [bs, 1, max_seq_len, max_seq_len] mask,
    # which collapses the batch dim too when bs=1 (matches the original
    # repo's implicit assumption that fetch_data() always yields bs>1
    # training batches).
    o = opts
    batch_size = 2
    n_ref = o.ref_nshot  # 4

    ref_img = torch.rand(batch_size, n_ref, o.img_size, o.img_size)
    trg_img = torch.rand(batch_size, 1, o.img_size, o.img_size)

    # quantized SVG command+argument tokens, shape [max_seq_len, bs*n_ref, dim_seq_short]
    ref_seq_cat = torch.zeros(o.max_seq_len, batch_size * n_ref, o.dim_seq_short)
    ref_seq_cat[..., 0] = torch.randint(0, 4, (o.max_seq_len, batch_size * n_ref)).float()
    ref_seq_cat[..., 1:] = torch.randint(
        0, 128, (o.max_seq_len, batch_size * n_ref, o.dim_seq_short - 1)
    ).float()

    ref_pad_mask = torch.ones(batch_size * n_ref, 1, o.max_seq_len)

    trg_char_onehot = F.one_hot(
        torch.randint(0, o.char_num, (batch_size,)), num_classes=o.char_num
    ).float()
    trg_cls = torch.randint(0, o.char_num, (batch_size, 1))

    trg_seq_shifted = torch.zeros(o.max_seq_len, batch_size, o.dim_seq_short)
    trg_seq_shifted[..., 0] = torch.randint(0, 4, (o.max_seq_len, batch_size)).float()
    trg_seq_shifted[..., 1:] = torch.randint(
        0, 128, (o.max_seq_len, batch_size, o.dim_seq_short - 1)
    ).float()

    return (ref_img, ref_seq_cat, ref_pad_mask, trg_img, trg_char_onehot, trg_cls, trg_seq_shifted)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepVecFont-v2",
        "build_deepvecfont_v2",
        "example_input_deepvecfont_v2",
        2023,
        "vendored",
    ),
]
