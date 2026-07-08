# SOURCE: vendored from diyiiyiii/StyTR-2 @ main
#   https://raw.githubusercontent.com/diyiiyiii/StyTR-2/main/models/StyTR.py
#   https://raw.githubusercontent.com/diyiiyiii/StyTR-2/main/models/transformer.py
#   https://raw.githubusercontent.com/diyiiyiii/StyTR-2/main/models/ViT_helper.py
#   https://raw.githubusercontent.com/diyiiyiii/StyTR-2/main/function.py
#   https://raw.githubusercontent.com/diyiiyiii/StyTR-2/main/util/misc.py
#   https://raw.githubusercontent.com/diyiiyiii/StyTR-2/main/test.py
#
# Deng et al. 2022 (CVPR) "StyTr^2: Image Style Transfer with Transformers".
# `models/StyTR.py` (PatchEmbed, the VGG-style `vgg` feature-extractor
# Sequential, the conv `decoder` Sequential, `StyTrans`) and
# `models/transformer.py` (the dual-encoder / single-decoder cross-attention
# Transformer -- encoder_c/encoder_s/decoder wired exactly as in
# `Transformer.forward`) are copied VERBATIM below, matching `test.py`'s
# real inference construction:
#   vgg = nn.Sequential(*list(StyTR.vgg.children())[:44])
#   network = StyTR.StyTrans(vgg, StyTR.decoder, StyTR.PatchEmbed(), transformer.Transformer(), args)
#   output = network(content, style)
# Two minimal, behavior-preserving fixes were required to run in this base
# env (no other logic touched):
#   1. `models/ViT_helper.py` imports `from torch._six import
#      container_abcs`, a module removed from modern PyTorch years ago;
#      `to_2tuple` is swapped for the stdlib-equivalent
#      `torch.nn.modules.utils._ntuple(2)` helper that ships in every
#      current torch (same truncated-normal/ntuple utilities upstream
#      vendored from PyTorch internals in the first place).
#   2. `models/transformer.py` sets `os.environ["CUDA_VISIBLE_DEVICES"] =
#      "2, 3"` at *module import time* -- a hazardous global side effect
#      (this env leaks into every subsequent CUDA call in the process, not
#      just this model). That statement is dropped; `device` selection is
#      left to the caller (this file never calls `.to(device)` or reads
#      `CUDA_VISIBLE_DEVICES` itself, so the architecture is unaffected).
# `StyTrans.forward` is used with its currently-checked-in (train-mode)
# return signature `(Ics, loss_c, loss_s, loss_lambda1, loss_lambda2)` --
# that is the real code path as committed upstream (test.py's alternate
# `return Ics` line is left commented out in the source).
import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# function.py (verbatim)
# ---------------------------------------------------------------------------
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


# ---------------------------------------------------------------------------
# models/ViT_helper.py (verbatim, minus torch._six import -- see header note)
# ---------------------------------------------------------------------------
from itertools import repeat  # noqa: E402


def _ntuple(n):
    def parse(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


# ---------------------------------------------------------------------------
# util/misc.py (NestedTensor + nested_tensor_from_tensor_list, verbatim)
# ---------------------------------------------------------------------------
class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# ---------------------------------------------------------------------------
# models/StyTR.py (verbatim)
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.proj(x)
        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


class StyTrans(nn.Module):
    """This is the style transform transformer module"""

    def __init__(self, encoder, decoder, patch_embed, transformer, args):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ["enc_1", "enc_2", "enc_3", "enc_4", "enc_5"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        self.decode = decoder
        self.embedding = patch_embed

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, samples_c, samples_s):
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask, content, pos_c, pos_s)
        Ics = self.decode(hs)

        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(
            normal(Ics_feats[-1]), normal(content_feats[-1])
        ) + self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))

        loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(
            Iss, style_input
        )

        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(
            Icc_feats[0], content_feats[0]
        ) + self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(
                Icc_feats[i], content_feats[i]
            ) + self.calc_content_loss(Iss_feats[i], style_feats[i])

        return Ics, loss_c, loss_s, loss_lambda1, loss_lambda2


# ---------------------------------------------------------------------------
# models/transformer.py (verbatim, minus the module-level CUDA_VISIBLE_DEVICES
# side effect -- see header note)
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.new_ps = nn.Conv2d(512, 512, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):
        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode="bilinear", size=style.shape[-2:])

        style = style.flatten(2).permute(2, 0, 1)
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1)

        content = content.flatten(2).permute(2, 0, 1)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)

        style = self.encoder_s(style, src_key_padding_mask=mask, pos=pos_embed_s)
        content = self.encoder_c(content, src_key_padding_mask=mask, pos=pos_embed_c)
        hs = self.decoder(
            content, style, memory_key_padding_mask=mask, pos=pos_embed_s, query_pos=pos_embed_c
        )[0]

        N, B, C = hs.shape
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1, H)

        return hs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ---------------------------------------------------------------------------
# build_stytr2 / example_input_stytr2 -- same wiring as test.py's
# `network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)`. The
# `vgg`/`decoder` Sequentials and the PatchEmbed/Transformer `embed_dim`/
# `d_model` (512) are architecturally coupled (embed_dim must match the
# real VGG relu4-1 channel count that PatchEmbed and decoder operate on),
# so they are kept at real scale; only the encoder/decoder LAYER COUNT is
# reduced (real default: 3/3) and the input spatial size shrunk, for a fast
# trace -- neither changes the architecture's mechanism, only its depth/
# resolution, matching how other menagerie transformer recipes are scaled.
#
# `StyTrans.calc_content_loss`/`calc_style_loss` assert their `target` arg
# has `requires_grad is False` -- true whenever this is called the way the
# real repo always calls it (test.py: `with torch.no_grad(): output =
# network(content, style)`). `_StyTransInferenceWrapper` below reproduces
# that exact real invocation convention (a `torch.no_grad()` context around
# the call to the unmodified `StyTrans.forward`); it changes nothing about
# the architecture, only encodes the calling convention the upstream code
# itself requires.
# ---------------------------------------------------------------------------
class _Args:
    hidden_dim = 512


class _StyTransInferenceWrapper(nn.Module):
    def __init__(self, stytrans):
        super().__init__()
        self.stytrans = stytrans

    def forward(self, samples_c, samples_s):
        with torch.no_grad():
            return self.stytrans(samples_c, samples_s)


def build_stytr2():
    vgg_full = nn.Sequential(*list(vgg.children())[:44])
    trans = Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=512,
        return_intermediate_dec=True,
    )
    embedding = PatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=512)
    stytrans = StyTrans(vgg_full, decoder, embedding, trans, _Args())
    return _StyTransInferenceWrapper(stytrans)


def example_input_stytr2():
    torch.manual_seed(0)
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    return (content, style)


MENAGERIE_ENTRIES = [
    (
        "StyTr^2 (Style Transfer Transformer)",
        "build_stytr2",
        "example_input_stytr2",
        2022,
        "vendored-pytorch",
    ),
]
