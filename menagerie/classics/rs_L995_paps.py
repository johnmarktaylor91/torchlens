# SOURCE: vendored from VSainteuf/utae-paps @ 987874e27a98bb399277f2d55eb010744d26f881
# (src/backbones/positional_encoding.py, src/backbones/ltae.py, src/backbones/utae.py,
#  src/panoptic/paps.py)
#
# PaPs (Parcels-as-Points): panoptic segmentation of agricultural parcels from satellite
# image time series (ICCV 2021, Sainte Fare Garnot & Landrieu). The encoder backbone is
# U-TAE (a U-Net with a Lightweight Temporal Attention Encoder bottleneck) and the PaPs
# head does center-point detection + per-instance shape/size/class regression. All classes
# below (PositionalEncoder, LTAE2d/MultiHeadAttention/ScaledDotProductAttention, UTAE +
# supporting conv blocks, PaPs + CenterExtractor) are the REAL model code from the listed
# files, copied verbatim (only import paths adjusted to be self-contained). Only base-lib
# deps: torch, numpy, torch_scatter (installed).

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

# --- src/backbones/positional_encoding.py ---------------------------------


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)

        return sinusoid_table


# --- src/backbones/ltae.py -------------------------------------------------


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


# --- src/backbones/utae.py --------------------------------------------------


class UTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super(UTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode
        )

    def forward(self, input, batch_positions=None, return_att=False):
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # BxT pad mask
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if return_att:
                return out, att
            if self.return_maps:
                return out, maps
            else:
                return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(self.out_shape, device=input.device, requires_grad=False)
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(  # noqa: E731 (verbatim from source)
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(
                        attn
                    )
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(
                        attn
                    )
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


# --- src/panoptic/paps.py ----------------------------------------------------


class PaPs(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes=20,
        shape_size=16,
        mask_conv=True,
        min_confidence=0.2,
        min_remain=0.5,
        mask_threshold=0.4,
    ):
        """
        Implementation of the Parcel-as-Points Module (PaPs) for panoptic segmentation of agricultural
        parcels from satellite image time series.
        Args:
            encoder (nn.Module): Backbone encoding network. The encoder is expected to return
            a feature map at the same resolution as the input images and a list of feature maps
            of lower resolution.
            num_classes (int): Number of classes (including stuff and void classes).
            shape_size (int): S hyperparameter defining the shape of the local patch.
            mask_conv (bool): If False no residual CNN is applied after combination of
            the predicted shape and the cropped saliency (default True)
            min_confidence (float): Cut-off confidence level for the pseudo NMS (predicted instances with
            lower condidence will not be included in the panoptic prediction).
            min_remain (float): Hyperparameter of the pseudo-NMS that defines the fraction of a candidate instance mask
            that needs to be new to be included in the final panoptic prediction (default  0.5).
            mask_threshold (float): Binary threshold for instance masks (default 0.4)

        """
        super(PaPs, self).__init__()
        self.encoder = encoder
        self.shape_size = shape_size
        self.num_classes = num_classes
        self.min_scale = 1 / shape_size
        self.register_buffer("min_confidence", torch.tensor([min_confidence]))
        self.min_remain = min_remain
        self.mask_threshold = mask_threshold
        self.center_extractor = CenterExtractor()

        enc_dim = encoder.enc_dim
        stack_dim = encoder.stack_dim
        self.heatmap_conv = nn.Sequential(
            ConvLayer(nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

        self.saliency_conv = ConvLayer(
            nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1, padding_mode="reflect"
        )

        self.shape_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, shape_size**2),
        )

        self.size_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.BatchNorm1d(stack_dim // 4),
            nn.ReLU(),
            nn.Linear(stack_dim // 4, 2),
            nn.Softplus(),
        )

        self.class_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.Linear(stack_dim // 4, num_classes),
        )

        if mask_conv:
            self.mask_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.GroupNorm(num_channels=16, num_groups=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            )
        else:
            self.mask_cnn = None

    def forward(
        self,
        input,
        batch_positions=None,
        zones=None,
        pseudo_nms=True,
        heatmap_only=False,
    ):
        """
        Args:
            input (tensor): Input image time series.
            batch_positions (tensor): Date sequence of the batch images.
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection). This mapping
            is used at train time to predict and supervise at most one prediction
            per ground truth object for efficiency.
            When not provided all predicted centers receive supervision.
            pseudo_nms (bool): If True performs pseudo_nms to produce a panoptic prediction,
            otherwise the model returns potentially overlapping instance segmentation masks (default True).
            heatmap_only (bool): If True the model only returns the centerness heatmap. Can be useful for some
            warmup epochs of the centerness prediction, as all the rest hinges on this.

        Returns:
            predictions (dict[tensor]): A dictionary of predictions with the following keys:
                center_mask         (B,H,W) Binary mask of centers.
                saliency            (B,1,H,W) Global Saliency.
                heatmap             (B,1,H,W) Predicted centerness heatmap.
                semantic            (M, K) Predicted class scores for each center (with M the number of predicted centers).
                size                (M, 2) Predicted sizes for each center.
                confidence          (M,1) Predicted centerness for each center.
                centerness          (M,1) Predicted centerness for each center.
                instance_masks      List of N binary masks of varying shape.
                instance_boxes      (N, 4) Coordinates of the N bounding boxes.
                pano_instance       (B,H,W) Predicted instance id for each pixel.
                pano_semantic       (B,K,H,W) Predicted class score for each pixel.

        """
        out, maps = self.encoder(input, batch_positions=batch_positions)

        # Global Predictions
        heatmap = self.heatmap_conv(out)
        saliency = self.saliency_conv(out)

        center_mask, _ = self.center_extractor(
            heatmap, zones=zones
        )  # (B,H,W) mask of N detected centers
        center_mask = center_mask.squeeze()

        if heatmap_only:
            predictions = dict(
                center_mask=center_mask,
                saliency=None,
                heatmap=heatmap,
                semantic=None,
                size=None,
                offsets=None,
                confidence=None,
                instance_masks=None,
                instance_boxes=None,
                pano_instance=None,
                pano_semantic=None,
            )
            return predictions

        # Retrieve info of detected centers
        H, W = heatmap.shape[-2:]
        center_batch, center_h, center_w = torch.where(center_mask)
        center_positions = torch.stack([center_h, center_w], dim=1)

        # Construct multi-level feature stack for centers
        stack = []
        for i, m in enumerate(maps):
            h_mask = center_h // (2 ** (len(maps) - 1 - i))
            # Assumes resolution is divided by 2 at each level
            w_mask = center_w // (2 ** (len(maps) - 1 - i))
            m = m.permute(0, 2, 3, 1)
            stack.append(m[center_batch, h_mask, w_mask])
        stack = torch.cat(stack, dim=1)

        # Center-level predictions
        size = self.size_mlp(stack)
        sem = self.class_mlp(stack)
        shapes = self.shape_mlp(stack)
        shapes = shapes.view((-1, 1, self.shape_size, self.shape_size))
        # (N,1,S,S) instance shapes

        centerness = heatmap[center_mask[:, None, :, :]].unsqueeze(-1)
        confidence = centerness

        # Instance Boxes Assembling
        ## Minimal box size of 1px
        ## Combine clamped sizes and center positions to obtain box coordinates
        clamp_size = size.detach().round().long().clamp_min(min=1)
        half_size = clamp_size // 2
        remainder_size = clamp_size % 2
        start_hw = center_positions - half_size
        stop_hw = center_positions + half_size + remainder_size

        instance_boxes = torch.cat([start_hw, stop_hw], dim=1)
        instance_boxes.clamp_(min=0, max=H)
        instance_boxes = instance_boxes[:, [1, 0, 3, 2]]  # h,w,h,w to x,y,x,y

        valid_start = (-start_hw).clamp_(min=0)  # if h=-5 crop the shape mask before the 5th pixel
        valid_stop = (stop_hw - start_hw) - (stop_hw - H).clamp_(min=0)  # crop if h_stop > H

        # Instance Masks Assembling
        instance_masks = []
        for i, s in enumerate(shapes.split(1, dim=0)):
            h, w = clamp_size[i]  # Box size
            w_start, h_start, w_stop, h_stop = instance_boxes[i]  # Box coordinates
            h_start_valid, w_start_valid = valid_start[i]  # Part of the Box that lies
            h_stop_valid, w_stop_valid = valid_stop[i]  # within the image's extent

            ## Resample local shape mask
            pred_mask = (F.interpolate(s, size=(h.item(), w.item()), mode="bilinear")).squeeze(0)
            pred_mask = pred_mask[:, h_start_valid:h_stop_valid, w_start_valid:w_stop_valid]

            ## Crop saliency
            crop_saliency = saliency[center_batch[i], :, h_start:h_stop, w_start:w_stop]

            ## Combine both
            if self.mask_cnn is None:
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(crop_saliency)
            else:
                pred_mask = pred_mask + crop_saliency
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(
                    self.mask_cnn(pred_mask.unsqueeze(0)).squeeze(0)
                )
            instance_masks.append(pred_mask)

        # PSEUDO-NMS
        if pseudo_nms:
            panoptic_instance = []
            panoptic_semantic = []
            for b in range(center_mask.shape[0]):  # iterate over elements of batch
                panoptic_mask = torch.zeros(center_mask[0].shape, device=center_mask.device)
                semantic_mask = torch.zeros(
                    (self.num_classes, *center_mask[0].shape), device=center_mask.device
                )

                candidates = torch.where(center_batch == b)[
                    0
                ]  # get indices of centers in this batch element
                for n, (c, idx) in enumerate(
                    zip(*torch.sort(confidence[candidates].squeeze(), descending=True))
                ):
                    if c < self.min_confidence:
                        break
                    else:
                        new_mask = torch.zeros(center_mask[0].shape, device=center_mask.device)
                        pred_mask_bin = (
                            instance_masks[candidates[idx]].squeeze(0) > self.mask_threshold
                        ).float()

                        if pred_mask_bin.sum() > 0:
                            xtl, ytl, xbr, ybr = instance_boxes[candidates[idx]]
                            new_mask[ytl:ybr, xtl:xbr] = pred_mask_bin

                            if ((new_mask != 0) * (panoptic_mask != 0)).any():
                                n_total = (new_mask != 0).sum()
                                non_overlaping_mask = (new_mask != 0) * (panoptic_mask == 0)
                                n_new = non_overlaping_mask.sum().float()
                                if n_new / n_total > self.min_remain:
                                    panoptic_mask[non_overlaping_mask] = n + 1
                                    semantic_mask[:, non_overlaping_mask] = sem[candidates[idx]][
                                        :, None
                                    ]
                            else:
                                panoptic_mask[(new_mask != 0)] = n + 1
                                semantic_mask[:, (new_mask != 0)] = sem[candidates[idx]][:, None]
                panoptic_instance.append(panoptic_mask)
                panoptic_semantic.append(semantic_mask)
            panoptic_instance = torch.stack(panoptic_instance, dim=0)
            panoptic_semantic = torch.stack(panoptic_semantic, dim=0)
        else:
            panoptic_instance = None
            panoptic_semantic = None

        predictions = dict(
            center_mask=center_mask,
            saliency=saliency,
            heatmap=heatmap,
            semantic=sem,
            size=size,
            confidence=confidence,
            centerness=centerness,
            instance_masks=instance_masks,
            instance_boxes=instance_boxes,
            pano_instance=panoptic_instance,
            pano_semantic=panoptic_semantic,
        )

        return predictions


class CenterExtractor(nn.Module):
    def __init__(self):
        """
        Module for local maxima extraction
        """
        super(CenterExtractor, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, input, zones=None):
        """

        Args:
            input (tensor): Centerness heatmap
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection).
            If provided, the highest local maxima in each zone is kept. As a result at most one
            prediction is made per ground truth object.
            If not provided, all local maxima are returned.
        """
        if zones is not None:
            masks = []
            for b, x in enumerate(input.split(1, dim=0)):
                x = x.view(-1)
                _, idxs = scatter_max(x, zones[b].view(-1).long())
                mask = torch.zeros(x.shape, device=x.device)
                mask[idxs[idxs != x.shape[0]]] = 1
                masks.append(mask.view(zones[b].shape).unsqueeze(0))
            centermask = torch.stack(masks, dim=0).bool()
        else:
            centermask = input == self.pool(input)
            no_valley = input > input.mean()
            centermask = centermask * no_valley

        n_centers = int(centermask.sum().detach().cpu())
        return centermask, n_centers


# --- staging entry points ----------------------------------------------------


def build_paps():
    """Tiny PaPs (U-TAE encoder + PaPs panoptic head) for tracing: shallow encoder
    widths, small n_head/d_model, small class/shape counts, mask_conv disabled to
    keep the traced graph compact (mask_cnn adds a highly data-dependent per-instance
    loop that is orthogonal to the architecture family being captured)."""
    encoder = UTAE(
        input_dim=3,
        encoder_widths=[8, 8, 16],
        decoder_widths=[8, 8, 16],
        out_conv=[8, 4],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=2,
        d_model=8,
        d_k=4,
        encoder=True,
        pad_value=0,
        padding_mode="reflect",
    )
    model = PaPs(
        encoder,
        num_classes=3,
        shape_size=4,
        mask_conv=False,
        min_confidence=0.0,
        min_remain=0.0,
        mask_threshold=0.4,
    )
    model.eval()
    return model


def example_input_paps():
    # (B, T, C, H, W) satellite image time series; H=W=16 keeps the two down-sample
    # stages (16 -> 8 -> 4) and the center-extraction/pseudo-NMS loop cheap.
    # batch_positions (B, T) is required: LTAE2d's positional encoder indexes it
    # directly and does not tolerate None. B=2 (not 1): PaPs.forward does
    # `center_mask.squeeze()` before `torch.where(center_mask)` -- with B=1 that
    # squeezes away the batch dim and torch.where returns 2 tensors instead of 3,
    # a real shape-collapse quirk of the original code, not something we should
    # paper over by editing the architecture.
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 16, 16)
    batch_positions = torch.arange(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    return (x, batch_positions)


MENAGERIE_ENTRIES = [
    ("PaPs", "build_paps", "example_input_paps", 2021, "vendored-pytorch"),
]
