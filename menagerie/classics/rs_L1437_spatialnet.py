# SOURCE: vendored from Audio-WestlakeU/NBSS @ 162a719080ec560ce5d1f93da2721283ffa0c158
# https://raw.githubusercontent.com/Audio-WestlakeU/NBSS/main/models/arch/SpatialNet.py
# https://raw.githubusercontent.com/Audio-WestlakeU/NBSS/main/models/arch/base/norm.py
# https://raw.githubusercontent.com/Audio-WestlakeU/NBSS/main/models/arch/base/non_linear.py
# https://raw.githubusercontent.com/Audio-WestlakeU/NBSS/main/models/arch/base/linear_group.py
#
# SpatialNet: multichannel joint speech separation/denoising/dereverberation network
# (TASLP 2024). Narrow-band self-attention (MHSA over time) + cross-band blocks
# (per-frequency depthwise Conv1d "fconv" modules straddling a shared full-band
# LinearGroup module) + a T-ConvFFN feed-forward module, stacked into SpatialNetLayer
# blocks, wrapped by a Conv1d encoder and Linear decoder. `SpatialNetLayer`/`SpatialNet`
# are transcribed verbatim from the official repo's `models/arch/SpatialNet.py`; the
# `LayerNorm`/`GlobalLayerNorm`/`BatchNorm1d`/`GroupNorm`/`GroupBatchNorm`/`new_norm`
# helpers from `models/arch/base/norm.py`, the `PReLU`/`new_non_linear` helpers from
# `models/arch/base/non_linear.py`, and `LinearGroup` from `models/arch/base/linear_group.py`
# are inlined verbatim (only the cross-file imports were flattened into this single
# module; no architectural code was changed).

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import MultiheadAttention, Module
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# --- from models/arch/base/norm.py ---


class LayerNorm(nn.LayerNorm):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o


class GlobalLayerNorm(nn.Module):
    """gLN in convtasnet"""

    def __init__(self, dim_hidden: int, seq_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_last = seq_last
        self.eps = eps

        if seq_last:
            self.weight = Parameter(torch.empty([dim_hidden, 1]))
            self.bias = Parameter(torch.empty([dim_hidden, 1]))
        else:
            self.weight = Parameter(torch.empty([dim_hidden]))
            self.bias = Parameter(torch.empty([dim_hidden]))
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): shape [B, Seq, H] or [B, H, Seq]
        """
        var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)

        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return "{dim_hidden}, seq_last={seq_last}, eps={eps}".format(**self.__dict__)


class BatchNorm1d(nn.Module):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = self.bn.forward(input)  # accepts [B, H, Seq]
        if not self.seq_last:
            o = o.transpose(-1, -2)
        return o


class GroupNorm(nn.GroupNorm):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last == False:
            input = input.transpose(-1, 1)  # [B, ..., H] -> [B, H, ...]
        o = super().forward(input)  # accepts [B, H, ...]
        if self.seq_last == False:
            o = o.transpose(-1, 1)
        return o


class GroupBatchNorm(Module):
    """Applies Group Batch Normalization over a group of inputs

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    see: `Changsheng Quan, Xiaofei Li. NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer. arXiv:2212.02076.`

    """

    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    seq_last: bool
    share_along_sequence_dim: bool

    def __init__(
        self,
        dim_hidden: int,
        group_size: Optional[int],
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
        dims_norm: List[int] = None,
        dim_affine: int = None,
    ) -> None:
        """
        Args:
            dim_hidden (int): hidden dimension
            group_size (int): the size of group, optional
            share_along_sequence_dim (bool): share statistics along the sequence dimension. Defaults to False.
            seq_last (bool): whether the shape of input is [B, Seq, H] or [B, H, Seq]. Defaults to False, i.e. [B, Seq, H].
            affine (bool): affine transformation. Defaults to True.
            eps (float): Defaults to 1e-5.
            dims_norm: the dims for normalization
            dim_affine: the dims for affine transformation
        """
        super(GroupBatchNorm, self).__init__()

        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if seq_last:
                weight = torch.empty([dim_hidden, 1])
                bias = torch.empty([dim_hidden, 1])
            else:
                weight = torch.empty([dim_hidden])
                bias = torch.empty([dim_hidden])

        assert (dims_norm is not None and dim_affine is not None) or (dims_norm is not None), (
            dims_norm,
            dim_affine,
            "should be none at the time",
        )
        self.dims_norm, self.dim_affine = dims_norm, dim_affine
        if dim_affine is not None:
            assert dim_affine < 0, dim_affine
            weight = weight.squeeze()
            bias = bias.squeeze()
            while dim_affine < -1:
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
                dim_affine += 1

        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, group_size: int = None) -> Tensor:
        """
        Args:
            x: shape [B, Seq, H] if seq_last=False, else shape [B, H, Seq] , where B = num of groups * group size.
            group_size: the size of one group. if not given anywhere, the input must be 4-dim tensor with shape [B, group_size, Seq, H] or [B, group_size, H, Seq]
        """
        if self.group_size != None:
            assert group_size == None or group_size == self.group_size, (
                group_size,
                self.group_size,
            )
            group_size = self.group_size

        if group_size is not None:
            assert (x.shape[0] // group_size) * group_size, (
                f"batch size {x.shape[0]} is not divisible by group size {group_size}"
            )

        original_shape = x.shape
        if self.dims_norm is not None:
            var, mean = torch.var_mean(x, dim=self.dims_norm, unbiased=False, keepdim=True)
            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
        elif self.seq_last == False:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, Seq, H = x.shape
            else:
                B, Seq, H = x.shape
                x = x.reshape(B // group_size, group_size, Seq, H)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 3), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)
        else:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, H, Seq = x.shape
            else:
                B, H, Seq = x.shape
                x = x.reshape(B // group_size, group_size, H, Seq)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 2), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)

        return output

    def extra_repr(self) -> str:
        return (
            "{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, seq_last={seq_last}, eps={eps}, "
            "affine={affine}".format(**self.__dict__)
        )


def new_norm(
    norm_type: str,
    dim_hidden: int,
    seq_last: bool,
    group_size: int = None,
    num_groups: int = None,
    dims_norm: List[int] = None,
    dim_affine: int = None,
) -> nn.Module:
    if norm_type.upper() == "LN":
        norm = LayerNorm(normalized_shape=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == "GBN":
        norm = GroupBatchNorm(
            dim_hidden=dim_hidden,
            seq_last=seq_last,
            group_size=group_size,
            share_along_sequence_dim=False,
            dims_norm=dims_norm,
            dim_affine=dim_affine,
        )
    elif norm_type == "GBNShare":
        norm = GroupBatchNorm(
            dim_hidden=dim_hidden,
            seq_last=seq_last,
            group_size=group_size,
            share_along_sequence_dim=True,
            dims_norm=dims_norm,
            dim_affine=dim_affine,
        )
    elif norm_type.upper() == "BN":
        norm = BatchNorm1d(num_features=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == "GN":
        norm = GroupNorm(num_groups=num_groups, num_channels=dim_hidden, seq_last=seq_last)
    elif norm_type == "gLN":
        norm = GlobalLayerNorm(dim_hidden, seq_last=seq_last)
    else:
        raise Exception(norm_type)
    return norm


# --- from models/arch/base/non_linear.py ---


class PReLU(nn.PReLU):
    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, dim: int = 1, device=None, dtype=None
    ) -> None:
        super().__init__(num_parameters, init, device, dtype)
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim == 1:
            # [B, Chn, Feature]
            return super().forward(input)
        else:
            return super().forward(input.transpose(self.dim, 1)).transpose(self.dim, 1)


def new_non_linear(non_linear_type: str, dim_hidden: int, seq_last: bool) -> nn.Module:
    if non_linear_type.lower() == "prelu":
        return PReLU(num_parameters=dim_hidden, dim=1 if seq_last == True else -1)
    elif non_linear_type.lower() == "silu":
        return nn.SiLU()
    elif non_linear_type.lower() == "sigmoid":
        return nn.Sigmoid()
    elif non_linear_type.lower() == "relu":
        return nn.ReLU()
    elif non_linear_type.lower() == "leakyrelu":
        return nn.LeakyReLU()
    elif non_linear_type.lower() == "elu":
        return nn.ELU()
    else:
        raise Exception(non_linear_type)


# --- from models/arch/base/linear_group.py ---


class LinearGroup(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_groups: int, bias: bool = True
    ) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"


# --- from models/arch/SpatialNet.py ---


class SpatialNetLayer(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        dim_ffn: int,
        dim_squeeze: int,
        num_freqs: int,
        num_heads: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
        padding: str = "zeros",
        full: nn.Module = None,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList(
            [
                new_norm(
                    norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups
                ),
                nn.Conv1d(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
            ]
        )
        # full-band linear module
        self.norm_full = new_norm(
            norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups
        )
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU()
        )
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = (
            LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        )
        self.unsqueeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU()
        )
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList(
            [
                new_norm(
                    norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups
                ),
                nn.Conv1d(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
            ]
        )

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(
            norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups
        )
        self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList(
            [
                new_norm(
                    norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups
                ),
                nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(
                    in_channels=dim_ffn,
                    out_channels=dim_ffn,
                    kernel_size=t_kernel_size,
                    padding="same",
                    groups=t_conv_groups,
                ),
                nn.SiLU(),
                nn.Conv1d(
                    in_channels=dim_ffn,
                    out_channels=dim_ffn,
                    kernel_size=t_kernel_size,
                    padding="same",
                    groups=t_conv_groups,
                ),
                new_norm(
                    norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups
                ),
                nn.SiLU(),
                nn.Conv1d(
                    in_channels=dim_ffn,
                    out_channels=dim_ffn,
                    kernel_size=t_kernel_size,
                    padding="same",
                    groups=t_conv_groups,
                ),
                nn.SiLU(),
                nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
            ]
        )
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        x_, attn = self._tsa(x, att_mask)
        x = x + x_
        x = x + self._tconvffn(x)
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        need_weights = False if hasattr(self, "need_weights") else self.need_weights
        x, attn = self.mhsa.forward(
            x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=attn_mask
        )
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class SpatialNet(nn.Module):
    def __init__(
        self,
        dim_input: int,  # the input dim for each time-frequency point
        dim_output: int,  # the output dim for each time-frequency point
        dim_squeeze: int,
        num_layers: int,
        num_freqs: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_heads: int = 2,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
        padding: str = "zeros",
        full_share: int = 0,  # share from layer 0
    ):
        super().__init__()

        # encoder
        self.encoder = nn.Conv1d(
            in_channels=dim_input,
            out_channels=dim_hidden,
            kernel_size=encoder_kernel_size,
            stride=1,
            padding="same",
        )

        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
            )
            if hasattr(layer, "full"):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor, return_attn_score: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        for m in self.layers:
            setattr(m, "need_weights", return_attn_score)
            x, attn = m(x)
            if return_attn_score:
                attns.append(attn)

        y = self.decoder(x)
        if return_attn_score:
            return y.contiguous(), attns
        else:
            return y.contiguous()


def build_spatialnet():
    model = SpatialNet(
        dim_input=4,
        dim_output=2,
        num_layers=1,
        dim_hidden=8,
        dim_ffn=16,
        kernel_size=(5, 3),
        conv_groups=(2, 2),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=2,
        num_freqs=9,
        full_share=0,
        num_heads=2,
    )
    model.eval()
    return model


def example_input_spatialnet():
    # [Batch, Freq, Time, Feature]; Feature=4 emulates 2-mic complex STFT (real+imag per mic)
    return torch.randn(1, 9, 6, 4)


MENAGERIE_ENTRIES = [
    ("SpatialNet", "build_spatialnet", "example_input_spatialnet", 2024, MENAGERIE_ZOO),
]
