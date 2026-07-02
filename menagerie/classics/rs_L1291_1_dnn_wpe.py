# FAITHFUL PORT of nttcslab-sp/dnn_wpe @ master (original framework: PyTorch + torch_complex)
#   Ported files: pytorch_wpe.py (wpe_one_iteration, get_correlations,
#   get_filter_matrix_conj, perform_filter_operation_v2, signal_framing, get_power),
#   example/model.py (DNN_WPE, Estimator, BLSTM, CNN, make_pad_mask).
# https://github.com/nttcslab-sp/dnn_wpe
#
# DNN-WPE (Kinoshita et al., "Neural network-based spectrum estimation for online
# WPE dereverberation", Interspeech 2017 / Nakatani et al. "Simultaneous
# Denoising, Dereverberation, and Source Separation Using a Unified Convolutional
# Beamformer") -- learns a per-time-frequency masking network ("Estimator": CNN or
# BLSTM over the STFT magnitude/power/log-power features) that reweights the WPE
# power spectrum, then filters the (multi-channel) complex STFT via classical
# iterative multi-channel linear-prediction WPE ("wpe_one_iteration": weighted
# correlation matrix/vector, closed-form filter solve, filter-and-subtract). The
# repo's original implementation represents complex spectra with the third-party
# `torch_complex.ComplexTensor` wrapper (thin real/imag pair with einsum/matmul/
# reverse/pad helpers in `torch_complex.functional`); `torch_complex` is not an
# installed base lib here, but every operation it wraps has a direct native-torch
# equivalent since PyTorch added first-class complex dtypes (`torch.complex64`).
#
# PORT CHANGES (mechanical dtype-substitution only, NOT an architecture change):
#   - `ComplexTensor(real, imag)` -> `torch.complex(real, imag)` (native complex64).
#   - `x.conj()` -> native `torch.Tensor.conj()` (same semantics on complex dtype).
#   - `FC.einsum(eq, (a, b))` -> `torch.einsum(eq, a, b)` (native torch.einsum
#     already supports complex dtypes).
#   - `FC.matmul` -> `torch.matmul`; `FC.reverse` -> `torch.flip`; `FC.pad` ->
#     `torch.nn.functional.pad` (all natively complex-dtype-safe).
#   - `FC.stack` -> `torch.stack`.
#   - `correlation_matrix.inverse()` kept as `.inverse()` (native complex64 support).
#   - Everything else (network topology of `Estimator`/`BLSTM`/`CNN`, the WPE
#     correlation/filter/subtract math, control flow, iteration counts, tensor
#     shapes) is an unmodified transcription of the original source.

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


# ---------------------------------------------------------------------------
# pytorch_wpe.py (ported to native torch.complex64)
# ---------------------------------------------------------------------------


def signal_framing(
    signal: torch.Tensor, frame_length: int, frame_step: int, pad_value=0
) -> torch.Tensor:
    """Expands signal into frames of frame_length.

    Args:
        signal : (B * F, D, T)
    Returns:
        torch.Tensor: (B * F, D, T, W)
    """
    signal = torch.nn.functional.pad(signal, (0, frame_length - 1), "constant", pad_value)
    indices = sum(
        [
            list(range(i, i + frame_length))
            for i in range(0, signal.size(-1) - frame_length + 1, frame_step)
        ],
        [],
    )
    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal


def get_power(signal: torch.Tensor, dim=-2) -> torch.Tensor:
    """Calculates power for `signal`.

    Args:
        signal : Single frequency signal with shape (F, C, T).
        dim: reduce_mean axis
    Returns:
        Power with shape (F, T)
    """
    power = signal.real**2 + signal.imag**2
    power = power.mean(dim=dim)
    return power


def get_correlations(
    Y: torch.Tensor, inverse_power: torch.Tensor, taps: int, delay: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates weighted correlations of a window of length taps.

    Args:
        Y : Complex-valued STFT signal with shape (F, C, T)
        inverse_power : Weighting factor with shape (F, T)
        taps (int): Length of correlation window
        delay (int): Delay for the weighting factor
    Returns:
        Correlation matrix of shape (F, taps*C, taps*C)
        Correlation vector of shape (F, taps, C, C)
    """
    assert inverse_power.dim() == 2, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), (inverse_power.size(0), Y.size(0))

    F_, C, T = Y.size()

    # Y: (F, C, T) -> Psi: (F, C, T, taps)
    Psi = signal_framing(Y, frame_length=taps, frame_step=1)[..., : T - delay - taps + 1, :]
    # Reverse along taps-axis
    Psi = torch.flip(Psi, dims=[-1])
    Psi_conj_norm = Psi.conj() * inverse_power[..., None, delay + taps - 1 :, None]

    # (F, C, T, taps) x (F, C, T, taps) -> (F, taps, C, taps, C)
    correlation_matrix = torch.einsum("fdtk,fetl->fkdle", Psi_conj_norm, Psi)
    # (F, taps, C, taps, C) -> (F, taps * C, taps * C)
    correlation_matrix = correlation_matrix.reshape(F_, taps * C, taps * C)

    # (F, C, T, taps) x (F, C, T) -> (F, taps, C, C)
    correlation_vector = torch.einsum("fdtk,fet->fked", Psi_conj_norm, Y[..., delay + taps - 1 :])

    return correlation_matrix, correlation_vector


def get_filter_matrix_conj(
    correlation_matrix: torch.Tensor, correlation_vector: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Calculate (conjugate) filter matrix based on correlations for one freq.

    Args:
        correlation_matrix : Correlation matrix (F, taps * C, taps * C)
        correlation_vector : Correlation vector (F, taps, C, C)
    Returns:
        filter_matrix_conj: (F, taps, C, C)
    """
    F_, taps, C, _ = correlation_vector.size()

    # (F, taps, C1, C2) -> (F, C1, taps, C2) -> (F, C1, taps * C2)
    correlation_vector = correlation_vector.permute(0, 2, 1, 3).contiguous().view(F_, C, taps * C)

    eye = torch.eye(
        correlation_matrix.size(-1),
        dtype=correlation_matrix.dtype,
        device=correlation_matrix.device,
    )
    shape = tuple(1 for _ in range(correlation_matrix.dim() - 2)) + correlation_matrix.shape[-2:]
    eye = eye.view(*shape)
    correlation_matrix = correlation_matrix + eps * eye

    inv_correlation_matrix = correlation_matrix.inverse()
    # (F, C, taps, C) x (F, taps * C, taps * C) -> (F, C, taps * C)
    stacked_filter_conj = torch.matmul(correlation_vector, inv_correlation_matrix.transpose(-1, -2))

    # (F, C1, taps * C2) -> (F, C1, taps, C2) -> (F, taps, C2, C1)
    filter_matrix_conj = stacked_filter_conj.view(F_, C, taps, C).permute(0, 2, 3, 1)
    return filter_matrix_conj


def perform_filter_operation_v2(
    Y: torch.Tensor, filter_matrix_conj: torch.Tensor, taps: int, delay: int
) -> torch.Tensor:
    """perform_filter_operation_v2

    Args:
        Y : Complex-valued STFT signal of shape (F, C, T)
        filter_matrix_conj: (F, taps, C, C)
    """
    T = Y.size(-1)
    # Y_tilde: (taps, F, C, T)
    Y_tilde = torch.stack(
        [
            torch.nn.functional.pad(
                Y[:, :, : T - delay - i], (delay + i, 0), mode="constant", value=0
            )
            for i in range(taps)
        ],
        dim=0,
    )
    reverb_tail = torch.einsum("fpde,pfdt->fet", filter_matrix_conj, Y_tilde)
    return Y - reverb_tail


def wpe_one_iteration(
    Y: torch.Tensor,
    power: torch.Tensor,
    taps: int = 10,
    delay: int = 3,
    eps: float = 1e-10,
    inverse_power: bool = True,
) -> torch.Tensor:
    """WPE for one iteration.

    Args:
        Y: Complex valued STFT signal with shape (..., C, T)
        power: (..., T)
        taps: Number of filter taps
        delay: Delay as a guard interval, such that X does not become zero.
    Returns:
        enhanced: (..., C, T)
    """
    assert Y.size()[:-2] == power.size()[:-1]
    batch_freq_size = Y.size()[:-2]
    Y = Y.reshape(-1, *Y.size()[-2:])
    power = power.reshape(-1, power.size()[-1])

    if inverse_power:
        inverse_power_t = 1 / torch.clamp(power, min=eps)
    else:
        inverse_power_t = power

    correlation_matrix, correlation_vector = get_correlations(Y, inverse_power_t, taps, delay)
    filter_matrix_conj = get_filter_matrix_conj(correlation_matrix, correlation_vector)
    enhanced = perform_filter_operation_v2(Y, filter_matrix_conj, taps, delay)

    enhanced = enhanced.view(*batch_freq_size, *Y.size()[-2:])
    return enhanced


# ---------------------------------------------------------------------------
# example/model.py
# ---------------------------------------------------------------------------


def make_pad_mask(lengths, xs=None, length_dim=-1):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class BLSTM(torch.nn.LSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = True,
        channel_independent: bool = True,
    ):
        self.channel_independent = channel_independent
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, xs: torch.Tensor, input_lengths: torch.LongTensor):
        # xs: (B, C, T, F)
        B, C, T, F_ = xs.size()
        if self.channel_independent:
            # xs: (B, C, T, F) -> xs: (B * C, 1, T, F)
            xs = xs.reshape(-1, 1, T, F_)
            # input_lengths: (B,) -> input_lengths_: (B * C)
            input_lengths = input_lengths[:, None].expand(-1, C).contiguous().reshape(-1)

        # xs: (B, C, T, F) -> xs: (B, T, C * F)
        xs = xs.transpose(1, 2).contiguous().view(xs.size(0), T, -1)

        xs_pack = pack_padded_sequence(
            xs, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        xs_pack, _ = super().forward(xs_pack)
        xs, _ = pad_packed_sequence(xs_pack, batch_first=True, total_length=T)

        if self.channel_independent:
            # xs: (B * C, 1, T, F) -> xs: (B, C, T, F)
            xs = xs.reshape(B, C, T, -1)
        else:
            # xs: (B, T, C * F) -> xs: (B, C, T, F)
            xs = xs.view(B, T, C, -1).transpose(1, 2)

        # xs: (B, C, T, F)
        return xs


class CNN(torch.nn.Sequential):
    def __init__(self, channels: Sequence[int] = (8, 64, 64, 8), conv_dim: int = 1):
        layers = []
        for i in range(len(channels) - 1):
            Convnd = getattr(torch.nn, f"Conv{conv_dim}d")
            layers.append(Convnd(channels[i], channels[i + 1], 3, stride=1, padding=1))
            layers.append(torch.nn.ReLU())
        super().__init__(*layers)


class Estimator(torch.nn.Module):
    def __init__(
        self,
        model_type,
        feat_type: str = "amplitude",
        input_size: int = 400,
        hidden_size: int = 1024,
        out_size: int = None,
        num_layers: int = 4,
        nchannel: int = 8,
        channel_independent: bool = True,
    ):
        super().__init__()

        self.channel_independent = channel_independent
        supported = ("amplitude", "power", "log_power", "concat")
        if feat_type not in supported:
            raise ValueError(f"feat_type must be one of {supported}: {feat_type} ")
        self.feat_type = feat_type

        self.model_type = model_type
        if model_type in ("blstm", "lstm"):
            self.net = BLSTM(
                input_size=input_size if channel_independent else nchannel * input_size,
                channel_independent=channel_independent,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                dropout=0,
                bidirectional="b" in model_type,
            )
            li_input_size = (2 if "b" in model_type else 1) * hidden_size

        elif model_type == "cnn":
            if channel_independent:
                # in: (B * C, F, T)
                channels = [input_size] + [hidden_size for _ in range(num_layers)]
                self.net = CNN(channels, conv_dim=1)
                li_input_size = hidden_size
            else:
                # in: (B, C, T, F)
                channels = [nchannel] + [hidden_size for _ in range(num_layers - 1)] + [nchannel]
                self.net = CNN(channels, conv_dim=2)
                li_input_size = input_size
        else:
            raise NotImplementedError(model_type)

        if out_size is None:
            out_size = input_size

        self.linear = torch.nn.Linear(li_input_size, out_size)

    def forward(self, xs: torch.Tensor, input_lengths: torch.LongTensor) -> torch.Tensor:
        assert xs.size(0) == input_lengths.size(0), (xs.size(0), input_lengths.size(0))

        # xs: (B, C, T, D)
        C = xs.size(1)
        if self.feat_type == "amplitude":
            # xs: (B, C, T, F) -> (B, C, T, F)
            xs = (xs.real**2 + xs.imag**2) ** 0.5
        elif self.feat_type == "power":
            xs = xs.real**2 + xs.imag**2
        elif self.feat_type == "log_power":
            xs = torch.log(xs.real**2 + xs.imag**2)
        elif self.feat_type == "concat":
            xs = torch.cat([xs.real, xs.imag], -1)
        else:
            raise NotImplementedError(f"Not implemented: {self.feat_type}")

        if self.model_type in ("blstm", "lstm"):
            # xs: (B, C, T, F) -> xs: (B, C, T, D)
            xs = self.net(xs, input_lengths)

        elif self.model_type == "cnn":
            if self.channel_independent:
                # xs: (B, C, T, F) -> xs: (B * C, F, T)
                xs = xs.reshape(-1, *xs.size()[2:]).transpose(1, 2)
                # xs: (B * C, F, T) -> xs: (B * C, D, T)
                xs = self.net(xs)
                # xs: (B * C, D, T) -> (B, C, T, D)
                xs = xs.transpose(1, 2).contiguous().view(-1, C, xs.size(2), xs.size(1))
            else:
                xs = self.net(xs)
        else:
            raise NotImplementedError(f"Not implemented: {self.model_type}")

        # xs: (B, C, T, D) -> out:(B, C, T, F)
        out = self.linear(xs)
        # Zero padding
        out = torch.sigmoid(out)
        out = out.masked_fill(make_pad_mask(input_lengths, out, length_dim=2), 0)

        return out


class DNN_WPE(torch.nn.Module):
    def __init__(
        self,
        model_type: str = "cnn",
        feat_type: str = "log_power",
        out_type: str = "mask",
        input_size: int = 257,
        hidden_size: int = 300,
        out_size: int = None,
        num_layers: int = 2,
        nchannel: int = 8,
        channel_independent: bool = True,
        taps: int = 5,
        delay: int = 3,
        use_dnn: bool = True,
        iterations: int = 1,
        normalization: bool = False,
        lcontext: int = 4,
        rcontext: int = 4,
        inverse_power: bool = True,
    ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay

        self.normalization = normalization
        self.use_dnn = use_dnn
        self.inverse_power = inverse_power

        self.model_type = model_type
        self.lcontext = lcontext
        self.rcontext = rcontext

        if out_type is None:
            self.out_type = feat_type
        else:
            self.out_type = out_type

        if use_dnn:
            self.estimator = Estimator(
                model_type=model_type,
                feat_type=feat_type,
                input_size=input_size,
                hidden_size=hidden_size,
                out_size=out_size,
                num_layers=num_layers,
                nchannel=nchannel,
                channel_independent=channel_independent,
            )
        else:
            self.estimator = None

    def forward(
        self, data: torch.Tensor, ilens: torch.LongTensor = None, return_wpe: bool = True
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if ilens is None:
            ilens = torch.full((data.size(0),), data.size(2), dtype=torch.long, device=data.device)

        r = -self.rcontext if self.rcontext != 0 else None
        enhanced = data[:, :, self.lcontext : r, :]

        if self.lcontext != 0 or self.rcontext != 0:
            assert all(ilens[0] == i for i in ilens)

            # Create context window (a.k.a Splicing)
            if self.model_type in ("blstm", "lstm"):
                width = data.size(2) - self.lcontext - self.rcontext
                # data: (B, C, l + w + r, F)
                indices = [
                    i + j for i in range(width) for j in range(1 + self.lcontext + self.rcontext)
                ]
                _y = data[:, :, indices]
                # data: (B, C, l, (1 + w + r), F)
                data = _y.reshape(
                    data.size(0),
                    data.size(1),
                    width,
                    (1 + self.lcontext + self.rcontext) * data.size(3),
                )
                ilens = torch.full((data.size(0),), width, dtype=torch.long, device=data.device)
                del _y

        for i in range(self.iterations):
            power = enhanced.real**2 + enhanced.imag**2
            # Calculate power: (B, C, T, Context, F)
            if i == 0 and self.use_dnn:
                # mask: (B, C, T, F)
                mask = self.estimator(data, ilens)
                if mask.size(2) != power.size(2):
                    assert mask.size(2) == (power.size(2) + self.rcontext + self.lcontext)
                    r = -self.rcontext if self.rcontext != 0 else None
                    mask = mask[:, :, self.lcontext : r, :]

                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(dim=-2)[..., None]
                if self.out_type == "mask":
                    power = power * mask
                else:
                    power = mask

                    if self.out_type == "amplitude":
                        power = power**2
                    elif self.out_type == "log_power":
                        power = power.exp()
                    elif self.out_type == "power":
                        pass
                    else:
                        raise NotImplementedError(self.out_type)

            if not return_wpe:
                return None, power

            # power: (B, C, T, F) -> _power: (B, F, T)
            _power = power.mean(dim=1).transpose(-1, -2).contiguous()

            # data: (B, C, T, F) -> _data: (B, F, C, T)
            _data = data.permute(0, 3, 1, 2).contiguous()
            # _enhanced: (B, F, C, T)
            _enhanced_list = []
            for d, p, n in zip(_data, _power, ilens):
                # e: (F, C, T) -> (T, C, F)
                e = wpe_one_iteration(
                    d[..., :n],
                    p[..., :n],
                    taps=self.taps,
                    delay=self.delay,
                    inverse_power=self.inverse_power,
                ).transpose(0, 2)
                _enhanced_list.append(e)
            # _enhanced: B x (T, C, F) -> (B, T, C, F) -> (B, F, C, T)
            _enhanced = pad_sequence(_enhanced_list, batch_first=True).transpose(1, 3)

            # enhanced: (B, F, C, T) -> (B, C, T, F)
            enhanced = _enhanced.permute(0, 2, 3, 1)

        # enhanced: (B, C, T, F), power: (B, C, T, F)
        return enhanced, power


MENAGERIE_ZOO = "ported-pytorch"


def build_dnn_wpe():
    # Small sizes to keep the trace/render fast: input_size (freq bins) shrunk from
    # the repo default 257 to 17, hidden_size 300->16, nchannel (mics) 8->2. taps/
    # delay kept small but nonzero so the WPE filtering path is exercised, as in
    # the real default config. lcontext/rcontext=0: the repo's context-splicing
    # branch reshapes the DNN input to (1+lcontext+rcontext)*F channels, which only
    # matches Estimator's channel count for model_type in ('blstm', 'lstm') --
    # Estimator's own CNN branch consumes the *unspliced* F -- so splicing is a
    # BLSTM-only feature in the real code; model_type="cnn" with lcontext=rcontext=0
    # is the faithful, shape-consistent real config for the CNN masking estimator.
    model = DNN_WPE(
        model_type="cnn",
        feat_type="log_power",
        out_type="mask",
        input_size=17,
        hidden_size=16,
        num_layers=2,
        nchannel=2,
        channel_independent=True,
        taps=2,
        delay=1,
        use_dnn=True,
        iterations=1,
        lcontext=0,
        rcontext=0,
    )
    model.eval()
    return model


def example_input_dnn_wpe():
    # data: (B, C, T, F) complex-valued STFT, matching DNN_WPE.forward's expected
    # layout (batch, channel/mic, time, freq).
    B, C, T, F_ = 1, 2, 10, 17
    real = torch.randn(B, C, T, F_)
    imag = torch.randn(B, C, T, F_)
    return torch.complex(real, imag)


MENAGERIE_ENTRIES = [
    ("DNN-WPE", "build_dnn_wpe", "example_input_dnn_wpe", 2017, MENAGERIE_ZOO),
]
