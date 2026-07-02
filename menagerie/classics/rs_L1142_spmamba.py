# FAITHFUL PORT of JusperLee/SPMamba @ main (original framework: PyTorch, but
# depends on compiled Mamba CUDA/Triton kernels that are unavailable/broken in
# this environment)
#
# SPMamba (Interspeech 2024 workshop preprint / Li et al., "SPMamba: State-space
# model is all you need in speech separation"). Real architecture: a TF-GridNet
# separator (Wang et al. 2023) with the intra-/inter-chunk BLSTM branches
# replaced by bidirectional Mamba (state-space) blocks.
#
# Sources transcribed faithfully, verbatim math/structure, from:
#   - look2hear/models/SPMamba.py (SPMamba, GridNetBlock, MambaBlock,
#     STFTEncoder, STFTDecoder, LayerNormalization4D/4DCF)
#       https://raw.githubusercontent.com/JusperLee/SPMamba/main/look2hear/models/SPMamba.py
#   - mamba_ssm/modules/mamba_simple.py (Mamba, the slow/non-fused-kernel
#     forward path: `causal_conv1d_fn is None` branch, which is the real
#     code's own CPU/no-CUDA-extension fallback)
#       https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba_simple.py
#   - mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref (the real
#     repo's own pure-PyTorch reference implementation of the selective scan,
#     used whenever the `selective_scan_cuda` extension is unavailable)
#       https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py
#   - mamba_ssm/modules/block.py (Block, `fused_add_norm=False` branch, which
#     is what SPMamba's own MambaBlock constructs)
#       https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/block.py
#   - mamba_ssm/models/mixer_seq_simple.py::_init_weights
#       https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/models/mixer_seq_simple.py
#
# Why a port instead of vendoring: the real look2hear/models/SPMamba.py imports
# `mamba_ssm.modules.mamba_simple.{Mamba,Block}` whose fast path needs the
# compiled `selective_scan_cuda`/`causal_conv1d` CUDA extensions (broken ABI in
# this env -- `undefined symbol` on import) and also imports `torch_complex`
# (not installed) purely for a pre-1.9 complex-tensor shim that this env's
# torch (2.8, far past 1.9) does not need. Every mechanism below is transcribed
# 1:1 from the real modules: the STFT/iSTFT encode-decode, the RMS-normalized
# Mamba SSM block (selective scan via the official pure-PyTorch reference
# formula), the bidirectional bundling into MambaBlock, the dual-path
# intra-/inter-chunk unfold + Mamba + deconv GridNetBlock, and the multi-head
# 4D-attention block. Only the librosa/CUDA-extension/ComplexTensor plumbing is
# swapped for torch's native `torch.stft`/`torch.complex` (which the real
# `Stft` class itself uses on this exact torch version -- see the
# `is_torch_1_10_plus` branch in look2hear/layers/stft_tfgn.py) and for
# `selective_scan_ref` (the real repo's own CUDA-free scan).

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# mamba_ssm/ops/selective_scan_interface.py :: selective_scan_ref
# (real repo's own pure-PyTorch selective-scan reference implementation, used
# whenever the compiled CUDA extension is unavailable)
# ---------------------------------------------------------------------------
def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = B.repeat_interleave(dim // B.shape[1], dim=1)
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = C.repeat_interleave(dim // C.shape[1], dim=1)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * D.unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# ---------------------------------------------------------------------------
# mamba_ssm/modules/mamba_simple.py :: Mamba
# (transcribed exactly, restricted to the CUDA-free forward path:
# use_fast_path is effectively False here since causal_conv1d_fn is None in
# this environment, which is precisely the real code's own fallback branch)
# ---------------------------------------------------------------------------
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = (
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
            .repeat(self.d_inner, 1)
            .contiguous()
        )
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        xz = (
            (self.in_proj.weight @ hidden_states.transpose(1, 2).reshape(dim, batch * seqlen))
            .reshape(-1, batch, seqlen)
            .transpose(0, 1)
        )
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.to(dtype=xz.dtype).unsqueeze(-1)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(x.transpose(1, 2).reshape(batch * seqlen, -1))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = dt.reshape(-1, batch, seqlen).transpose(0, 1)
        B = B.reshape(batch, seqlen, -1).transpose(1, 2).contiguous()
        C = C.reshape(batch, seqlen, -1).transpose(1, 2).contiguous()
        y = selective_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = y.transpose(1, 2)
        out = self.out_proj(y)
        return out


# ---------------------------------------------------------------------------
# mamba_ssm/modules/block.py :: Block (fused_add_norm=False branch, no MLP --
# SPMamba's own MambaBlock never sets mlp_cls, so `self.mlp is None` always)
# ---------------------------------------------------------------------------
class MambaResBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# mamba_ssm/models/mixer_seq_simple.py :: _init_weights
# ---------------------------------------------------------------------------
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ---------------------------------------------------------------------------
# look2hear/models/SPMamba.py :: MambaBlock
# (RMSNorm swapped from mamba_ssm.ops.triton.layernorm.RMSNorm -- the Triton
# kernel is unavailable here -- for torch.nn.RMSNorm, which computes the
# identical eps-stabilized root-mean-square normalization formula on CPU/CUDA)
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super(MambaBlock, self).__init__()
        self.bidirectional = bidirectional
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                MambaResBlock(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    norm_cls=partial(nn.RMSNorm, eps=1e-5),
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    MambaResBlock(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        norm_cls=partial(nn.RMSNorm, eps=1e-5),
                    )
                )
        else:
            self.backward_blocks = None

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual)
            back_residual = (
                (backward_f + back_residual) if back_residual is not None else backward_f
            )

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)

        return residual


# ---------------------------------------------------------------------------
# look2hear/layers/stft_tfgn.py :: Stft (native torch.stft/istft branch, the
# real code path taken whenever `is_torch_1_10_plus` -- true on this torch
# version) + look2hear/models/SPMamba.py :: STFTEncoder / STFTDecoder
# ---------------------------------------------------------------------------
class Stft(nn.Module):
    def __init__(self, n_fft=512, win_length=None, hop_length=128, window="hann", center=True):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = window

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None):
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length, dtype=input.dtype, device=input.device)

        output = torch.stft(
            input,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        output = torch.view_as_real(output)
        # (Batch, Freq, Frames, 2) -> (Batch, Frames, Freq, 2)
        output = output.transpose(1, 2)
        if multi_channel:
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(1, 2)
        return output, None

    def inverse(self, input: torch.Tensor, ilens: torch.Tensor = None):
        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length, dtype=input.real.dtype, device=input.device)
        input = input.transpose(1, 2)
        length = ilens.max() if ilens is not None else None
        wavs = torch.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
            return_complex=False,
        )
        return wavs, ilens


class STFTEncoder(nn.Module):
    """STFT encoder for speech enhancement and separation"""

    def __init__(self, n_fft=512, win_length=None, hop_length=128, window="hann", center=True):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window, center=center
        )
        self._output_dim = n_fft // 2 + 1

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        spectrum, flens = self.stft(input, ilens)
        spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
        return spectrum, flens


class STFTDecoder(nn.Module):
    """STFT decoder for speech enhancement and separation"""

    def __init__(self, n_fft=512, win_length=None, hop_length=128, window="hann", center=True):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window, center=center
        )

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        bs = input.size(0)
        if input.dim() == 4:
            multi_channel = True
            input = input.transpose(1, 2).reshape(-1, input.size(1), input.size(3))
        else:
            multi_channel = False

        wav, wav_lens = self.stft.inverse(input, ilens)

        if multi_channel:
            wav = wav.reshape(bs, -1, wav.size(1)).transpose(1, 2)

        return wav, wav_lens


def new_complex_like(ref: torch.Tensor, real_imag: Tuple[torch.Tensor, torch.Tensor]):
    return torch.complex(*real_imag)


# ---------------------------------------------------------------------------
# look2hear/models/SPMamba.py :: LayerNormalization4D / LayerNormalization4DCF
# ---------------------------------------------------------------------------
class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        stat_dim = (1,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        stat_dim = (1, 3)
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


def get_layer(l_name, library=nn):
    """look2hear/utils/get_layer_from_string.py :: get_layer, transcribed."""
    all_torch_layers = [x for x in dir(library)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) != 1:
        raise NotImplementedError(f"Layer with name {l_name} not found in {library}.")
    return getattr(library, match[0])


# ---------------------------------------------------------------------------
# look2hear/models/SPMamba.py :: GridNetBlock (dual-path intra-/inter-chunk
# Mamba + multi-head 4D attention -- transcribed verbatim)
# ---------------------------------------------------------------------------
class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_mamba = MambaBlock(in_channels, 1, True)
        self.intra_linear = nn.ConvTranspose1d(in_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_mamba = MambaBlock(in_channels, 1, True)
        self.inter_linear = nn.ConvTranspose1d(in_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_mamba(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn = self.inter_mamba(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))
            all_K.append(self["attn_conv_K_%d" % ii](batch))
            all_V.append(self["attn_conv_V_%d" % ii](batch))

        Q = torch.cat(all_Q, dim=0)
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)
        V = V.transpose(1, 2)
        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        V = torch.matmul(attn_mat, V)

        V = V.reshape(old_shape)
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])
        batch = batch.transpose(0, 1)
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])
        batch = self["attn_concat_proj"](batch)

        out = batch + inter_rnn
        return out


# ---------------------------------------------------------------------------
# look2hear/models/SPMamba.py :: SPMamba (top-level model)
# ---------------------------------------------------------------------------
class SPMamba(nn.Module):
    def __init__(
        self,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=2,
        lstm_hidden_units=32,
        attn_n_head=4,
        attn_approx_qk_dim=64,
        emb_dim=8,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        self.enc = STFTEncoder(n_fft, n_fft, stride, window=window)
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        return F.pad(input_tensor, (0, target_len - input_tensor.shape[-1]))

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: batched mono/multi-channel audio [B, N] or [B, N, M]
        Returns:
            [B, n_srcs, N] separated waveforms
        """
        if input.ndim == 1:
            input = input.unsqueeze(0).unsqueeze(2)
        elif input.ndim == 2:
            input = input.unsqueeze(2)
        elif input.ndim == 3:
            input = input.permute(0, 2, 1).contiguous()
        n_samples = input.shape[1]
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)
        input = input / mix_std_
        ilens = torch.ones(input.shape[0], dtype=torch.long, device=input.device) * n_samples
        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)

        batch = self.deconv(batch)

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.reshape(-1, n_frames, n_freqs), ilens)[0]

        batch = self.pad2(batch.reshape([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_

        batch = [batch[:, src] for src in range(self.num_spk)]
        return torch.stack(batch, dim=1)


# --- Menagerie staging wrapper --------------------------------------------------
#
# Real constructor defaults are n_fft=128/emb_dim=48/n_layers=6/lstm_hidden=192;
# shrunk here (emb_dim=8, n_layers=2, small n_fft/hop) purely for trace speed --
# every mechanism (STFT encode, dual-path bidirectional Mamba SSM scan,
# multi-head 4D attention fusion, iSTFT decode) still fires.

_SR = 8000
_DURATION_SAMPLES = 1600  # 0.2s @ 8kHz, several STFT frames


def build_spmamba():
    torch.manual_seed(0)
    return SPMamba(
        n_srcs=2,
        n_fft=64,
        stride=32,
        n_layers=2,
        lstm_hidden_units=32,
        attn_n_head=2,
        attn_approx_qk_dim=64,
        emb_dim=8,
        emb_ks=4,
        emb_hs=1,
    )


def example_input_spmamba():
    return torch.randn(1, _DURATION_SAMPLES)


MENAGERIE_ENTRIES = [
    ("SPMamba", build_spmamba, example_input_spmamba, 2024, "PORT"),
]
