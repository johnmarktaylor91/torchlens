# SOURCE: vendored from cq615/kt-Dynamic-MRI-Reconstruction @ master
# https://github.com/cq615/kt-Dynamic-MRI-Reconstruction
#
# Files vendored (concatenated, imports collapsed into this file):
#   network/layers.py  (fftshift/ifftshift helpers, data_consistency, complex_multiply,
#                        DataConsistencyInKspace, CRNNcell, BCRNNlayer, TransformDataInXfSpaceTA)
#   network/kt_NEXT.py (xf_CNN, CRNN_MRI, kt_NEXT_model)
#
# k-t NEXT (Qin et al., MICCAI 2019, "k-t NEXT: Dynamic MRI Reconstruction Exploiting
# Spatio-Temporal Correlations") is a cascade of x-f-domain CNN blocks and x-t-domain
# convolutional-RNN blocks, each followed by a k-space data-consistency layer, applied to
# accelerated dynamic (cardiac cine) MRI reconstruction.
#
# Only mechanical fixes applied (no architectural change):
#  1. `torch.fft(x, signal_ndim, normalized=...)` / `torch.ifft(...)` -- the pre-1.7
#     real-tensor-pair complex FFT API -- was removed from torch entirely (torch 2.x only
#     exposes the `torch.fft` submodule). Replaced 1:1 with the numerically-equivalent
#     modern call `torch.view_as_real(torch.fft.fft(torch.view_as_complex(x.contiguous()),
#     dim=<matching axis>, norm='ortho'))` (verified bit-exact against the old semantics via
#     a NumPy cross-check: signal_ndim=1 -> FFT over the axis immediately before the
#     trailing real/imag pair dim; signal_ndim=2 -> FFT over the two axes before it).
#  2. `Variable(...).cuda()` hard-coded device placement in `BCRNNlayer.forward` replaced
#     with `.to(input.device)` so the module runs on CPU for tracing (same tensor
#     semantics, just device-parametric instead of CUDA-only).
#  3. `from network.layers import *` collapsed into this single file.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# network/layers.py
# ------------------------------------------------------------------
def _fftshift(x, axes, offset=1):
    """Apply (i)fftshift to x. axes: tuple of axes to apply ifftshift, e.g. axes=(-1)."""
    x_shape = x.shape
    ndim = len(x_shape)
    axes = [(ndim + ax) % ndim for ax in axes]

    for ax in axes:
        if x_shape[ax] == 1:
            continue
        n = x_shape[ax]
        half_n = (n + offset) // 2
        curr_slice = [slice(0, half_n) if i == ax else slice(x_shape[i]) for i in range(ndim)]
        curr_slice_2 = [
            slice(half_n, x_shape[i]) if i == ax else slice(x_shape[i]) for i in range(ndim)
        ]
        x = torch.cat([x[curr_slice_2], x[curr_slice]], dim=ax)
    return x


def fftshift_pytorch(x, axes):
    return _fftshift(x, axes, offset=1)


def ifftshift_pytorch(x, axes):
    return _fftshift(x, axes, offset=0)


def _fft_nd(x, signal_ndim, normalized):
    """Modern-torch replacement for the removed `torch.fft(x, signal_ndim, normalized)`
    real-tensor-pair API. x has a trailing size-2 (real, imag) axis; FFT is applied over
    the `signal_ndim` axes immediately preceding it. Verified numerically identical to the
    old API via NumPy cross-check (see module header)."""
    xc = torch.view_as_complex(x.contiguous())
    dims = tuple(range(-signal_ndim, 0))
    norm = "ortho" if normalized else "backward"
    return torch.view_as_real(torch.fft.fftn(xc, dim=dims, norm=norm))


def _ifft_nd(x, signal_ndim, normalized):
    """Modern-torch replacement for the removed `torch.ifft(x, signal_ndim, normalized)`."""
    xc = torch.view_as_complex(x.contiguous())
    dims = tuple(range(-signal_ndim, 0))
    norm = "ortho" if normalized else "backward"
    return torch.view_as_real(torch.fft.ifftn(xc, dim=dims, norm=norm))


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


def complex_multiply(x, y, u, v):
    """
    Computes (x+iy) * (u+iv) = (x * u - y * v) + (x * v + y * u)i = z1 + iz2
    Returns (real z1, imaginary z2)
    """
    z1 = x * u - y * v
    z2 = x * v + y * u
    return torch.stack((z1, z2), dim=-1)


class DataConsistencyInKspace(nn.Module):
    """Create data consistency operator.

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of
    the input. This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D
    data) and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm="ortho"):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == "ortho"
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        if x.dim() == 4:  # input is 2D
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = _fft_nd(x, 2, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = _ifft_nd(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    """

    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.i2h = nn.Conv2d(
            input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation
        )
        self.h2h = nn.Conv2d(
            hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation
        )
        if self.iteration:
            self.ih2ih = nn.Conv2d(
                hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2
            )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden, hidden_iteration=None):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)
            hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid + hid_to_hid)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration]
               test: True if in test mode, False if in train mode
    """

    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.iteration = iteration
        self.CRNN_model = CRNNcell(
            self.input_size, self.hidden_size, self.kernel_size, dilation, iteration=self.iteration
        )

    def forward(self, input, input_iteration=None, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = torch.zeros(size_h, device=input.device, dtype=input.dtype)

        output_f = []
        output_b = []
        if input_iteration is not None:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden, input_iteration[i])
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden, input_iteration[nt - i - 1])
                output_b.append(hidden)
        else:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden)
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden)
                output_b.append(hidden)

        output_f = torch.cat(output_f)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class TransformDataInXfSpaceTA(nn.Module):
    def __init__(self, divide_by_n=False, norm=True):
        super(TransformDataInXfSpaceTA, self).__init__()
        self.normalized = norm
        self.divide_by_n = divide_by_n

    def forward(self, x, k0, mask):
        return self.perform(x, k0, mask)

    def perform(self, x, k0, mask):
        """
        transform to x-f space with subtraction of average temporal frame
        :param x: input image with shape [n, 2, nx, ny, nt]
        :param mask: undersampling mask
        :return: difference data; DC baseline
        """
        # temporally average kspace and image data
        x = x.permute(0, 4, 2, 3, 1)
        mask = mask.permute(0, 4, 2, 3, 1)
        k0 = k0.permute(0, 4, 2, 3, 1)
        k = _fft_nd(x, 2, normalized=self.normalized)
        if self.divide_by_n:
            k_avg = torch.div(torch.sum(k, 1), k.shape[1])
        else:
            k_avg = torch.div(torch.sum(k, 1), torch.clamp(torch.sum(mask, 1), min=1))

        nb, nx, ny, nc = k_avg.shape
        k_avg = k_avg.view(nb, 1, nx, ny, nc)
        # repeat the temporal frame
        k_avg = k_avg.repeat(1, k.shape[1], 1, 1, 1)

        # subtract the temporal average frame
        k_diff = torch.sub(k, k_avg)
        x_diff = _ifft_nd(k_diff, 2, normalized=self.normalized)

        # transform to x-f space to get the baseline
        k_avg = data_consistency(k_avg, k0, mask)
        x_avg = _ifft_nd(k_avg, 2, normalized=self.normalized)

        x_avg = x_avg.permute(0, 2, 3, 1, 4)  # [n, nx, ny, nt, 2]
        x_f_avg = fftshift_pytorch(
            _fft_nd(ifftshift_pytorch(x_avg, axes=[-2]), 1, normalized=self.normalized), axes=[-2]
        )
        x_f_avg = x_f_avg.permute(0, 4, 1, 2, 3)

        # difference data
        x_diff = x_diff.permute(0, 2, 3, 1, 4)  # [n, nx, ny, nt, 2]
        x_f_diff = fftshift_pytorch(
            _fft_nd(ifftshift_pytorch(x_diff, axes=[-2]), 1, normalized=self.normalized), axes=[-2]
        )
        x_f_diff = x_f_diff.permute(0, 4, 1, 2, 3)

        return x_f_diff, x_f_avg


# ------------------------------------------------------------------
# network/kt_NEXT.py
# ------------------------------------------------------------------
def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def xf_CNN(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl="lrelu", conv_dim=2, n_out=None):
    """xf-CNN block in x-f domain"""
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    if not n_out:
        n_out = n_ch

    pad_conv = 1

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=dilation, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    nll = relu if nl == "relu" else lrelu

    layers = [conv_1, nll()]
    for i in range(nd - 2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class CRNN_MRI(nn.Module):
    """
    CRNN-MRI block in image domain
    RNN evolves over temporal dimension only
    """

    def __init__(self, n_ch, nf=64, ks=3, dilation=2):
        super(CRNN_MRI, self).__init__()
        self.nf = nf
        self.ks = ks

        self.bcrnn_1 = BCRNNlayer(n_ch, nf, ks, dilation=1)
        self.bcrnn_2 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_3 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_4 = BCRNNlayer(nf, nf, ks, dilation)

        self.conv4_x = nn.Conv2d(nf, 2, ks, padding=ks // 2)

    def forward(self, x, test=False):
        n_batch, n_ch, width, length, n_seq = x.size()

        x = x.permute(4, 0, 1, 2, 3)

        out = self.bcrnn_1(x, None, test)
        out = self.bcrnn_2(out, None, test)
        out = self.bcrnn_3(out, None, test)
        out = self.bcrnn_4(out, None, test)
        out = out.view(-1, self.nf, width, length)
        out = self.conv4_x(out)

        out = out.view(-1, n_batch, 2, width, length)
        out = out.permute(1, 2, 3, 4, 0)

        return out


class kt_NEXT_model(nn.Module):
    """
    network architecture for k-t NEXT
    """

    def __init__(self, n_channels=2, nd=5, nc=5, nf=64, dilation=3):
        super(kt_NEXT_model, self).__init__()
        self.nc = nc
        self.dilation = dilation
        xf_conv_blocks = []
        xt_conv_blocks = []
        dcs_xf = []
        dcs_xt = []
        tdxf = []

        for i in range(nc):
            xf_conv_blocks.append(
                xf_CNN(2, nd, nf, n_out=n_channels, conv_dim=2, dilation=self.dilation)
            )
            xt_conv_blocks.append(CRNN_MRI(n_channels, nf, dilation=self.dilation))
            dcs_xf.append(DataConsistencyInKspace(norm="ortho"))
            dcs_xt.append(DataConsistencyInKspace(norm="ortho"))
            tdxf.append(TransformDataInXfSpaceTA(i > 0, norm=True))

        self.xf_conv_blocks = nn.ModuleList(xf_conv_blocks)
        self.xt_conv_blocks = nn.ModuleList(xt_conv_blocks)
        self.dcs_xf = nn.ModuleList(dcs_xf)
        self.dcs_xt = nn.ModuleList(dcs_xt)
        self.tdxf = nn.ModuleList(tdxf)

    def forward(self, x, k, m):
        net = {}
        net_xf = {}
        for i in range(self.nc):
            # x-f domain reconstruction
            xf, xf_avg = self.tdxf[i].perform(x, k, m)
            nb, nc, nx, ny, nt = xf.shape
            xf = xf.permute(0, 3, 1, 2, 4)
            xf = xf.reshape(-1, nc, nx, nt)
            xf_out = self.xf_conv_blocks[i](xf)
            xf_out = xf_out.reshape(-1, ny, 2, nx, nt)
            xf_out = xf_out.permute(0, 2, 3, 1, 4)  # (n, nc, nx, ny, nt)
            xf_out = xf_out + xf_avg

            # transform signal from x-f domain to image domain
            out_img = fftshift_pytorch(
                _ifft_nd(
                    ifftshift_pytorch(xf_out.permute(0, 2, 3, 4, 1), axes=[-2]), 1, normalized=True
                ),
                axes=[-2],
            )
            out_img = out_img.permute(0, 4, 1, 2, 3)
            x = self.dcs_xf[i].perform(out_img, k, m)

            # image domain reconstruction
            out = self.xt_conv_blocks[i](x)
            x = x + out
            x = self.dcs_xt[i].perform(x, k, m)
            net["t%d" % i] = x
            net_xf["t%d" % i] = xf_out

        return net_xf, net


# ------------------------------------------------------------------
# Menagerie staging entry points
# ------------------------------------------------------------------
def build_ktnext():
    # nc (number of cascade stages) shrunk from the paper default (5) for a fast trace;
    # architecture (per-stage xf-CNN + CRNN-MRI + data-consistency triplet) unchanged.
    return kt_NEXT_model(n_channels=2, nd=3, nc=2, nf=8, dilation=2)


def example_input_ktnext():
    # (batch, real/imag=2, nx, ny, nt) undersampled image, k-space data, and sampling mask
    # -- matches `prep_input()` / `main_kt_NEXT.py`'s `to_tensor_format` layout.
    n, c, nx, ny, nt = 1, 2, 16, 16, 4
    x = torch.randn(n, c, nx, ny, nt)
    k = torch.randn(n, c, nx, ny, nt)
    mask = torch.randint(0, 2, (n, c, nx, ny, nt)).float()
    return (x, k, mask)


MENAGERIE_ENTRIES = [
    ("k-t NEXT", "build_ktnext", "example_input_ktnext", "2019", "vendored-pytorch"),
]
