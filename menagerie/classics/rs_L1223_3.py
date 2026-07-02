# FAITHFUL PORT of js3611/Deep-MRI-Reconstruction @ master (original framework: PyTorch 0.4)
#
# cq615/CRNN-MRI's own README points to js3611/Deep-MRI-Reconstruction's
# cascadenet_pytorch/model_pytorch.py + cascadenet_pytorch/kspace_pytorch.py as "the
# pytorch implementation of our work" (CRNN-MRI, TMI 2019 / MICCAI 2018): a Convolutional
# Recurrent Neural Network for dynamic MR image reconstruction from undersampled k-space.
# A bidirectional convolutional-RNN layer (BCRNNlayer/CRNNcell, evolving over both the
# temporal frame axis and the unrolled-iteration axis) is interleaved with 3 more
# iteration-recurrent convolutional layers and a data-consistency-in-k-space layer,
# repeated for `nc` unrolled iterations, each iteration correcting the reconstructed
# image toward the acquired (undersampled) k-space samples.
#
# The original code cannot run unmodified on modern torch: it calls the pre-1.8
# `torch.fft(x, 2, normalized=...)` / `torch.ifft(...)` functions (removed; torch now
# only exposes `torch.fft.fft2`/`torch.fft.ifft2` on complex tensors), and it hardcodes
# `.cuda()` inside `BCRNNlayer.forward`/`CRNN_MRI.forward` for the zero-initialized
# hidden state (device-specific, breaks on CPU-only tracing). This port:
#   - replaces `torch.fft(x, 2, normalized=self.normalized)` with the numerically
#     equivalent `torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x), norm='ortho'))`
#     (and the inverse analogously for `torch.ifft`), preserving the exact permute
#     bookkeeping the original used to route the "real/imag pair" trailing axis;
#   - replaces the hardcoded `.cuda()` hidden-state init with `torch.zeros(..., device=input.device)`
#     so the model traces on CPU exactly as it would on GPU;
#   - wraps `self.dcs` (a plain Python list of `DataConsistencyInKspace` submodules in
#     the original `CRNN_MRI.__init__`) in `nn.ModuleList` so parameters register
#     correctly under `nn.Module` (the original relied on CUDA `.cuda()` calls elsewhere
#     to move these submodules, which is not a viable substitute for registration).
# Every other computation -- the CRNNcell gating, the BCRNNlayer forward/backward
# temporal sweep, the 4-layer iteration-recurrent conv stack, the data-consistency
# k-space correction, and the unrolled-iteration loop over `nc` -- is unchanged from
# the source. Only base-lib deps: torch.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


# ---- cascadenet_pytorch/kspace_pytorch.py (ported: torch.fft/ifft -> torch.fft.fft2/ifft2) ----


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


class DataConsistencyInKspace(nn.Module):
    """Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
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

        norm = "ortho" if self.normalized else "backward"
        k = torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x.contiguous()), norm=norm))
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.view_as_real(
            torch.fft.ifft2(torch.view_as_complex(out.contiguous()), norm=norm)
        )

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


# ---- cascadenet_pytorch/model_pytorch.py (ported: hardcoded .cuda() -> device-aware zeros,
# ---- self.dcs list -> nn.ModuleList) ----


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = torch.zeros(size_h, device=input.device, dtype=input.dtype)

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i - 1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNN_MRI(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """

    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.bcrnn = BCRNNlayer(n_ch, nf, ks)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding=ks // 2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm="ortho"))
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, m, test=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq * n_batch, self.nf, width, height]
        hid_init = torch.zeros(size_h, device=x.device, dtype=x.dtype)

        for j in range(self.nd - 1):
            net["t0_x%d" % j] = hid_init

        for i in range(1, self.nc + 1):
            x = x.permute(4, 0, 1, 2, 3)
            x = x.contiguous()
            net["t%d_x0" % (i - 1)] = net["t%d_x0" % (i - 1)].view(
                n_seq, n_batch, self.nf, width, height
            )
            net["t%d_x0" % i] = self.bcrnn(x, net["t%d_x0" % (i - 1)], test)
            net["t%d_x0" % i] = net["t%d_x0" % i].view(-1, self.nf, width, height)

            net["t%d_x1" % i] = self.conv1_x(net["t%d_x0" % i])
            net["t%d_h1" % i] = self.conv1_h(net["t%d_x1" % (i - 1)])
            net["t%d_x1" % i] = self.relu(net["t%d_h1" % i] + net["t%d_x1" % i])

            net["t%d_x2" % i] = self.conv2_x(net["t%d_x1" % i])
            net["t%d_h2" % i] = self.conv2_h(net["t%d_x2" % (i - 1)])
            net["t%d_x2" % i] = self.relu(net["t%d_h2" % i] + net["t%d_x2" % i])

            net["t%d_x3" % i] = self.conv3_x(net["t%d_x2" % i])
            net["t%d_h3" % i] = self.conv3_h(net["t%d_x3" % (i - 1)])
            net["t%d_x3" % i] = self.relu(net["t%d_h3" % i] + net["t%d_x3" % i])

            net["t%d_x4" % i] = self.conv4_x(net["t%d_x3" % i])

            x = x.view(-1, n_ch, width, height)
            net["t%d_out" % i] = x + net["t%d_x4" % i]

            net["t%d_out" % i] = net["t%d_out" % i].view(-1, n_batch, n_ch, width, height)
            net["t%d_out" % i] = net["t%d_out" % i].permute(1, 2, 3, 4, 0)
            net["t%d_out" % i].contiguous()
            net["t%d_out" % i] = self.dcs[i - 1].perform(net["t%d_out" % i], k, m)
            x = net["t%d_out" % i]

            # clean up i-1
            if test:
                to_delete = [key for key in net if ("t%d" % (i - 1)) in key]

                for elt in to_delete:
                    del net[elt]

        return net["t%d_out" % i]


# ---- tiny build/example (architecture unmodified; sizes shrunk for fast tracing) ----


def build_crnn_mri():
    """Tiny CRNN-MRI (few filters/iterations, small spatial/temporal size) for tracing.
    Architecture is unmodified from the ported source. `nd` is kept at the source's
    default of 5: the forward pass always references exactly 4 fixed iteration-recurrent
    conv layers (conv1_x..conv4_x) regardless of `nd`, so `nd` must stay >= 5 for the
    `t0_x%d` placeholder keys (initialized for `j in range(nd - 1)`) to cover every key
    the forward body looks up at iteration 0."""
    torch.manual_seed(0)
    model = CRNN_MRI(n_ch=2, nf=8, ks=3, nc=2, nd=5)
    model.eval()
    return model


def example_input_crnn_mri():
    torch.manual_seed(0)
    n_batch, n_ch, width, height, n_seq = 1, 2, 16, 16, 4
    x = torch.randn(n_batch, n_ch, width, height, n_seq)
    k = torch.randn(n_batch, n_ch, width, height, n_seq)
    m = torch.randint(0, 2, (n_batch, n_ch, width, height, n_seq)).float()
    return (x, k, m)


MENAGERIE_ENTRIES = [
    ("CRNN-MRI", "build_crnn_mri", "example_input_crnn_mri", 2019, MENAGERIE_ZOO),
]
