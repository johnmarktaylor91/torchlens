# SOURCE: vendored from sgybupt/SketchHealer @ 4188475d05e865c644081efff4a28dddbafe32ed
# https://raw.githubusercontent.com/sgybupt/SketchHealer/4188475d05e865c644081efff4a28dddbafe32ed/encoder.py
# https://raw.githubusercontent.com/sgybupt/SketchHealer/4188475d05e865c644081efff4a28dddbafe32ed/decoder.py
#
# Su et al. 2020 (BMVC 2020) "SketchHealer: A Graph-to-Sequence Network for Recreating
# Partial Human Sketches" -- a graph-convolutional VAE encoder over per-stroke image
# patches ("FeatureExtraction"/"GCNProcessor"/"EncoderGCN") feeding a Sketch-RNN-style
# bivariate-Gaussian-mixture LSTM decoder ("DecoderRNN"). The GCN encoder treats each
# sketch as a graph of stroke-patch nodes connected by a (learned/derived) adjacency
# matrix -- this is the paper's actual architectural contribution over a plain
# Sketch-RNN VAE, so it must be vendored rather than mapped to a library class.
#
# No architectural changes were made; only mechanical fixes for import isolation:
#   - `encoder.py`/`decoder.py` both did `from hyper_params import hp` (a module-level
#     singleton `HParams()` instance); that file is reproduced here inline as a small
#     `SimpleNamespace`-like `_HP` class with only the fields these two files reference
#     (`Nz`, `M`, `dec_hidden_size`, `dropout`, `Nmax`), since the rest of `hyper_params.py`
#     is training/data-loading configuration unrelated to the traced architecture.
#   - `encoder.py`'s two `if __name__ == '__main__':` smoke-test blocks (including the
#     stray top-level `class GCNProcessor` / `class EncoderGCN` that follow the first
#     `__main__` guard in the original file -- Python still executes class bodies after
#     an `if __name__ == '__main__': ... exit(0)` block at *import* time since `exit(0)`
#     only fires when run as a script) are dropped; only the four `nn.Module` classes are
#     kept.
#   - `decoder.py`'s `F.softmax(...)` calls (no `dim=`) are 2.x-syntax-incompatible only
#     in the sense of emitting a deprecation warning upstream; behavior is unchanged, and
#     an explicit `dim=` matching the tensor's last axis was added to keep torch 2.x quiet
#     without altering outputs (upstream relied on the legacy dim-inference fallback).

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HP:
    """Minimal stand-in for the upstream `hyper_params.HParams` singleton, restricted
    to the fields `encoder.py`/`decoder.py` actually read."""

    Nz = 32  # encoder output size (paper default: 128)
    M = 6  # number of bivariate-Gaussian mixture components (paper default: 20)
    dec_hidden_size = 48  # decoder LSTM hidden size (paper default: 512)
    dropout = 0.0
    Nmax = 5  # max stroke count for a tiny synthetic sketch


hp = _HP()


class FeatureExtractionBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 2, 2, 0)  # 64
        self.conv2 = nn.Conv2d(8, 32, 2, 2, 0)  # 32
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 0)  # 16
        self.conv4 = nn.Conv2d(64, 128, 2, 2, 0)  # 8
        self.conv5 = nn.Conv2d(128, 256, 2, 2, 0)  # 4
        self.conv6 = nn.Conv2d(256, 512, 2, 2, 0)  # 2
        self.maxpooling1 = nn.MaxPool2d(2)  # 1

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x: torch.Tensor = self.maxpooling1(x)
        x = x.view(-1, 512)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self, graph_num=0, graph_size=0, train=True):
        super().__init__()
        self.graph_num = graph_num
        self.graph_size = graph_size
        assert self.graph_num
        assert self.graph_size
        self.featureGenerator = FeatureExtractionBasic()
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: (batch_size, graph_num, 3, graph_size, graph_size)
        :return:
        """
        if inputs.shape[0] != 1:
            tmp_batch = 1  # split count
            tmp_result = []
            inputs = inputs.view(tmp_batch, -1, 1, self.graph_size, self.graph_size)
            for i in range(tmp_batch):
                tmp_result.append(self.featureGenerator(inputs[i]))
            result = torch.cat(tmp_result).view(-1, self.graph_num, 512)  # (batch, 30, 1000)
        else:
            result = self.featureGenerator(
                inputs.view(-1, 1, self.graph_size, self.graph_size)
            ).view(-1, self.graph_num, 512)
        result = self.bn1(result.view(-1, 512)).view(-1, self.graph_num, 512)
        return result


class GCNProcessor(nn.Module):
    def __init__(self, graph_num, out_f_num, bias_bool=True):
        super().__init__()
        # shapes
        self.graph_num = graph_num
        self.out_f_num = out_f_num
        self.bias_bool = bias_bool
        # params
        self.weight = nn.Parameter(
            torch.randn(512, self.out_f_num, dtype=torch.float, requires_grad=True)
        )
        self.bias = nn.Parameter(
            torch.randn(self.graph_num, self.out_f_num, dtype=torch.float, requires_grad=True)
        )
        self.merge = nn.Parameter(
            torch.randn(1, self.graph_num, dtype=torch.float, requires_grad=True)
        )

        self.bn1 = nn.BatchNorm1d(out_f_num)

    def params_reset(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.merge, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bias, 0)

    def set_trainable(self, train=True):
        for param in self.parameters():
            param.requires_grad = train

    def forward(self, X, A):
        """
        :param X: (batch, graph_num, in_feature_num)
        :param A: (batch, graph_num, graph_num)
        :return:
        """
        x = torch.matmul(A, X)
        if self.bias_bool:
            x = torch.matmul(x, self.weight) + self.bias
        else:
            x = torch.matmul(x, self.weight)

        result = torch.sum(x, dim=1)
        return self.bn1(result)


class EncoderGCN(nn.Module):
    def __init__(
        self,
        graph_num,
        graph_size,
        out_f_num,
        out_mu_sigma_num,
        bias_need=False,
        FE_trainable=False,
    ):
        super(EncoderGCN, self).__init__()
        self.graph_num = graph_num
        self.graph_size = graph_size
        self.out_f_num = out_f_num
        self.bias_need = bias_need
        self.out_mu_sigma_num = out_mu_sigma_num
        assert self.graph_num
        assert self.graph_size
        assert self.out_f_num
        assert self.out_mu_sigma_num

        # model
        self.feature_extractor = FeatureExtraction(self.graph_num, self.graph_size, FE_trainable)
        self.gcn = GCNProcessor(self.graph_num, self.out_f_num, self.bias_need)

        # z, mu, sigma
        self.fc_mu = nn.Linear(self.out_f_num, self.out_mu_sigma_num)
        self.fc_sigma = nn.Linear(self.out_f_num, self.out_mu_sigma_num)

    def forward(self, input_imgs, adj_matrix):
        """
        return z, mu, sigma
        :param input_imgs: (batch_size, graph_num, 3, graph_size, graph_size)
        :param adj_matrix: (batch_size, graph_num, graph_num)
        """
        x = self.feature_extractor(input_imgs)
        x = self.gcn(x, adj_matrix)
        final = torch.tanh(x)

        # generate mu sigma
        mu = self.fc_mu(final)
        sigma = self.fc_sigma(final)
        sigma_e = torch.exp(sigma / 2.0)

        # normal sample
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n
        return z, mu, sigma, final


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # to init **hidden and cell** from z:
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(5 + hp.Nz, hp.dec_hidden_size, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the LSTM with the whole input in one shot
        # and use all outputs contained in 'outputs',
        # while in generate mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])  # trajectory
        params_pen = params[-1]  # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # preprocess params:
        if self.training:
            len_out = hp.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=-1).view(len_out, -1, hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell


class SketchHealer(nn.Module):
    """Wraps the encoder/decoder pair so the model traces as a single module: the
    real (encoder, decoder) split matches the upstream `Model` class, which owns
    separate `self.encoder`/`self.decoder` submodules and separate optimizers."""

    def __init__(self, graph_num, graph_size, out_f_num, out_mu_sigma_num):
        super().__init__()
        self.encoder = EncoderGCN(
            graph_num, graph_size, out_f_num, out_mu_sigma_num, bias_need=False
        )
        self.decoder = DecoderRNN()

    def forward(self, input_imgs, adj_matrix, dec_inputs):
        z, mu, sigma, _ = self.encoder(input_imgs, adj_matrix)
        z_stack = torch.stack([z] * dec_inputs.size(0))
        lstm_inputs = torch.cat([dec_inputs, z_stack], 2)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = self.decoder(lstm_inputs, z)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma


def build_sketchhealer():
    graph_num = 6  # tiny stroke-patch graph (paper default: 21)
    # `FeatureExtractionBasic` applies six stride-2 conv layers followed by one more
    # stride-2 maxpool (`maxpooling1`), so `graph_size` must be >= 128 (2**7) for the
    # spatial dims to survive down to the final 1x1 output; this matches the paper's
    # own default and is the smallest power of 2 that keeps the real seven-downsample
    # stack intact.
    graph_size = 128
    out_f_num = 24
    model = SketchHealer(graph_num, graph_size, out_f_num, hp.Nz)
    model.eval()
    return model


def example_input_sketchhealer():
    batch = 2
    graph_num = 6
    graph_size = 128
    input_imgs = torch.rand(batch, graph_num, 1, graph_size, graph_size)
    adj_matrix = torch.stack([torch.eye(graph_num) for _ in range(batch)])
    dec_inputs = torch.rand(hp.Nmax + 1, batch, 5)
    return (input_imgs, adj_matrix, dec_inputs)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SketchHealer", "build_sketchhealer", "example_input_sketchhealer", 2020, "vendored"),
]
