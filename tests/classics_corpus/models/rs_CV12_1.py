# SOURCE: vendored from kevinzakka/recurrent-visual-attention @ master (model.py, modules.py)
# SOURCE: vendored from mperezcarrasco/PyTorch-DAGMM @ master (model.py)
# SOURCE: vendored from RomainSabathe/dagmm @ master
# (model.py, compression_networks.py, estimation_networks.py, gmm.py)
# SOURCE: vendored from DmitryUlyanov/deep-image-prior @ master
# (models/__init__.py, models/skip.py, models/common.py)
# SOURCE: vendored from GitiHubi/deepAI @ master (GTC_2018_Lab.ipynb)
# SOURCE: vendored from huochaitiantang/pytorch-deep-image-matting @ master (core/net.py)
# SOURCE: vendored from MinhNguyenIKM/dem_hyperelasticity @ master (MultiLayerNet.py)
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

MENAGERIE_ZOO = "vendored-pytorch"


class Retina:
    """Visual retina from recurrent-visual-attention."""

    def __init__(self, g: int, k: int, s: int) -> None:
        """Initialize the foveated retina extractor."""
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x: Tensor, loc: Tensor) -> Tensor:
        """Extract and flatten foveated glimpses."""
        phi = []
        size = self.g
        for _ in range(self.k):
            phi.append(self.extract_patch(x, loc, size))
            size = int(self.s * size)
        for i in range(1, len(phi)):
            kernel = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], kernel)
        return torch.cat(phi, 1).view(phi[0].shape[0], -1)

    def extract_patch(self, x: Tensor, loc: Tensor, size: int) -> Tensor:
        """Extract one square patch per image."""
        batch, _, height, _ = x.shape
        start = self.denormalize(height, loc)
        end = start + size
        padded = F.pad(x, (size // 2, size // 2, size // 2, size // 2))
        patch = []
        for i in range(batch):
            patch.append(padded[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])
        return torch.stack(patch)

    def denormalize(self, size: int, coords: Tensor) -> Tensor:
        """Convert normalized coordinates to pixel coordinates."""
        return (0.5 * ((coords + 1.0) * size)).long()

    def exceeds(self, from_x: int, to_x: int, from_y: int, to_y: int, size: int) -> bool:
        """Return whether a patch exceeds image boundaries."""
        return (from_x < 0) or (from_y < 0) or (to_x > size) or (to_y > size)


class GlimpseNetwork(nn.Module):
    """Glimpse network from recurrent-visual-attention."""

    def __init__(self, h_g: int, h_l: int, g: int, k: int, s: int, c: int) -> None:
        """Initialize the glimpse network."""
        super().__init__()
        self.retina = Retina(g, k, s)
        self.fc1 = nn.Linear(k * g * g * c, h_g)
        self.fc2 = nn.Linear(2, h_l)
        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x: Tensor, l_t_prev: Tensor) -> Tensor:
        """Compute the glimpse representation."""
        phi = self.retina.foveate(x, l_t_prev)
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))
        return F.relu(self.fc3(phi_out) + self.fc4(l_out))


class CoreNetwork(nn.Module):
    """Core recurrent network from recurrent-visual-attention."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize the core recurrent network."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t: Tensor, h_t_prev: Tensor) -> Tensor:
        """Update the hidden state."""
        return F.relu(self.i2h(g_t) + self.h2h(h_t_prev))


class ActionNetwork(nn.Module):
    """Action classifier from recurrent-visual-attention."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize the action classifier."""
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t: Tensor) -> Tensor:
        """Return log class probabilities."""
        return F.log_softmax(self.fc(h_t), dim=1)


class LocationNetwork(nn.Module):
    """Location policy from recurrent-visual-attention."""

    def __init__(self, input_size: int, output_size: int, std: float) -> None:
        """Initialize the location policy."""
        super().__init__()
        self.std = std
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t: Tensor) -> tuple[Tensor, Tensor]:
        """Sample the next fixation and return its log probability."""
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        l_t = torch.distributions.Normal(mu, self.std).rsample().detach()
        log_pi = Normal(mu, self.std).log_prob(l_t).sum(dim=1)
        return log_pi, torch.clamp(l_t, -1, 1)


class BaselineNetwork(nn.Module):
    """Baseline regressor from recurrent-visual-attention."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize the baseline network."""
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t: Tensor) -> Tensor:
        """Return baseline prediction."""
        return self.fc(h_t.detach())


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention."""

    def __init__(
        self,
        g: int,
        k: int,
        s: int,
        c: int,
        h_g: int,
        h_l: int,
        std: float,
        hidden_size: int,
        num_classes: int,
    ) -> None:
        """Initialize the recurrent attention model."""
        super().__init__()
        self.std = std
        self.sensor = GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = CoreNetwork(hidden_size, hidden_size)
        self.locator = LocationNetwork(hidden_size, 2, std)
        self.classifier = ActionNetwork(hidden_size, num_classes)
        self.baseliner = BaselineNetwork(hidden_size, 1)

    def forward(
        self,
        x: Tensor,
        l_t_prev: Tensor,
        h_t_prev: Tensor,
        last: bool = False,
    ) -> tuple[Tensor, ...]:
        """Run one recurrent attention timestep."""
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()
        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi
        return h_t, l_t, b_t, log_pi


class DAGMMKDD(nn.Module):
    """DAGMM KDDCup network from PyTorch-DAGMM."""

    def __init__(self, n_gmm: int = 2, z_dim: int = 1) -> None:
        """Initialize the KDD DAGMM network."""
        super().__init__()
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)
        self.fc9 = nn.Linear(z_dim + 2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x: Tensor) -> Tensor:
        """Encode an input vector."""
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x: Tensor) -> Tensor:
        """Decode a latent vector."""
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)

    def estimate(self, z: Tensor) -> Tensor:
        """Estimate Gaussian mixture memberships."""
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5, training=self.training)
        return F.softmax(self.fc10(h), dim=1)

    def compute_reconstruction(self, x: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor]:
        """Compute DAGMM reconstruction features."""
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the DAGMM forward pass."""
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma


class AccountingEncoder(nn.Module):
    """Deep accounting anomaly encoder from GitiHubi/deepAI."""

    def __init__(self, input_dim: int = 618) -> None:
        """Initialize the notebook encoder."""
        super().__init__()
        self.encoder_L1 = nn.Linear(in_features=input_dim, out_features=512, bias=True)
        nn.init.xavier_uniform_(self.encoder_L1.weight)
        self.encoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L2 = nn.Linear(512, 256, bias=True)
        nn.init.xavier_uniform_(self.encoder_L2.weight)
        self.encoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L3 = nn.Linear(256, 128, bias=True)
        nn.init.xavier_uniform_(self.encoder_L3.weight)
        self.encoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L4 = nn.Linear(128, 64, bias=True)
        nn.init.xavier_uniform_(self.encoder_L4.weight)
        self.encoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L5 = nn.Linear(64, 32, bias=True)
        nn.init.xavier_uniform_(self.encoder_L5.weight)
        self.encoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L6 = nn.Linear(32, 16, bias=True)
        nn.init.xavier_uniform_(self.encoder_L6.weight)
        self.encoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L7 = nn.Linear(16, 8, bias=True)
        nn.init.xavier_uniform_(self.encoder_L7.weight)
        self.encoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L8 = nn.Linear(8, 4, bias=True)
        nn.init.xavier_uniform_(self.encoder_L8.weight)
        self.encoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.encoder_L9 = nn.Linear(4, 3, bias=True)
        nn.init.xavier_uniform_(self.encoder_L9.weight)
        self.encoder_R9 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.dropout = nn.Dropout(p=0.0, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a journal-entry feature vector."""
        x = self.encoder_R1(self.dropout(self.encoder_L1(x)))
        x = self.encoder_R2(self.dropout(self.encoder_L2(x)))
        x = self.encoder_R3(self.dropout(self.encoder_L3(x)))
        x = self.encoder_R4(self.dropout(self.encoder_L4(x)))
        x = self.encoder_R5(self.dropout(self.encoder_L5(x)))
        x = self.encoder_R6(self.dropout(self.encoder_L6(x)))
        x = self.encoder_R7(self.dropout(self.encoder_L7(x)))
        x = self.encoder_R8(self.dropout(self.encoder_L8(x)))
        return self.encoder_R9(self.encoder_L9(x))


class AccountingDecoder(nn.Module):
    """Deep accounting anomaly decoder from GitiHubi/deepAI."""

    def __init__(self, output_dim: int = 618) -> None:
        """Initialize the notebook decoder."""
        super().__init__()
        self.decoder_L1 = nn.Linear(in_features=3, out_features=4, bias=True)
        nn.init.xavier_uniform_(self.decoder_L1.weight)
        self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L2 = nn.Linear(4, 8, bias=True)
        nn.init.xavier_uniform_(self.decoder_L2.weight)
        self.decoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L3 = nn.Linear(8, 16, bias=True)
        nn.init.xavier_uniform_(self.decoder_L3.weight)
        self.decoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L4 = nn.Linear(16, 32, bias=True)
        nn.init.xavier_uniform_(self.decoder_L4.weight)
        self.decoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L5 = nn.Linear(32, 64, bias=True)
        nn.init.xavier_uniform_(self.decoder_L5.weight)
        self.decoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L6 = nn.Linear(64, 128, bias=True)
        nn.init.xavier_uniform_(self.decoder_L6.weight)
        self.decoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L7 = nn.Linear(128, 256, bias=True)
        nn.init.xavier_uniform_(self.decoder_L7.weight)
        self.decoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L8 = nn.Linear(256, 512, bias=True)
        nn.init.xavier_uniform_(self.decoder_L8.weight)
        self.decoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.decoder_L9 = nn.Linear(in_features=512, out_features=output_dim, bias=True)
        nn.init.xavier_uniform_(self.decoder_L9.weight)
        self.decoder_R9 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.dropout = nn.Dropout(p=0.0, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Decode a journal-entry latent vector."""
        x = self.decoder_R1(self.dropout(self.decoder_L1(x)))
        x = self.decoder_R2(self.dropout(self.decoder_L2(x)))
        x = self.decoder_R3(self.dropout(self.decoder_L3(x)))
        x = self.decoder_R4(self.dropout(self.decoder_L4(x)))
        x = self.decoder_R5(self.dropout(self.decoder_L5(x)))
        x = self.decoder_R6(self.dropout(self.decoder_L6(x)))
        x = self.decoder_R7(self.dropout(self.decoder_L7(x)))
        x = self.decoder_R8(self.dropout(self.decoder_L8(x)))
        return self.decoder_R9(self.decoder_L9(x))


class AccountingAutoencoder(nn.Module):
    """Deep accounting anomaly autoencoder from GitiHubi/deepAI."""

    def __init__(self, input_dim: int = 618) -> None:
        """Initialize the accounting autoencoder."""
        super().__init__()
        self.encoder = AccountingEncoder(input_dim=input_dim)
        self.decoder = AccountingDecoder(output_dim=input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct a journal-entry feature vector."""
        return self.decoder(self.encoder(x))


class DeepImageMattingVGG16(nn.Module):
    """VGG16-style deep image matting model from pytorch-deep-image-matting."""

    def __init__(self, stage: int = 1) -> None:
        """Initialize the matting network."""
        super().__init__()
        self.stage = stage
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.deconv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2, bias=True)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2, bias=True)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2, bias=True)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=True)
        if self.stage == 2:
            for param in self.parameters():
                param.requires_grad = False
        if self.stage in {2, 3}:
            self.refine_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | int]:
        """Run the matting network."""
        x11 = F.relu(self.conv1_1(x))
        x12 = F.relu(self.conv1_2(x11))
        x1p, id1 = F.max_pool2d(x12, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x21 = F.relu(self.conv2_1(x1p))
        x22 = F.relu(self.conv2_2(x21))
        x2p, id2 = F.max_pool2d(x22, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x31 = F.relu(self.conv3_1(x2p))
        x32 = F.relu(self.conv3_2(x31))
        x33 = F.relu(self.conv3_3(x32))
        x3p, id3 = F.max_pool2d(x33, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x41 = F.relu(self.conv4_1(x3p))
        x42 = F.relu(self.conv4_2(x41))
        x43 = F.relu(self.conv4_3(x42))
        x4p, id4 = F.max_pool2d(x43, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x51 = F.relu(self.conv5_1(x4p))
        x52 = F.relu(self.conv5_2(x51))
        x53 = F.relu(self.conv5_3(x52))
        x5p, id5 = F.max_pool2d(x53, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x61 = F.relu(self.conv6_1(x5p))
        x61d = F.relu(self.deconv6_1(x61))
        x5d = F.max_unpool2d(x61d, id5, kernel_size=2, stride=2) + x53
        x51d = F.relu(self.deconv5_1(x5d))
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2) + x43
        x41d = F.relu(self.deconv4_1(x4d))
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2) + x33
        x31d = F.relu(self.deconv3_1(x3d))
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2) + x22
        x21d = F.relu(self.deconv2_1(x2d))
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2) + x12
        x12d = F.relu(self.deconv1_1(x1d))
        raw_alpha = self.deconv1(x12d)
        pred_mattes = torch.sigmoid(raw_alpha)
        if self.stage <= 1:
            return pred_mattes, 0
        refine0 = torch.cat((x[:, :3, :, :], pred_mattes), 1)
        refine1 = F.relu(self.refine_conv1(refine0))
        refine2 = F.relu(self.refine_conv2(refine1))
        refine3 = F.relu(self.refine_conv3(refine2))
        pred_refine = self.refine_pred(refine3)
        return pred_mattes, torch.sigmoid(raw_alpha + pred_refine)


class MultiLayerNet(nn.Module):
    """Deep Energy Method multilayer net from dem_hyperelasticity."""

    def __init__(self, d_in: int, hidden: int, d_out: int) -> None:
        """Initialize the DEM multilayer net."""
        super().__init__()
        self.linear1 = nn.Linear(d_in, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, hidden)
        self.linear4 = nn.Linear(hidden, d_out)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
        nn.init.constant_(self.linear3.bias, 0.0)
        nn.init.constant_(self.linear4.bias, 0.0)
        nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear3.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear4.weight, mean=0, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the DEM multilayer net."""
        y1 = torch.tanh(self.linear1(x))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2))
        return self.linear4(y3)


class CompressionNetworkArrhythmia(nn.Module):
    """Arrhythmia compression network from RomainSabathe/dagmm."""

    def __init__(self) -> None:
        """Initialize the compression network."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(274, 10), nn.Tanh(), nn.Linear(10, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 274))
        self._reconstruction_loss = nn.MSELoss()

    def forward(self, input: Tensor) -> Tensor:
        """Encode and reconstruct an input."""
        return self.decoder(self.encoder(input))

    def encode(self, input: Tensor) -> Tensor:
        """Encode an input."""
        return self.encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        """Decode a latent vector."""
        return self.decoder(input)

    def reconstruction_loss(self, input: Tensor, target: Tensor) -> Tensor:
        """Return reconstruction loss."""
        return self._reconstruction_loss(self(input), target)


class EstimationNetworkArrhythmia(nn.Module):
    """Arrhythmia estimation network from RomainSabathe/dagmm."""

    def __init__(self) -> None:
        """Initialize the estimation network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Estimate mixture memberships."""
        return self.net(input)


class Mixture(nn.Module):
    """Gaussian mixture component from RomainSabathe/dagmm."""

    def __init__(self, dimension_embedding: int) -> None:
        """Initialize one Gaussian component."""
        super().__init__()
        self.dimension_embedding = dimension_embedding
        self.Phi = nn.Parameter(torch.rand(1), requires_grad=False)
        self.mu = nn.Parameter(2.0 * torch.rand(dimension_embedding) - 0.5, requires_grad=False)
        self.Sigma = nn.Parameter(torch.eye(dimension_embedding), requires_grad=False)
        self.register_buffer("eps_Sigma", torch.eye(dimension_embedding) * 1.0e-8)

    def forward(self, samples: Tensor, with_log: bool = True) -> Tensor:
        """Return component densities or energies."""
        inv_sigma = torch.inverse(self.Sigma)
        det_sigma = torch.det(self.Sigma).clamp_min(1.0e-12)
        outputs = []
        normalizer = torch.sqrt(2.0 * math.pi * det_sigma)
        for sample in samples:
            diff = (sample - self.mu).view(-1, 1)
            exponent = -0.5 * torch.mm(torch.mm(diff.view(1, -1), inv_sigma), diff)
            out = (self.Phi * torch.exp(exponent)).view(1) / normalizer
            if with_log:
                out = -torch.log(out)
            outputs.append(out.squeeze(0))
        return torch.stack(outputs)

    def _update_parameters(self, samples: Tensor, affiliations: Tensor) -> None:
        """Update mixture parameters from affiliated samples."""
        if not self.training:
            return
        phi = torch.mean(affiliations)
        self.Phi.data = phi.data.view_as(self.Phi)
        num = torch.zeros_like(self.mu)
        for i in range(samples.shape[0]):
            num += affiliations[i] * samples[i, :]
        denom = torch.sum(affiliations)
        self.mu.data = (num / denom).data
        sigma_num = torch.zeros_like(self.Sigma)
        for i in range(samples.shape[0]):
            diff = (samples[i, :] - self.mu).view(-1, 1)
            sigma_num += affiliations[i] * torch.mm(diff, diff.view(1, -1))
        self.Sigma.data = (sigma_num / denom).data + self.eps_Sigma


class GMM(nn.Module):
    """Gaussian mixture model from RomainSabathe/dagmm."""

    def __init__(self, num_mixtures: int, dimension_embedding: int) -> None:
        """Initialize the Gaussian mixture model."""
        super().__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding
        self.mixtures = nn.ModuleList([Mixture(dimension_embedding) for _ in range(num_mixtures)])

    def forward(self, inputs: Tensor) -> Tensor:
        """Return negative log mixture density."""
        out = None
        for mixture in self.mixtures:
            to_add = mixture(inputs, with_log=False)
            out = to_add if out is None else out + to_add
        if out is None:
            raise ValueError("GMM requires at least one mixture")
        return -torch.log(out)

    def _update_mixtures_parameters(
        self,
        samples: Tensor,
        mixtures_affiliations: Tensor,
    ) -> None:
        """Update every mixture from sample affiliations."""
        if not self.training:
            return
        for i, mixture in enumerate(self.mixtures):
            mixture._update_parameters(samples, mixtures_affiliations[:, i])


class DAGMM(nn.Module):
    """Composable DAGMM from RomainSabathe/dagmm."""

    def __init__(
        self,
        compression_module: nn.Module,
        estimation_module: nn.Module,
        gmm_module: GMM,
    ) -> None:
        """Initialize the composable DAGMM."""
        super().__init__()
        self.compressor = compression_module
        self.estimator = estimation_module
        self.gmm = gmm_module

    def forward(self, input: Tensor) -> Tensor:
        """Run compression, estimation, and GMM energy."""
        encoded = self.compressor.encode(input)
        decoded = self.compressor.decode(encoded)
        relative_ed = relative_euclidean_distance(input, decoded).view(-1, 1)
        cosine_sim = relative_ed.view(-1, 1)
        latent_vectors = torch.cat([encoded, relative_ed, cosine_sim], dim=1)
        if self.training:
            mixtures_affiliations = self.estimator(latent_vectors)
            self.gmm._update_mixtures_parameters(latent_vectors, mixtures_affiliations)
        return self.gmm(latent_vectors)


class DAGMMArrhythmia(DAGMM):
    """Arrhythmia DAGMM from RomainSabathe/dagmm."""

    def __init__(self) -> None:
        """Initialize the Arrhythmia DAGMM."""
        super().__init__(
            compression_module=CompressionNetworkArrhythmia(),
            estimation_module=EstimationNetworkArrhythmia(),
            gmm_module=GMM(num_mixtures=2, dimension_embedding=4),
        )


class Concat(nn.Module):
    """Concat helper from deep-image-prior models/common.py."""

    def __init__(self, dim: int, *args: nn.Module) -> None:
        """Initialize a module that concatenates child outputs."""
        super().__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input: Tensor) -> Tensor:
        """Run children on the same input and concatenate cropped outputs."""
        inputs = [module(input) for module in self._modules.values()]
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)
        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2 : diff2 + target_shape2, diff3 : diff3 + target_shape3])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self) -> int:
        """Return number of child modules."""
        return len(self._modules)


class Swish(nn.Module):
    """Swish activation from deep-image-prior models/common.py."""

    def __init__(self) -> None:
        """Initialize Swish."""
        super().__init__()
        self.s = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Apply Swish activation."""
        return x * self.s(x)


def act(act_fun: str | type[nn.Module] = "LeakyReLU") -> nn.Module:
    """Return an activation module from the DIP helper spelling."""
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        if act_fun == "Swish":
            return Swish()
        if act_fun == "ELU":
            return nn.ELU()
        if act_fun == "none":
            return nn.Sequential()
        raise ValueError(f"Unsupported activation: {act_fun}")
    return act_fun()


def bn(num_features: int) -> nn.Module:
    """Return a DIP batch-normalization layer."""
    return nn.BatchNorm2d(num_features)


def conv(
    in_f: int,
    out_f: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
    pad: str = "zero",
    downsample_mode: str = "stride",
) -> nn.Module:
    """Return a DIP convolution block."""
    downsampler = None
    if stride != 1 and downsample_mode != "stride":
        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
        stride = 1
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = [layer for layer in [padder, convolver, downsampler] if layer is not None]
    return nn.Sequential(*layers)


def seq_add(seq: nn.Sequential, module: nn.Module) -> None:
    """Append a module using the numbering convention in deep-image-prior."""
    seq.add_module(str(len(seq) + 1), module)


def skip(
    num_input_channels: int = 2,
    num_output_channels: int = 3,
    num_channels_down: list[int] | None = None,
    num_channels_up: list[int] | None = None,
    num_channels_skip: list[int] | None = None,
    filter_size_down: int = 3,
    filter_size_up: int = 3,
    filter_skip_size: int = 1,
    need_sigmoid: bool = True,
    need_bias: bool = True,
    pad: str = "zero",
    upsample_mode: str = "nearest",
    downsample_mode: str = "stride",
    act_fun: str | type[nn.Module] = "LeakyReLU",
    need1x1_up: bool = True,
) -> nn.Sequential:
    """Assemble the official Deep Image Prior skip encoder-decoder."""
    if num_channels_down is None:
        num_channels_down = [16, 32, 64, 128, 128]
    if num_channels_up is None:
        num_channels_up = [16, 32, 64, 128, 128]
    if num_channels_skip is None:
        num_channels_skip = [4, 4, 4, 4, 4]
    if not len(num_channels_down) == len(num_channels_up) == len(num_channels_skip):
        raise ValueError("DIP channel lists must have the same length")
    n_scales = len(num_channels_down)
    upsample_modes = [upsample_mode] * n_scales
    downsample_modes = [downsample_mode] * n_scales
    filter_size_downs = [filter_size_down] * n_scales
    filter_size_ups = [filter_size_up] * n_scales
    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_branch = nn.Sequential()
        if num_channels_skip[i] != 0:
            seq_add(model_tmp, Concat(1, skip_branch, deeper))
        else:
            seq_add(model_tmp, deeper)
        next_depth = num_channels_up[i + 1] if i < last_scale else num_channels_down[i]
        seq_add(model_tmp, bn(num_channels_skip[i] + next_depth))
        if num_channels_skip[i] != 0:
            seq_add(
                skip_branch,
                conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad),
            )
            seq_add(skip_branch, bn(num_channels_skip[i]))
            seq_add(skip_branch, act(act_fun))
        seq_add(
            deeper,
            conv(
                input_depth,
                num_channels_down[i],
                filter_size_downs[i],
                2,
                bias=need_bias,
                pad=pad,
                downsample_mode=downsample_modes[i],
            ),
        )
        seq_add(deeper, bn(num_channels_down[i]))
        seq_add(deeper, act(act_fun))
        seq_add(
            deeper,
            conv(
                num_channels_down[i],
                num_channels_down[i],
                filter_size_downs[i],
                bias=need_bias,
                pad=pad,
            ),
        )
        seq_add(deeper, bn(num_channels_down[i]))
        seq_add(deeper, act(act_fun))
        deeper_main = nn.Sequential()
        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            seq_add(deeper, deeper_main)
            k = num_channels_up[i + 1]
        seq_add(deeper, nn.Upsample(scale_factor=2, mode=upsample_modes[i]))
        seq_add(
            model_tmp,
            conv(
                num_channels_skip[i] + k,
                num_channels_up[i],
                filter_size_ups[i],
                1,
                bias=need_bias,
                pad=pad,
            ),
        )
        seq_add(model_tmp, bn(num_channels_up[i]))
        seq_add(model_tmp, act(act_fun))
        if need1x1_up:
            seq_add(
                model_tmp, conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad)
            )
            seq_add(model_tmp, bn(num_channels_up[i]))
            seq_add(model_tmp, act(act_fun))
        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    seq_add(model, conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        seq_add(model, nn.Sigmoid())
    return model


def relative_euclidean_distance(x1: Tensor, x2: Tensor, eps: Tensor | None = None) -> Tensor:
    """Return relative Euclidean distance."""
    if eps is None:
        eps = torch.tensor([1.0e-8], device=x1.device, dtype=x1.dtype)
    num = torch.norm(x1 - x2, p=2, dim=1)
    denom = torch.norm(x1, p=2, dim=1)
    return num / torch.maximum(denom, eps)


def cosine_similarity(x1: Tensor, x2: Tensor, eps: Tensor | None = None) -> Tensor:
    """Return cosine similarity."""
    if eps is None:
        eps = torch.tensor([1.0e-8], device=x1.device, dtype=x1.dtype)
    dot_prod = torch.sum(x1 * x2, dim=1)
    dist_x1 = torch.norm(x1, p=2, dim=1)
    dist_x2 = torch.norm(x2, p=2, dim=1)
    return dot_prod / torch.maximum(dist_x1 * dist_x2, eps)


def build_deep_active_vision_recurrent_model() -> RecurrentAttention:
    """Build a tiny recurrent visual attention model."""
    return RecurrentAttention(
        g=4,
        k=2,
        s=2,
        c=1,
        h_g=8,
        h_l=8,
        std=0.17,
        hidden_size=16,
        num_classes=5,
    )


def example_input_deep_active_vision_recurrent_model() -> tuple[Tensor, Tensor, Tensor]:
    """Return an example image, location, and hidden state."""
    return torch.randn(1, 1, 16, 16), torch.zeros(1, 2), torch.zeros(1, 16)


def build_deep_autoencoder_for_accounting_anomalies() -> AccountingAutoencoder:
    """Build the deep accounting anomaly autoencoder."""
    return AccountingAutoencoder(input_dim=618).eval()


def example_input_deep_autoencoder_for_accounting_anomalies() -> Tensor:
    """Return an example accounting feature vector."""
    return torch.randn(1, 618)


def build_deep_autoencoding_gaussian_mixture_model_for_journal_entries() -> DAGMMKDD:
    """Build a tiny KDD-style DAGMM."""
    return DAGMMKDD(n_gmm=2, z_dim=2).eval()


def example_input_deep_autoencoding_gaussian_mixture_model_for_journal_entries() -> Tensor:
    """Return an example DAGMM input."""
    return torch.randn(2, 118)


def build_dagmm_hep() -> DAGMMArrhythmia:
    """Build a tiny Arrhythmia-style DAGMM from RomainSabathe/dagmm."""
    return DAGMMArrhythmia().eval()


def example_input_dagmm_hep() -> Tensor:
    """Return an example Arrhythmia DAGMM input."""
    return torch.randn(2, 274)


def build_deep_energy_method() -> MultiLayerNet:
    """Build the DEM multilayer network."""
    return MultiLayerNet(d_in=2, hidden=8, d_out=2).eval()


def example_input_deep_energy_method() -> Tensor:
    """Return an example DEM coordinate input."""
    return torch.randn(4, 2)


def build_deep_image_matting() -> DeepImageMattingVGG16:
    """Build the deep image matting VGG16 model."""
    return DeepImageMattingVGG16(stage=1).eval()


def example_input_deep_image_matting() -> Tensor:
    """Return an example RGB+trimap input."""
    return torch.randn(1, 4, 32, 32)


def build_deep_image_prior() -> nn.Sequential:
    """Build a tiny Deep Image Prior skip network."""
    return skip(
        num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[4, 8],
        num_channels_up=[4, 8],
        num_channels_skip=[2, 2],
        upsample_mode="nearest",
        downsample_mode="stride",
        pad="zero",
    ).eval()


def example_input_deep_image_prior() -> Tensor:
    """Return an example DIP noise input."""
    return torch.randn(1, 2, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "Deep Active Vision recurrent model",
        "build_deep_active_vision_recurrent_model",
        "example_input_deep_active_vision_recurrent_model",
        2014,
        "CV12-366",
    ),
    (
        "Deep Autoencoder for Accounting Anomalies",
        "build_deep_autoencoder_for_accounting_anomalies",
        "example_input_deep_autoencoder_for_accounting_anomalies",
        2017,
        "CV12-369",
    ),
    (
        "Deep Autoencoding Gaussian Mixture Model for Journal Entries",
        "build_deep_autoencoding_gaussian_mixture_model_for_journal_entries",
        "example_input_deep_autoencoding_gaussian_mixture_model_for_journal_entries",
        2018,
        "CV12-370",
    ),
    (
        "Deep Autoencoding Gaussian Mixture Model for LHC (DAGMM-HEP)",
        "build_dagmm_hep",
        "example_input_dagmm_hep",
        2018,
        "CV12-371",
    ),
    (
        "Deep Energy Method (DEM)",
        "build_deep_energy_method",
        "example_input_deep_energy_method",
        2019,
        "CV12-381",
    ),
    (
        "Deep Image Matting",
        "build_deep_image_matting",
        "example_input_deep_image_matting",
        2017,
        "CV12-391",
    ),
    (
        "Deep Image Prior",
        "build_deep_image_prior",
        "example_input_deep_image_prior",
        2018,
        "CV12-392",
    ),
]
