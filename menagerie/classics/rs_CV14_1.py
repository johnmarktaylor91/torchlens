# SOURCE: vendored from aAbdz/DeepACSON @ master (models/model.py)
# SOURCE: vendored from Ghadjeres/DeepBach @ master (DeepBach/voice_model.py)
# SOURCE: vendored from junyanz/interactive-deep-colorization @ master (models/pytorch/model.py)
# SOURCE: vendored from profjsb/deepCR @ master (deepCR/unet.py, deepCR/parts.py)
# SOURCE: vendored from qinnzou/DeepCrack @ master (codes/model/deepcrack.py)
# SOURCE: vendored from DIVA-DIA/DeepDIVA @ master (models/semantic_segmentation/BabyUnet.py)
# SOURCE: vendored from jbohnslav/deepethogram @ master (flow_generator/models/TinyMotionNet.py, components.py)
# SOURCE: vendored from alzayats/DeepFish @ master (DeepFish/models/unet.py)
# SOURCE: vendored from cabooster/DeepCAD @ master (DeepCAD_pytorch/model_3DUnet.py, buildingblocks.py)
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ZOO = "vendored-pytorch"


class DeepACSONResBlock(nn.Module):
    """Residual block from DeepACSON."""

    def __init__(self, in_channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        in_channels
            Number of input and output channels.
        """
        super().__init__()
        self.res_blk = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        return x + self.res_blk(x)


class DeepACSONBasicBlock(nn.Module):
    """Convolution, batch norm, and ReLU block from DeepACSON."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
    ) -> None:
        """Initialize the basic block."""
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the basic block."""
        return self.basic_block(x)


class DeepACSONUp(nn.Module):
    """Upsampling block from DeepACSON."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the upsampling block."""
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv1 = DeepACSONBasicBlock(in_channels, out_channels, 3, 1, 1)
        self.conv2 = DeepACSONBasicBlock(out_channels, out_channels, 3, 1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Upsample and fuse decoder and encoder features."""
        x1 = self.upsample(x1)
        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )
        return self.conv2(self.conv1(torch.cat([x2, x1], dim=1)))


class DeepACSONDown(nn.Module):
    """Downsampling block from DeepACSON."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the downsampling block."""
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DeepACSONResBlock(in_channels),
            DeepACSONBasicBlock(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool3d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual processing and max pooling."""
        return self.maxpool_conv(x)


class DeepACSONUNet3D(nn.Module):
    """3D U-Net with residual down blocks from DeepACSON."""

    def __init__(self, n_channels: int, n_classes: int, base_channels: int = 8) -> None:
        """Initialize the DeepACSON U-Net."""
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.conv_first = DeepACSONBasicBlock(n_channels, c1, 3, 1, 1)
        self.down1 = DeepACSONDown(c1, c2)
        self.down2 = DeepACSONDown(c2, c3)
        self.down3 = DeepACSONDown(c3, c4)
        self.down4 = DeepACSONDown(c4, c5)
        self.up1 = DeepACSONUp(c5 + c4, c4)
        self.up2 = DeepACSONUp(c4 + c3, c3)
        self.up3 = DeepACSONUp(c3 + c2, c2)
        self.up4 = DeepACSONUp(c2 + c1, c1)
        self.conv_last = nn.Conv3d(c1, n_classes, kernel_size=1)
        self.softmax = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Run DeepACSON segmentation."""
        x1 = self.conv_first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.softmax(self.conv_last(x))


class _DeepBachMetadata:
    """Minimal metadata record matching DeepBach's dataset contract."""

    def __init__(self, num_values: int) -> None:
        """Initialize metadata cardinality."""
        self.num_values = num_values


class _DeepBachDataset:
    """Small stand-in for the fields read by DeepBach's VoiceModel."""

    def __init__(self, num_voices: int = 4, notes_per_voice: int = 8, metas: int = 3) -> None:
        """Initialize tiny DeepBach dataset metadata."""
        self.num_voices = num_voices
        self.note2index_dicts = [dict.fromkeys(range(notes_per_voice)) for _ in range(num_voices)]
        self.metadatas = [_DeepBachMetadata(metas)]


def _deepbach_init_hidden(
    num_layers: int, batch_size: int, lstm_hidden_size: int
) -> tuple[Tensor, Tensor]:
    """Create zero LSTM states as in DeepBach helpers."""
    hidden = torch.zeros(num_layers, batch_size, lstm_hidden_size)
    cell = torch.zeros(num_layers, batch_size, lstm_hidden_size)
    return hidden, cell


class DeepBachVoiceModel(nn.Module):
    """VoiceModel from DeepBach."""

    def __init__(
        self,
        dataset: _DeepBachDataset,
        main_voice_index: int,
        note_embedding_dim: int,
        meta_embedding_dim: int,
        num_layers: int,
        lstm_hidden_size: int,
        dropout_lstm: float,
        hidden_size_linear: int = 200,
    ) -> None:
        """Initialize the DeepBach voice model."""
        super().__init__()
        self.dataset = dataset
        self.main_voice_index = main_voice_index
        self.note_embedding_dim = note_embedding_dim
        self.meta_embedding_dim = meta_embedding_dim
        self.num_notes_per_voice = [len(d) for d in dataset.note2index_dicts]
        self.num_voices = self.dataset.num_voices
        self.num_metas_per_voice = [metadata.num_values for metadata in dataset.metadatas] + [
            self.num_voices
        ]
        self.num_metas = len(self.dataset.metadatas) + 1
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_lstm = dropout_lstm
        self.hidden_size_linear = hidden_size_linear
        self.other_voices_indexes = [i for i in range(self.num_voices) if i != main_voice_index]
        self.note_embeddings = nn.ModuleList(
            [nn.Embedding(num_notes, note_embedding_dim) for num_notes in self.num_notes_per_voice]
        )
        self.meta_embeddings = nn.ModuleList(
            [nn.Embedding(num_metas, meta_embedding_dim) for num_metas in self.num_metas_per_voice]
        )
        recurrent_size = note_embedding_dim * self.num_voices + meta_embedding_dim * self.num_metas
        self.lstm_left = nn.LSTM(
            input_size=recurrent_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True,
        )
        self.lstm_right = nn.LSTM(
            input_size=recurrent_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True,
        )
        self.mlp_center = nn.Sequential(
            nn.Linear(
                note_embedding_dim * (self.num_voices - 1) + meta_embedding_dim * self.num_metas,
                hidden_size_linear,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, lstm_hidden_size),
        )
        self.mlp_predictions = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 3, hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, self.num_notes_per_voice[main_voice_index]),
        )

    def forward(
        self, notes: tuple[Tensor, Tensor, Tensor], metas: tuple[Tensor, Tensor, Tensor]
    ) -> Tensor:
        """Predict the masked center voice token."""
        batch_size, _, _ = notes[0].size()
        ln, cn, rn = notes
        ln, rn = [t.transpose(1, 2) for t in (ln, rn)]
        notes_embedded = self.embed((ln, cn, rn), "note")
        metas_embedded = self.embed(metas, "meta")
        input_embedded = [
            torch.cat([note_tensor, meta_tensor], 2) if note_tensor is not None else None
            for note_tensor, meta_tensor in zip(notes_embedded, metas_embedded)
        ]
        left, center, right = input_embedded
        hidden = _deepbach_init_hidden(self.num_layers, batch_size, self.lstm_hidden_size)
        left, _ = self.lstm_left(left, hidden)
        left = left[:, -1, :]
        center = self.mlp_center(center[:, 0, :])
        hidden = _deepbach_init_hidden(self.num_layers, batch_size, self.lstm_hidden_size)
        right, _ = self.lstm_right(right, hidden)
        right = right[:, -1, :]
        return self.mlp_predictions(torch.cat([left, center, right], 1))

    def embed(
        self, notes_or_metas: tuple[Tensor, Tensor, Tensor], embedding_type: str
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Embed note or metadata tensors."""
        if embedding_type == "note":
            embeddings = self.note_embeddings
            embedding_dim = self.note_embedding_dim
            other_voices_indexes = self.other_voices_indexes
        else:
            embeddings = self.meta_embeddings
            embedding_dim = self.meta_embedding_dim
            other_voices_indexes = range(self.num_metas)
        batch_size, timesteps_left_ticks, num_voices = notes_or_metas[0].size()
        _, timesteps_right_ticks, _ = notes_or_metas[2].size()
        left, center, right = notes_or_metas
        left_embedded = torch.cat(
            [
                embeddings[voice_id](left[:, :, voice_id])[:, :, None, :]
                for voice_id in range(num_voices)
            ],
            2,
        )
        right_embedded = torch.cat(
            [
                embeddings[voice_id](right[:, :, voice_id])[:, :, None, :]
                for voice_id in range(num_voices)
            ],
            2,
        )
        center_embedded = torch.cat(
            [
                embeddings[voice_id](center[:, k].unsqueeze(1))
                for k, voice_id in enumerate(other_voices_indexes)
            ],
            1,
        )
        center_embedded = center_embedded.view(
            batch_size, 1, len(other_voices_indexes) * embedding_dim
        )
        left_embedded = left_embedded.view(
            batch_size, timesteps_left_ticks, num_voices * embedding_dim
        )
        right_embedded = right_embedded.view(
            batch_size, timesteps_right_ticks, num_voices * embedding_dim
        )
        return left_embedded, center_embedded, right_embedded


class SIGGRAPHGenerator(nn.Module):
    """PyTorch SIGGRAPHGenerator from interactive deep colorization."""

    def __init__(self, dist: bool = False) -> None:
        """Initialize the colorization generator."""
        super().__init__()
        self.dist = dist
        use_bias = True
        norm_layer = nn.BatchNorm2d
        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(64),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(128),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(256),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(512),
        )
        self.model5 = self._dilated_block(512, 512)
        self.model6 = self._dilated_block(512, 512)
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(512),
        )
        self.model8up = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=use_bias))
        self.model3short8 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=use_bias))
        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(256),
        )
        self.model9up = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=use_bias))
        self.model2short9 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=use_bias))
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(128),
        )
        self.model10up = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=use_bias))
        self.model1short10 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=use_bias))
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=use_bias),
            nn.LeakyReLU(0.2),
        )
        self.model_class = nn.Sequential(nn.Conv2d(256, 529, 1, 1, 0, bias=use_bias))
        self.model_out = nn.Sequential(nn.Conv2d(128, 2, 1, 1, 0, bias=use_bias), nn.Tanh())
        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=4, mode="nearest"))
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

    def _dilated_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create the original repeated dilated convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(
        self, input_a: Tensor, input_b: Tensor, mask_b: Tensor, maskcent: float = 0.0
    ) -> Tensor:
        """Colorize from L-channel, user ab hints, and hint mask."""
        input_a = torch.as_tensor(input_a)[None, :, :, :]
        input_b = torch.as_tensor(input_b)[None, :, :, :]
        mask_b = torch.as_tensor(mask_b)[None, :, :, :] - maskcent
        conv1_2 = self.model1(torch.cat((input_a / 100.0, input_b / 110.0, mask_b), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)
        return out_reg * 110


class DeepCRDoubleConv(nn.Module):
    """Double convolution block from deepCR."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the deepCR double convolution block."""
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply two convolution layers."""
        return self.double_conv(x)


class DeepCRInConv(nn.Module):
    """Input convolution block from deepCR."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the input block."""
        super().__init__()
        self.conv = DeepCRDoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the input block."""
        return self.conv(x)


class DeepCRDown(nn.Module):
    """Downsampling block from deepCR."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the downsampling block."""
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DeepCRDoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """Apply max pooling and double convolution."""
        return self.mpconv(x)


class DeepCRUp(nn.Module):
    """Upsampling block from deepCR."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the upsampling block."""
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DeepCRDoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Upsample and fuse with an encoder skip tensor."""
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class DeepCROutConv(nn.Module):
    """Output convolution block from deepCR."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the output block."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the output convolution."""
        return self.conv(x)


class DeepCRUNet2Sigmoid(nn.Module):
    """Two-level U-Net with sigmoid output from deepCR."""

    def __init__(self, n_channels: int, n_classes: int, hidden: int = 8) -> None:
        """Initialize the deepCR U-Net."""
        super().__init__()
        self.inc = DeepCRInConv(n_channels, hidden)
        self.down1 = DeepCRDown(hidden, hidden * 2)
        self.up8 = DeepCRUp(hidden * 2, hidden)
        self.outc = DeepCROutConv(hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run cosmic-ray mask prediction."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        return torch.sigmoid(self.outc(x))


def deepcrack_conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Create DeepCrack's padded 3x3 convolution."""
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class DeepCrackConvRelu(nn.Module):
    """Convolution and ReLU block from DeepCrack."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the convolution block."""
        super().__init__()
        self.conv = deepcrack_conv3x3(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution and ReLU."""
        return self.activation(self.conv(x))


class DeepCrackDown(nn.Module):
    """DeepCrack encoder block with max-pool indices."""

    def __init__(self, network: nn.Module) -> None:
        """Initialize the encoder block."""
        super().__init__()
        self.network = network
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, torch.Size]:
        """Apply the encoder block."""
        down = self.network(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape


class DeepCrackUp(nn.Module):
    """DeepCrack decoder block with max-unpool."""

    def __init__(self, network: nn.Module) -> None:
        """Initialize the decoder block."""
        super().__init__()
        self.network = network
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs: Tensor, indices: Tensor, output_shape: torch.Size) -> Tensor:
        """Unpool and apply decoder convolutions."""
        return self.network(self.unpool(inputs, indices=indices, output_size=output_shape))


class DeepCrackFuse(nn.Module):
    """DeepCrack side-output fusion block."""

    def __init__(self, network: nn.Module, scale: int) -> None:
        """Initialize the fusion block."""
        super().__init__()
        self.network = network
        self.scale = scale
        self.conv = deepcrack_conv3x3(64, 1)

    def forward(self, down_inp: Tensor, up_inp: Tensor) -> Tensor:
        """Fuse encoder and decoder features."""
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode="bilinear")
        return self.conv(self.network(outputs))


class DeepCrack(nn.Module):
    """Multi-scale crack segmentation model from DeepCrack."""

    def __init__(self) -> None:
        """Initialize DeepCrack."""
        super().__init__()
        self.down1 = DeepCrackDown(
            nn.Sequential(DeepCrackConvRelu(3, 64), DeepCrackConvRelu(64, 64))
        )
        self.down2 = DeepCrackDown(
            nn.Sequential(DeepCrackConvRelu(64, 128), DeepCrackConvRelu(128, 128))
        )
        self.down3 = DeepCrackDown(
            nn.Sequential(
                DeepCrackConvRelu(128, 256),
                DeepCrackConvRelu(256, 256),
                DeepCrackConvRelu(256, 256),
            )
        )
        self.down4 = DeepCrackDown(
            nn.Sequential(
                DeepCrackConvRelu(256, 512),
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 512),
            )
        )
        self.down5 = DeepCrackDown(
            nn.Sequential(
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 512),
            )
        )
        self.up1 = DeepCrackUp(nn.Sequential(DeepCrackConvRelu(64, 64), DeepCrackConvRelu(64, 64)))
        self.up2 = DeepCrackUp(
            nn.Sequential(DeepCrackConvRelu(128, 128), DeepCrackConvRelu(128, 64))
        )
        self.up3 = DeepCrackUp(
            nn.Sequential(
                DeepCrackConvRelu(256, 256),
                DeepCrackConvRelu(256, 256),
                DeepCrackConvRelu(256, 128),
            )
        )
        self.up4 = DeepCrackUp(
            nn.Sequential(
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 256),
            )
        )
        self.up5 = DeepCrackUp(
            nn.Sequential(
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 512),
                DeepCrackConvRelu(512, 512),
            )
        )
        self.fuse5 = DeepCrackFuse(DeepCrackConvRelu(1024, 64), scale=16)
        self.fuse4 = DeepCrackFuse(DeepCrackConvRelu(768, 64), scale=8)
        self.fuse3 = DeepCrackFuse(DeepCrackConvRelu(384, 64), scale=4)
        self.fuse2 = DeepCrackFuse(DeepCrackConvRelu(192, 64), scale=2)
        self.fuse1 = DeepCrackFuse(DeepCrackConvRelu(128, 64), scale=1)
        self.final = deepcrack_conv3x3(5, 1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run DeepCrack segmentation."""
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        up5 = self.up5(out, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        fuse5 = self.fuse5(down5, up5)
        fuse4 = self.fuse4(down4, up4)
        fuse3 = self.fuse3(down3, up3)
        fuse2 = self.fuse2(down2, up2)
        fuse1 = self.fuse1(down1, up1)
        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))
        return output, fuse5, fuse4, fuse3, fuse2, fuse1


class DeepDIVABabyUnet(nn.Module):
    """BabyUnet semantic segmentation model from DeepDIVA."""

    def __init__(self, input_channels: int = 3, output_channels: int = 2) -> None:
        """Initialize BabyUnet."""
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, output_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Run BabyUnet segmentation."""
        x1 = F.relu(self.conv2(F.relu(self.conv1(input_tensor))))
        x2 = self.pool(x1)
        x3 = F.relu(self.conv4(F.relu(self.conv3(x2))))
        x4 = self.pool(x3)
        x5 = F.relu(self.conv6(F.relu(self.conv5(x4))))
        x6 = self.upconv1(x5)
        x7 = F.relu(self.conv8(F.relu(self.conv7(torch.cat([x6, x3], dim=1)))))
        x8 = self.upconv2(x7)
        x9 = F.relu(self.conv10(F.relu(self.conv9(torch.cat([x8, x1], dim=1)))))
        return self.conv11(x9)


def deg_conv(
    batch_norm: bool,
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    """Create deepethogram conv2d + optional BN + LeakyReLU."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, (kernel_size - 1) // 2, bias=bias)
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


def deg_deconv(in_planes: int, out_planes: int, bias: bool = True) -> nn.Sequential:
    """Create deepethogram deconvolution block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=bias),
        nn.LeakyReLU(0.1, inplace=True),
    )


def deg_i_conv(
    batch_norm: bool,
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    """Create deepethogram conv2d + optional BN without activation."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, (kernel_size - 1) // 2, bias=bias)
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    return nn.Sequential(*layers)


def deg_predict_flow(in_planes: int, out_planes: int = 2, bias: bool = False) -> nn.Conv2d:
    """Create deepethogram flow prediction layer."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


class DEGCropConcat(nn.Module):
    """Crop-concat module from deepethogram."""

    def __init__(self, dim: int = 1) -> None:
        """Initialize the crop-concat module."""
        super().__init__()
        self.dim = dim

    def forward(self, tensors: tuple[Tensor, ...]) -> Tensor:
        """Crop tensors to common spatial size and concatenate."""
        height = min(tensor.size(-2) for tensor in tensors)
        width = min(tensor.size(-1) for tensor in tensors)
        return torch.cat(tuple(tensor[..., :height, :width] for tensor in tensors), dim=self.dim)


class TinyMotionNet(nn.Module):
    """TinyMotionNet flow generator from deepethogram."""

    def __init__(
        self,
        num_images: int = 3,
        input_channels: int | None = None,
        batch_norm: bool = True,
        output_channels: int | None = None,
        flow_div: int = 1,
    ) -> None:
        """Initialize TinyMotionNet."""
        super().__init__()
        self.num_images = num_images
        self.input_channels = (
            int(input_channels) if input_channels is not None else self.num_images * 3
        )
        self.output_channels = (
            int(output_channels) if output_channels is not None else int((num_images - 1) * 2)
        )
        self.batch_norm = batch_norm
        self.flow_div = flow_div
        self.conv1 = deg_conv(self.batch_norm, self.input_channels, 64, kernel_size=7)
        self.conv2 = deg_conv(self.batch_norm, 64, 128, stride=2, kernel_size=5)
        self.conv3 = deg_conv(self.batch_norm, 128, 256, stride=2)
        self.conv4 = deg_conv(self.batch_norm, 256, 128, stride=2)
        self.deconv3 = deg_deconv(128, 128)
        self.deconv2 = deg_deconv(128, 64)
        self.xconv3 = deg_i_conv(self.batch_norm, 384 + self.output_channels, 128)
        self.xconv2 = deg_i_conv(self.batch_norm, 192 + self.output_channels, 64)
        self.predict_flow4 = deg_predict_flow(128, out_planes=self.output_channels)
        self.predict_flow3 = deg_predict_flow(128, out_planes=self.output_channels)
        self.predict_flow2 = deg_predict_flow(64, out_planes=self.output_channels)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            self.output_channels, self.output_channels, 4, 2, 1
        )
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            self.output_channels, self.output_channels, 4, 2, 1
        )
        self.concat = DEGCropConcat(dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict multiscale optical flow."""
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        flow4 = self.predict_flow4(out_conv4) * self.flow_div
        flow4_up = self.upsampled_flow4_to_3(flow4) * 2
        out_deconv3 = self.deconv3(out_conv4)
        concat3 = self.concat((out_conv3, out_deconv3, flow4_up))
        out_interconv3 = self.xconv3(concat3)
        flow3 = self.predict_flow3(out_interconv3) * self.flow_div
        flow3_up = self.upsampled_flow3_to_2(flow3) * 2
        out_deconv2 = self.deconv2(out_interconv3)
        concat2 = self.concat((out_conv2, out_deconv2, flow3_up))
        out_interconv2 = self.xconv2(concat2)
        flow2 = self.predict_flow2(out_interconv2) * self.flow_div
        return flow2, flow3, flow4


class DeepFishDoubleConv(nn.Module):
    """Double convolution block from DeepFish U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the DeepFish double convolution block."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply two convolutions."""
        return self.conv(x)


class DeepFishInConv(nn.Module):
    """Input convolution from DeepFish U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the input block."""
        super().__init__()
        self.conv = DeepFishDoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the input block."""
        return self.conv(x)


class DeepFishDown(nn.Module):
    """Down block from DeepFish U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the down block."""
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DeepFishDoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """Apply max pooling and double convolution."""
        return self.mpconv(x)


class DeepFishUp(nn.Module):
    """Up block from DeepFish U-Net."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        """Initialize the up block."""
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
        self.conv = DeepFishDoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Upsample and merge with the skip tensor."""
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        return self.conv(torch.cat([x2, x1], dim=1))


class DeepFishOutConv(nn.Module):
    """Output convolution from DeepFish U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the output convolution."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the output convolution."""
        return self.conv(x)


class DeepFishUNET(nn.Module):
    """UNET from DeepFish."""

    def __init__(self, n_channels: int = 3, n_classes: int = 2) -> None:
        """Initialize DeepFish U-Net."""
        super().__init__()
        self.inc = DeepFishInConv(n_channels, 64)
        self.down1 = DeepFishDown(64, 128)
        self.down2 = DeepFishDown(128, 256)
        self.down3 = DeepFishDown(256, 512)
        self.down4 = DeepFishDown(512, 512)
        self.up1 = DeepFishUp(1024, 256)
        self.up2 = DeepFishUp(512, 128)
        self.up3 = DeepFishUp(256, 64)
        self.up4 = DeepFishUp(128, 64)
        self.outc = DeepFishOutConv(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run DeepFish U-Net segmentation."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


def _deepcad_create_feature_maps(init_channel_number: int, number_of_fmaps: int) -> list[int]:
    """Return the DeepCAD geometric feature-map schedule."""
    return [init_channel_number * 2**index for index in range(number_of_fmaps)]


def _deepcad_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool,
    padding: int = 1,
) -> nn.Conv3d:
    """Create the DeepCAD 3D convolution."""
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def _deepcad_create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    order: str,
    num_groups: int,
    padding: int = 1,
) -> list[tuple[str, nn.Module]]:
    """Create ordered DeepCAD convolution layer pieces."""
    if "c" not in order:
        raise ValueError("Conv layer MUST be present")
    if order[0] in "rle":
        raise ValueError("Non-linearity cannot be the first operation")

    modules: list[tuple[str, nn.Module]] = []
    for index, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            bias = "g" not in order and "b" not in order
            modules.append(
                ("conv", _deepcad_conv3d(in_channels, out_channels, kernel_size, bias, padding))
            )
        elif char == "g":
            if index < order.index("c"):
                raise ValueError("GroupNorm MUST go after the Conv3d")
            groups = min(num_groups, out_channels)
            modules.append(
                ("groupnorm", nn.GroupNorm(num_groups=groups, num_channels=out_channels))
            )
        elif char == "b":
            channels = in_channels if index < order.index("c") else out_channels
            modules.append(("batchnorm", nn.BatchNorm3d(channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'")
    return modules


class DeepCADSingleConv(nn.Sequential):
    """Single DeepCAD Conv3d layer with ordered normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        order: str = "cr",
        num_groups: int = 8,
        padding: int = 1,
    ) -> None:
        """Initialize the single convolution."""
        super().__init__()
        for name, module in _deepcad_create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding=padding
        ):
            self.add_module(name, module)


class DeepCADDoubleConv(nn.Sequential):
    """Two DeepCAD convolution layers used in the encoder and decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: bool,
        kernel_size: int = 3,
        order: str = "cr",
        num_groups: int = 8,
    ) -> None:
        """Initialize the double convolution."""
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = max(out_channels // 2, in_channels)
            conv2_in_channels = conv1_out_channels
            conv2_out_channels = out_channels
        else:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            conv2_in_channels = out_channels
            conv2_out_channels = out_channels
        self.add_module(
            "SingleConv1",
            DeepCADSingleConv(
                conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups
            ),
        )
        self.add_module(
            "SingleConv2",
            DeepCADSingleConv(
                conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups
            ),
        )


class DeepCADEncoder(nn.Module):
    """DeepCAD encoder block with optional 3D pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        apply_pooling: bool = True,
        pool_kernel_size: tuple[int, int, int] = (2, 2, 2),
        pool_type: str = "max",
        conv_layer_order: str = "cr",
        num_groups: int = 8,
    ) -> None:
        """Initialize the encoder block."""
        super().__init__()
        if pool_type not in ["max", "avg"]:
            raise ValueError("pool_type must be 'max' or 'avg'")
        self.pooling: nn.Module | None
        if apply_pooling:
            self.pooling = (
                nn.MaxPool3d(kernel_size=pool_kernel_size)
                if pool_type == "max"
                else nn.AvgPool3d(kernel_size=pool_kernel_size)
            )
        else:
            self.pooling = None
        self.basic_module = DeepCADDoubleConv(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply pooling and convolutions."""
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)


class DeepCADDecoder(nn.Module):
    """DeepCAD decoder block with nearest-neighbor upsampling and concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale_factor: tuple[int, int, int] = (2, 2, 2),
        conv_layer_order: str = "cr",
        num_groups: int = 8,
    ) -> None:
        """Initialize the decoder block."""
        super().__init__()
        del scale_factor
        self.basic_module = DeepCADDoubleConv(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
        )

    def forward(self, encoder_features: Tensor, x: Tensor) -> Tensor:
        """Upsample, concatenate, and convolve decoder features."""
        output_size = encoder_features.size()[2:]
        x = F.interpolate(x, size=output_size, mode="nearest")
        x = torch.cat((encoder_features, x), dim=1)
        return self.basic_module(x)


class DeepCADUNet3D(nn.Module):
    """3D U-Net model from DeepCAD."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool,
        f_maps: int | list[int] = 64,
        layer_order: str = "cr",
        num_groups: int = 8,
    ) -> None:
        """Initialize the DeepCAD U-Net."""
        super().__init__()
        if isinstance(f_maps, int):
            f_maps = _deepcad_create_feature_maps(f_maps, number_of_fmaps=4)

        encoders = []
        for index, out_feature_num in enumerate(f_maps):
            if index == 0:
                encoder = DeepCADEncoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            else:
                encoder = DeepCADEncoder(
                    f_maps[index - 1],
                    out_feature_num,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for index in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[index] + reversed_f_maps[index + 1]
            out_feature_num = reversed_f_maps[index + 1]
            decoders.append(
                DeepCADDecoder(
                    in_feature_num,
                    out_feature_num,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            )
        self.decoders = nn.ModuleList(decoders)
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.final_activation: nn.Module
        self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the DeepCAD encoder-decoder path."""
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        return self.final_conv(x)


def build_deepacson() -> nn.Module:
    """Build a tiny DeepACSON model."""
    model = DeepACSONUNet3D(1, 1, base_channels=4)
    model.eval()
    return model


def example_input_deepacson() -> Tensor:
    """Return a tiny DeepACSON example input."""
    return torch.randn(1, 1, 16, 16, 16)


def build_deepbach() -> nn.Module:
    """Build a tiny DeepBach voice model."""
    model = DeepBachVoiceModel(_DeepBachDataset(), 0, 4, 3, 1, 8, 0.0, 16)
    model.eval()
    return model


def example_input_deepbach() -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    """Return a DeepBach tuple input."""
    left_notes = torch.zeros(1, 4, 3, dtype=torch.long)
    center_notes = torch.zeros(1, 3, dtype=torch.long)
    right_notes = torch.zeros(1, 4, 3, dtype=torch.long)
    left_metas = torch.zeros(1, 3, 2, dtype=torch.long)
    center_metas = torch.zeros(1, 2, dtype=torch.long)
    right_metas = torch.zeros(1, 3, 2, dtype=torch.long)
    return (left_notes, center_notes, right_notes), (left_metas, center_metas, right_metas)


def build_deepcolor() -> nn.Module:
    """Build the PyTorch DeepColor generator."""
    model = SIGGRAPHGenerator()
    model.eval()
    return model


def example_input_deepcolor() -> tuple[Tensor, Tensor, Tensor]:
    """Return a DeepColor tuple input."""
    return torch.randn(1, 16, 16), torch.randn(2, 16, 16), torch.ones(1, 16, 16)


def build_deepcr() -> nn.Module:
    """Build a tiny deepCR U-Net."""
    model = DeepCRUNet2Sigmoid(1, 1, hidden=4)
    model.eval()
    return model


def example_input_deepcr() -> Tensor:
    """Return a deepCR example image."""
    return torch.randn(1, 1, 16, 16)


def build_deepcrack() -> nn.Module:
    """Build DeepCrack."""
    model = DeepCrack()
    model.eval()
    return model


def example_input_deepcrack() -> Tensor:
    """Return a DeepCrack example image."""
    return torch.randn(1, 3, 32, 32)


def build_deepdiva() -> nn.Module:
    """Build DeepDIVA BabyUnet."""
    model = DeepDIVABabyUnet(3, 2)
    model.eval()
    return model


def example_input_deepdiva() -> Tensor:
    """Return a DeepDIVA example image."""
    return torch.randn(1, 3, 32, 32)


def build_deepethogram() -> nn.Module:
    """Build deepethogram TinyMotionNet."""
    model = TinyMotionNet(num_images=3, batch_norm=True)
    model.eval()
    return model


def example_input_deepethogram() -> Tensor:
    """Return a deepethogram stacked-frame example."""
    return torch.randn(1, 9, 32, 32)


def build_deepfish() -> nn.Module:
    """Build DeepFish U-Net."""
    model = DeepFishUNET(3, 2)
    model.eval()
    return model


def example_input_deepfish() -> Tensor:
    """Return a DeepFish example image."""
    return torch.randn(1, 3, 32, 32)


def build_deepcad() -> nn.Module:
    """Build DeepCAD UNet3D."""
    model = DeepCADUNet3D(1, 1, final_sigmoid=True, f_maps=4)
    model.eval()
    return model


def example_input_deepcad() -> Tensor:
    """Return a DeepCAD 3D patch."""
    return torch.randn(1, 1, 16, 16, 16)


MENAGERIE_ENTRIES = [
    ("DeepACSON", "build_deepacson", "example_input_deepacson", 2021, "CV14-DEEPACSON"),
    ("DeepBach", "build_deepbach", "example_input_deepbach", 2017, "CV14-DEEPBACH"),
    ("DeepColor", "build_deepcolor", "example_input_deepcolor", 2017, "CV14-DEEPCOLOR"),
    ("DeepCR", "build_deepcr", "example_input_deepcr", 2019, "CV14-DEEPCR"),
    ("DeepCrack", "build_deepcrack", "example_input_deepcrack", 2019, "CV14-DEEPCRACK"),
    ("DeepDIVA models", "build_deepdiva", "example_input_deepdiva", 2018, "CV14-DEEPDIVA"),
    (
        "DeepEthogram",
        "build_deepethogram",
        "example_input_deepethogram",
        2021,
        "CV14-DEEPETHOGRAM",
    ),
    ("DeepFish", "build_deepfish", "example_input_deepfish", 2020, "CV14-DEEPFISH"),
    ("DeepCAD", "build_deepcad", "example_input_deepcad", 2021, "CV14-DEEPCAD"),
]
