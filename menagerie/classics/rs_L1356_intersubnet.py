# SOURCE: vendored from https://github.com/RookieJunChen/Inter-SubNet @ master (ae1af09b)
# (speech_enhance/inter_subnet/model/Inter_SubNet.py +
#  speech_enhance/audio_zen/model/base_model.py (BaseModel: unfold/norm_wrapper/weight_init) +
#  speech_enhance/audio_zen/model/module/sequence_model.py (SIL_Block, stacked_SIL_blocks_SequenceModel) +
#  speech_enhance/audio_zen/model/module/si_module.py (subband_interaction) +
#  speech_enhance/audio_zen/acoustics/feature.py (drop_band) +
#  speech_enhance/audio_zen/constant.py (EPSILON))
#
# Inter-SubNet: Speech Enhancement with Subband Interaction (Chen et al., ICASSP 2023).
# https://github.com/RookieJunChen/Inter-SubNet
#
# The classes below are the REAL Inter-SubNet model code: a subband-only speech
# enhancement network built from stacked "SIL" (Subband Interaction + LSTM) blocks.
# Each SIL block first lets neighboring subbands exchange context via a
# `subband_interaction` module (per-frequency linear + cross-subband mean pooling,
# TAC-style), then runs an LSTM per subband and a GroupNorm. Only minimal, purely
# mechanical changes were made to make the file self-contained: repo-relative imports
# (`audio_zen.*`, `utils.logger`) were collapsed into this single module (with the
# print-based file logger stubbed to plain `print`), and the unused librosa-only STFT
# helpers in `feature.py` were dropped (only `drop_band`, a pure-torch function, is
# used by the traced forward path). No architecture was altered.

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional

MENAGERIE_ZOO = "vendored-pytorch"

EPSILON = torch.finfo(torch.float32).eps


def log(msg, slack=False):
    # Stand-in for the repo's utils.logger.log (file/Slack sink dropped; not
    # needed for a forward pass).
    print(msg)


# ---------------------------------------------------------------------------
# audio_zen/acoustics/feature.py :: drop_band
# ---------------------------------------------------------------------------
def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet /
    Inter-SubNet family.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, (
        f"Batch size = {batch_size}, num_groups = {num_groups}. "
        f"The batch size should be larger than num_groups."
    )

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must have the same number of frequencies for parallel training.
    # Drop remaining high-frequency bins that don't evenly divide num_groups.
    if num_freqs % num_groups != 0:
        input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)

        output.append(selected)

    return torch.cat(output, dim=0)


# ---------------------------------------------------------------------------
# audio_zen/model/module/si_module.py :: subband_interaction
# ---------------------------------------------------------------------------
class subband_interaction(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(subband_interaction, self).__init__()
        """
        Subband Interaction Module
        """

        self.input_linear = nn.Sequential(nn.Linear(input_size, hidden_size), nn.PReLU())
        self.mean_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.PReLU())
        self.output_linear = nn.Sequential(nn.Linear(hidden_size * 2, input_size), nn.PReLU())
        self.norm = nn.GroupNorm(1, input_size)

    def forward(self, input):
        """
        input: [B, F, F_s, T]
        """
        B, G, N, T = input.shape

        # Transform
        group_input = input  # [B, F, F_s, T]
        group_input = group_input.permute(0, 3, 1, 2).contiguous().view(-1, N)  # [B * T * F, F_s]
        group_output = self.input_linear(group_input).view(B, T, G, -1)  # [B, T, F, H]

        # Avg pooling
        group_mean = group_output.mean(2).view(B * T, -1)  # [B * T, H]

        # Concat and transform
        group_output = group_output.view(B * T, G, -1)  # [B * T, F, H]
        group_mean = self.mean_linear(group_mean).unsqueeze(1).expand_as(group_output).contiguous()
        group_output = torch.cat([group_output, group_mean], 2)  # [B * T, F, 2H]
        group_output = self.output_linear(group_output.view(-1, group_output.shape[-1]))
        group_output = (
            group_output.view(B, T, G, -1).permute(0, 2, 3, 1).contiguous()
        )  # [B, F, F_s, T]
        group_output = self.norm(group_output.view(B * G, N, T))  # [B * F, F_s, T]
        output = input + group_output.view(input.shape)  # [B, F, F_s, T]

        return output


# ---------------------------------------------------------------------------
# audio_zen/model/module/sequence_model.py :: SIL_Block, stacked_SIL_blocks_SequenceModel
# ---------------------------------------------------------------------------
class SIL_Block(nn.Module):
    def __init__(
        self, input_size, tac_hidden_size, lstm_hidden_size, bidirectional, sequence_model="GRU"
    ):
        super().__init__()
        self.SubInter = subband_interaction(input_size=input_size, hidden_size=tac_hidden_size)
        self.sequence_model_type = sequence_model
        if self.sequence_model_type == "LSTM":
            self.RNN = nn.LSTM(
                input_size=input_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.sequence_model_type == "GRU":
            self.RNN = nn.GRU(
                input_size=input_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
        self.norm = nn.GroupNorm(1, lstm_hidden_size)

    def forward(self, x):
        """
        Args:
            [B, F, N(H), T]
        Returns:
            [B, F, N(H), T]
        """
        # SubInter processing
        B, G, N, T = x.size()
        x = self.SubInter(x)

        # RNN processing
        self.RNN.flatten_parameters()
        x = x.reshape(B * G, N, T)
        x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        rnn_o, _ = self.RNN(x)
        o = self.norm(rnn_o.permute(0, 2, 1))  # [B, T, H] => [B, H, T]
        _, H, _ = o.size()
        return o.reshape(B, G, H, T)


class stacked_SIL_blocks_SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        norm=None,
        sequence_model="GRU",
        output_activate_function="Tanh",
        middle_tac_hidden_times=0.66,
    ):
        super().__init__()
        self.sequence_model_type = sequence_model
        self.num_layers = num_layers
        if self.sequence_model_type in ("LSTM", "GRU"):
            self.sequence_list = nn.ModuleList()
            first_SIL = SIL_Block(
                input_size=input_size,
                tac_hidden_size=3 * input_size,
                lstm_hidden_size=hidden_size,
                bidirectional=bidirectional,
                sequence_model=self.sequence_model_type,
            )
            self.sequence_list.append(first_SIL)
            for i in range(1, self.num_layers):
                self.sequence_list.append(
                    SIL_Block(
                        input_size=hidden_size,
                        tac_hidden_size=int(middle_tac_hidden_times * hidden_size),
                        lstm_hidden_size=hidden_size,
                        bidirectional=bidirectional,
                        sequence_model=self.sequence_model_type,
                    )
                )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        if self.sequence_model_type in ("LSTM", "GRU"):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, G=F, N, T]
        Returns:
            [B, F, T]
        """
        for SIL_block in self.sequence_list:
            x = SIL_block(x)

        B, G, H, T = x.size()
        x = x.reshape(B * G, H, T)
        x = x.permute(0, 2, 1).contiguous()

        o = self.fc_output_layer(x)
        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o


# ---------------------------------------------------------------------------
# audio_zen/model/base_model.py :: BaseModel (unfold / norm_wrapper / weight_init)
# ---------------------------------------------------------------------------
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbor):
        """
        Along with the frequency dim, split overlapped sub band units from spectrogram.

        Args:
            input: [B, C, F, T]
            num_neighbor:

        Returns:
            [B, N, C, F_s, T]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbor < 1:
            # No change for the input
            return input.permute(0, 2, 1, 3).reshape(
                batch_size, num_freqs, num_channels, 1, num_frames
            )

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        sub_band_unit_size = num_neighbor * 2 + 1

        # Pad to the top and bottom
        output = functional.pad(output, [0, 0, num_neighbor, num_neighbor], mode="reflect")

        output = functional.unfold(output, (sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, (
            f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"
        )

        # Split the dim of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    @staticmethod
    def offline_laplace_norm(input):
        """
        Args:
            input: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        normed = input / (mu + 1e-5)
        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs, num_freqs * num_frames + 1, num_freqs, dtype=input.dtype, device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)
        entry_count = entry_count.expand_as(cumulative_sum)

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        normed = input.reshape(
            batch_size * num_channels, num_freqs, num_frames
        ) - cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)
        normed = (input - mu) / (std + 1e-5)
        return normed

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        else:
            raise NotImplementedError(
                "You must set up a type of Norm. "
                "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc."
            )
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)


# ---------------------------------------------------------------------------
# speech_enhance/inter_subnet/model/Inter_SubNet.py :: Inter_SubNet
# ---------------------------------------------------------------------------
class Inter_SubNet(BaseModel):
    def __init__(
        self,
        num_freqs,
        look_ahead,
        sequence_model,
        sb_num_neighbors,
        sb_output_activate_function,
        sb_model_hidden_size,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        weight_init=True,
        sbinter_middle_hidden_times=0.66,
    ):
        """
        Inter-SubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), (
            f"{self.__class__.__name__} only support GRU and LSTM."
        )

        subband_input_size = sb_num_neighbors * 2 + 1
        self.sb_model = stacked_SIL_blocks_SequenceModel(
            input_size=subband_input_size,
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function,
            middle_tac_hidden_times=sbinter_middle_hidden_times,
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Unfold noisy input, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(
            batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames
        )

        sb_input = self.norm(noisy_mag_unfolded)

        if batch_size > 1:
            sb_input = drop_band(
                sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band
            )  # [B, F_s, F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, F_s, T]

        # [B, F//num_groups, F_s, T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = (
            sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()
        )

        output = sb_mask[:, :, :, self.look_ahead :]
        return output


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
# ---------------------------------------------------------------------------
_NUM_FREQS = 17
_SB_NEIGHBORS = 2
_NUM_FRAMES = 6


def build_intersubnet():
    torch.manual_seed(0)
    model = Inter_SubNet(
        num_freqs=_NUM_FREQS,
        look_ahead=2,
        sequence_model="LSTM",
        sb_num_neighbors=_SB_NEIGHBORS,
        sb_output_activate_function=False,
        sb_model_hidden_size=8,
        weight_init=False,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        sbinter_middle_hidden_times=0.5,
    )
    model.eval()
    return model


def example_input_intersubnet():
    torch.manual_seed(0)
    # [B, 1, F, T]; batch size must be > num_groups_in_drop_band (2) since the
    # forward path always applies drop_band when batch_size > 1.
    return torch.randn(3, 1, _NUM_FREQS, _NUM_FRAMES)


MENAGERIE_ENTRIES = [
    ("Inter-SubNet", "build_intersubnet", "example_input_intersubnet", 2023, MENAGERIE_ZOO),
]
