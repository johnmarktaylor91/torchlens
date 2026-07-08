# SOURCE: vendored from MLSpeech/DeepFormants @ shallow clone fetched for CV15
# SOURCE: vendored from lucascjysdl/deepfreight @ shallow clone fetched for CV15
# SOURCE: vendored from rajatsen91/deepglo @ shallow clone fetched for CV15
# SOURCE: vendored from scikit-mobility/DeepGravity @ shallow clone fetched for CV15
# SOURCE: vendored from xptree/DeepInf @ shallow clone fetched for CV15
# SOURCE: vendored from HaojieSHI98/DeepKoopmanWithControl @ shallow clone fetched for CV15
# SOURCE: vendored from zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books @ shallow clone fetched for CV15
# SOURCE: vendored from sjmoran/DeepLPF @ shallow clone fetched for CV15

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


class DeepFormantsSpectrogramNet(nn.Module):
    """DeepFormants spectrogram-estimator CNN from `spectrogramEstimate.py`."""

    def __init__(self) -> None:
        """Initialize the vendored spectrogram-estimator layers."""
        super().__init__()
        self.Conv1 = nn.Conv2d(1, 96, kernel_size=(3, 3), stride=1, padding=0)
        self.Conv2 = nn.Conv2d(96, 32, kernel_size=(3, 3), stride=1, padding=0)
        self.Conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=0)
        self.Conv4 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=1, padding=0)
        self.Dense5 = nn.Linear(43 * 38 * 64, 512)
        self.out = nn.Linear(512, 4)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepFormants spectrogram-estimator forward path."""
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.Dense5(x))
        return self.out(x)


class DeepFormantsLpcNet(nn.Module):
    """DeepFormants LPC estimator MLP from `LPC_Estimate_Class.py`."""

    def __init__(self, input_dim: int = 350) -> None:
        """Initialize the vendored LPC-estimator layers."""
        super().__init__()
        self.Dense1 = nn.Linear(input_dim, 1024)
        self.Dense2 = nn.Linear(1024, 512)
        self.Dense3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 4)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepFormants LPC-estimator forward path."""
        x = torch.sigmoid(self.Dense1(x))
        x = torch.sigmoid(self.Dense2(x))
        x = torch.sigmoid(self.Dense3(x))
        return self.out(x)


class DeepFormantsTrackerLSTM(nn.Module):
    """DeepFormants tracker LSTM from `LPC_tracker.py`."""

    def __init__(self, input_dim: int = 350) -> None:
        """Initialize the vendored tracker LSTM layers."""
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, 4)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepFormants tracker forward path."""
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x)


class DeepFreightArgs:
    """Tiny argument container matching the DeepFreight agent/mixer attributes."""

    def __init__(self) -> None:
        """Initialize the fields used by the vendored DeepFreight modules."""
        self.rnn_hidden_dim = 8
        self.n_actions = 4
        self.n_agents = 3
        self.state_shape = (6,)
        self.mixing_embed_dim = 5
        self.hypernet_layers = 1
        self.hypernet_embed = 8


class RNNAgent(nn.Module):
    """DeepFreight RNN agent from `network/actors/rnn_agent.py`."""

    def __init__(self, input_shape: int, args: DeepFreightArgs) -> None:
        """Initialize the vendored DeepFreight recurrent agent."""
        super().__init__()
        self.args = args
        self.name = "rnn"
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self) -> Tensor:
        """Create a zero hidden state matching the module parameter device."""
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs: Tensor, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Run the real DeepFreight RNN-agent forward path."""
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    """DeepFreight QMIX mixer from `network/mixers/qmix.py`."""

    def __init__(self, args: DeepFreightArgs) -> None:
        """Initialize the vendored DeepFreight mixer."""
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(torch.tensor(args.state_shape).prod().item())
        self.embed_dim = args.mixing_embed_dim
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: Tensor, states: Tensor) -> Tensor:
        """Run the real DeepFreight QMIX forward path."""
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        return y.view(bs, -1, 1)


class DeepFreightCombined(nn.Module):
    """Traceable wrapper around the vendored DeepFreight agent and mixer."""

    def __init__(self) -> None:
        """Initialize the agent and mixer with tiny real-compatible settings."""
        super().__init__()
        self.args = DeepFreightArgs()
        self.agent = RNNAgent(5, self.args)
        self.mixer = QMixer(self.args)

    def forward(self, x: Tensor) -> Tensor:
        """Run agent Q-values into the real QMIX forward path."""
        obs = x[:, :5]
        hidden = x[:, 5:13]
        states = x[:, 13:19].reshape(x.size(0), 1, 6)
        q, _ = self.agent(obs, hidden)
        agent_qs = q[:, : self.args.n_agents].reshape(x.size(0), 1, self.args.n_agents)
        return self.mixer(agent_qs, states)


class Chomp1d(nn.Module):
    """DeepGLO temporal-convolution chomp layer from `LocalModel.py`."""

    def __init__(self, chomp_size: int) -> None:
        """Initialize the amount of right-side trimming."""
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        """Trim padded timesteps from the convolution output."""
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """DeepGLO residual temporal block from `LocalModel.py`."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the vendored DeepGLO residual temporal block."""
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride, padding, dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Run the DeepGLO temporal residual block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlockLast(TemporalBlock):
    """DeepGLO final temporal block variant without final ReLUs in the net."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the vendored DeepGLO final temporal block."""
        super().__init__(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.dropout2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the DeepGLO final temporal residual block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    """DeepGLO temporal convolutional network from `LocalModel.py`."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the vendored DeepGLO temporal convolution stack."""
        super().__init__()
        layers: list[nn.Module] = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            block_cls = TemporalBlockLast if i == len(num_channels) - 1 else TemporalBlock
            layers.append(
                block_cls(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepGLO temporal convolutional network."""
        return self.network(x)


class NNOriginalGravity(nn.Module):
    """DeepGravity original gravity neural model from `od_models.py`."""

    def __init__(self, dim_input: int) -> None:
        """Initialize the original gravity linear model."""
        super().__init__()
        self.linear_out = nn.Linear(dim_input, 1)

    def forward(self, vX: Tensor) -> Tensor:
        """Run the original gravity model forward path."""
        return self.linear_out(vX)


class NNMultinomialRegression(NNOriginalGravity):
    """DeepGravity deep multinomial regressor from `deepgravity.py`."""

    def __init__(self, dim_input: int, dim_hidden: int, dropout_p: float = 0.35) -> None:
        """Initialize the vendored DeepGravity MLP layers."""
        super().__init__(dim_input)
        p = dropout_p
        self.linear1 = nn.Linear(dim_input, dim_hidden)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p)
        self.linear3 = nn.Linear(dim_hidden, dim_hidden)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(p)
        self.linear4 = nn.Linear(dim_hidden, dim_hidden)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(p)
        self.linear5 = nn.Linear(dim_hidden, dim_hidden)
        self.relu5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(p)
        self.linear6 = nn.Linear(dim_hidden, dim_hidden // 2)
        self.relu6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout(p)
        self.linear7 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu7 = nn.LeakyReLU()
        self.dropout7 = nn.Dropout(p)
        self.linear8 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu8 = nn.LeakyReLU()
        self.dropout8 = nn.Dropout(p)
        self.linear9 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu9 = nn.LeakyReLU()
        self.dropout9 = nn.Dropout(p)
        self.linear10 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu10 = nn.LeakyReLU()
        self.dropout10 = nn.Dropout(p)
        self.linear11 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu11 = nn.LeakyReLU()
        self.dropout11 = nn.Dropout(p)
        self.linear12 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu12 = nn.LeakyReLU()
        self.dropout12 = nn.Dropout(p)
        self.linear13 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu13 = nn.LeakyReLU()
        self.dropout13 = nn.Dropout(p)
        self.linear14 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu14 = nn.LeakyReLU()
        self.dropout14 = nn.Dropout(p)
        self.linear15 = nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu15 = nn.LeakyReLU()
        self.dropout15 = nn.Dropout(p)
        self.linear_out = nn.Linear(dim_hidden // 2, 1)

    def forward(self, vX: Tensor) -> Tensor:
        """Run the real DeepGravity deep MLP forward path."""
        drop1 = self.dropout1(self.relu1(self.linear1(vX)))
        drop2 = self.dropout2(self.relu2(self.linear2(drop1)))
        drop3 = self.dropout3(self.relu3(self.linear3(drop2)))
        drop4 = self.dropout4(self.relu4(self.linear4(drop3)))
        drop5 = self.dropout5(self.relu5(self.linear5(drop4)))
        drop6 = self.dropout6(self.relu6(self.linear6(drop5)))
        drop7 = self.dropout7(self.relu7(self.linear7(drop6)))
        drop8 = self.dropout8(self.relu8(self.linear8(drop7)))
        drop9 = self.dropout9(self.relu9(self.linear9(drop8)))
        drop10 = self.dropout10(self.relu10(self.linear10(drop9)))
        drop11 = self.dropout11(self.relu11(self.linear11(drop10)))
        drop12 = self.dropout12(self.relu12(self.linear12(drop11)))
        drop13 = self.dropout13(self.relu13(self.linear13(drop12)))
        drop14 = self.dropout14(self.relu14(self.linear14(drop13)))
        drop15 = self.dropout15(self.relu15(self.linear15(drop14)))
        return self.linear_out(drop15)


class BatchGraphConvolution(nn.Module):
    """DeepInf batch graph convolution from `gcn_layers.py`."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize the vendored batch graph convolution."""
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)
        init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, lap: Tensor) -> Tensor:
        """Run the real DeepInf batch graph convolution."""
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        support = torch.bmm(x, expand_weight)
        output = torch.bmm(lap, support)
        if self.bias is not None:
            return output + self.bias
        return output


class BatchGCN(nn.Module):
    """DeepInf batch GCN from `gcn.py`."""

    def __init__(self, n_units: list[int], dropout: float, pretrained_emb: Tensor) -> None:
        """Initialize the vendored DeepInf batch GCN."""
        super().__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = False
        n_units[0] += pretrained_emb.size(1)
        self.layer_stack = nn.ModuleList(
            [BatchGraphConvolution(n_units[i], n_units[i + 1]) for i in range(self.num_layer)]
        )

    def forward(self, x: Tensor, vertices: Tensor, lap: Tensor) -> Tensor:
        """Run the real DeepInf batch-GCN forward path."""
        emb = self.embedding(vertices)
        x = torch.cat((x, emb), dim=2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)


class DeepInfGCNWrapper(nn.Module):
    """Traceable single-input wrapper for the vendored DeepInf GCN."""

    def __init__(self) -> None:
        """Initialize a tiny DeepInf GCN with real layers."""
        super().__init__()
        self.model = BatchGCN([2, 4, 3], 0.1, torch.randn(8, 3))

    def forward(self, x: Tensor) -> Tensor:
        """Build graph tensors from one input and run the real GCN."""
        features = x[:, :, :2]
        vertices = x[:, :, 2].abs().long() % 8
        lap = torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        return self.model(features, vertices, lap)


class KoopmanNetwork(nn.Module):
    """DeepKoopmanWithControl MLP `Network` from `Learn_Knonlinear.py`."""

    def __init__(self, layers: list[int], activation_mode: str = "ReLU") -> None:
        """Initialize the vendored DeepKoCo feed-forward Koopman network."""
        super().__init__()
        ordered_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for layer_i in range(len(layers) - 1):
            ordered_layers[f"linear_{layer_i}"] = nn.Linear(layers[layer_i], layers[layer_i + 1])
            if layer_i != len(layers) - 2:
                if activation_mode.startswith("tanh"):
                    ordered_layers[f"relu_{layer_i}"] = nn.Tanh()
                else:
                    ordered_layers[f"relu_{layer_i}"] = nn.ReLU()
        self.Enet = nn.Sequential(ordered_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepKoCo feed-forward forward path."""
        return self.Enet(x)


class KoopmanRNNNetwork(nn.Module):
    """DeepKoopmanWithControl RNN `Network` from `Learn_Knonlinear_RNN.py`."""

    def __init__(self, input_size: int, output_size: int, hidden_dim: int, n_layers: int) -> None:
        """Initialize the vendored DeepKoCo recurrent network."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def init_hidden(self, batch_size: int, x: Tensor) -> Tensor:
        """Create the initial hidden state on the input device."""
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the real DeepKoCo recurrent forward path."""
        hidden = self.init_hidden(x.size(0), x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden


class DeepLOB(nn.Module):
    """DeepLOB PyTorch notebook model from `jupyter_pytorch/run_train_pytorch.ipynb`."""

    def __init__(self, y_len: int) -> None:
        """Initialize the vendored DeepLOB CNN-Inception-LSTM model."""
        super().__init__()
        self.y_len = y_len
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepLOB forward path with hidden state on `x.device`."""
        h0 = torch.zeros(1, x.size(0), 64, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(1, x.size(0), 64, device=x.device, dtype=x.dtype)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        return torch.softmax(x, dim=1)


class LocalNet(nn.Module):
    """DeepLPF local double-convolution block from `unet.py`."""

    def __init__(self, in_channels: int = 16, out_channels: int = 64) -> None:
        """Initialize the vendored DeepLPF local block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, x_in: Tensor) -> Tensor:
        """Run the real DeepLPF local block forward path."""
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        return self.lrelu(self.conv2(self.refpad(x)))


class UNet(nn.Module):
    """DeepLPF U-Net from `unet.py`."""

    def __init__(self) -> None:
        """Initialize the vendored DeepLPF U-Net."""
        super().__init__()
        self.dconv_down1 = LocalNet(3, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)
        self.maxpool = nn.MaxPool2d(2, padding=0)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_2 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_3 = nn.Conv2d(64, 64, 1)
        self.up_conv1x1_4 = nn.Conv2d(32, 32, 1)
        self.dconv_up4 = LocalNet(256, 128)
        self.dconv_up3 = LocalNet(192, 64)
        self.dconv_up2 = LocalNet(96, 32)
        self.dconv_up1 = LocalNet(48, 16)
        self.conv_last = LocalNet(16, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Run the real DeepLPF U-Net forward path."""
        x_in_tile = x.clone()
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)
        x = self.up_conv1x1_1(self.upsample(x))
        x = self._pad_to_match(x, conv4)
        x = self.dconv_up4(torch.cat([x, conv4], dim=1))
        x = self.up_conv1x1_2(self.upsample(x))
        x = self._pad_to_match(x, conv3)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.up_conv1x1_3(self.upsample(x))
        x = self._pad_to_match(x, conv2)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.up_conv1x1_4(self.upsample(x))
        x = self._pad_to_match(x, conv1)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        out = self.conv_last(x)
        return out + x_in_tile

    def _pad_to_match(self, x: Tensor, skip: Tensor) -> Tensor:
        """Pad `x` with the same branch logic as the vendored DeepLPF U-Net."""
        if x.shape[3] != skip.shape[3] and x.shape[2] != skip.shape[2]:
            return F.pad(x, (1, 0, 0, 1))
        if x.shape[2] != skip.shape[2]:
            return F.pad(x, (0, 0, 0, 1))
        if x.shape[3] != skip.shape[3]:
            return F.pad(x, (1, 0, 0, 0))
        return x


class UNetModel(nn.Module):
    """DeepLPF `UNetModel` from `unet.py`."""

    def __init__(self) -> None:
        """Initialize the vendored DeepLPF top-level model."""
        super().__init__()
        self.unet = UNet()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, img: Tensor) -> Tensor:
        """Run the real DeepLPF top-level model forward path."""
        output_img = self.unet(img)
        return self.final_conv(self.refpad(output_img))


def build_deepformants_spectrogram() -> DeepFormantsSpectrogramNet:
    """Build a traceable DeepFormants spectrogram estimator."""
    return DeepFormantsSpectrogramNet().eval()


def example_input_deepformants_spectrogram() -> Tensor:
    """Return a real-shaped spectrogram tensor for DeepFormants."""
    return torch.randn(1, 1, 55, 50)


def build_deepformants_lpc() -> DeepFormantsLpcNet:
    """Build a traceable DeepFormants LPC estimator."""
    return DeepFormantsLpcNet().eval()


def example_input_deepformants_lpc() -> Tensor:
    """Return a tiny LPC feature tensor for DeepFormants."""
    return torch.randn(2, 350)


def build_deepformants_tracker() -> DeepFormantsTrackerLSTM:
    """Build a traceable DeepFormants tracker LSTM."""
    return DeepFormantsTrackerLSTM().eval()


def example_input_deepformants_tracker() -> Tensor:
    """Return a tiny sequence tensor for the DeepFormants tracker."""
    return torch.randn(2, 3, 350)


def build_deepfreight() -> DeepFreightCombined:
    """Build a traceable DeepFreight RNN-agent plus QMIX stack."""
    return DeepFreightCombined().eval()


def example_input_deepfreight() -> Tensor:
    """Return packed observation, hidden state, and state tensor for DeepFreight."""
    return torch.randn(2, 19)


def build_deepglo() -> TemporalConvNet:
    """Build a traceable DeepGLO temporal convolution network."""
    return TemporalConvNet(1, [4, 4, 1], kernel_size=3, dropout=0.0).eval()


def example_input_deepglo() -> Tensor:
    """Return a tiny time-series tensor for DeepGLO."""
    return torch.randn(2, 1, 16)


def build_deepgravity() -> NNMultinomialRegression:
    """Build a traceable DeepGravity deep multinomial regressor."""
    return NNMultinomialRegression(5, 8, dropout_p=0.0).eval()


def example_input_deepgravity() -> Tensor:
    """Return a tiny origin-destination feature tensor for DeepGravity."""
    return torch.randn(3, 5)


def build_deepinf() -> DeepInfGCNWrapper:
    """Build a traceable DeepInf GCN wrapper."""
    return DeepInfGCNWrapper().eval()


def example_input_deepinf() -> Tensor:
    """Return a packed graph feature tensor for DeepInf."""
    return torch.randn(2, 4, 3)


def build_deepkoco_mlp() -> KoopmanNetwork:
    """Build a traceable DeepKoCo feed-forward Koopman network."""
    return KoopmanNetwork([4, 8, 3]).eval()


def example_input_deepkoco_mlp() -> Tensor:
    """Return a tiny control/state tensor for DeepKoCo MLP."""
    return torch.randn(2, 4)


def build_deepkoco_rnn() -> KoopmanRNNNetwork:
    """Build a traceable DeepKoCo recurrent Koopman network."""
    return KoopmanRNNNetwork(4, 3, 8, 1).eval()


def example_input_deepkoco_rnn() -> Tensor:
    """Return a tiny control/state sequence for DeepKoCo RNN."""
    return torch.randn(2, 5, 4)


def build_deeplob_original() -> DeepLOB:
    """Build a traceable DeepLOB original CNN-LSTM model."""
    return DeepLOB(3).eval()


def example_input_deeplob_original() -> Tensor:
    """Return a real-layout limit-order-book tensor for DeepLOB."""
    return torch.randn(2, 1, 100, 40)


def build_deeplpf() -> UNetModel:
    """Build a traceable DeepLPF U-Net model."""
    return UNetModel().eval()


def example_input_deeplpf() -> Tensor:
    """Return a tiny RGB image tensor for DeepLPF."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "DeepFormants spectrogram estimator",
        "build_deepformants_spectrogram",
        "example_input_deepformants_spectrogram",
        2020,
        "CV15_DEEPFORMANTS_SPEC",
    ),
    (
        "DeepFormants LPC estimator",
        "build_deepformants_lpc",
        "example_input_deepformants_lpc",
        2020,
        "CV15_DEEPFORMANTS_LPC",
    ),
    (
        "DeepFormants tracker LSTM",
        "build_deepformants_tracker",
        "example_input_deepformants_tracker",
        2020,
        "CV15_DEEPFORMANTS_TRACKER",
    ),
    (
        "DeepFreight RNN-QMIX",
        "build_deepfreight",
        "example_input_deepfreight",
        2020,
        "CV15_DEEPFREIGHT",
    ),
    ("DeepGLO local TCN", "build_deepglo", "example_input_deepglo", 2019, "CV15_DEEPGLO"),
    ("DeepGravity MLP", "build_deepgravity", "example_input_deepgravity", 2021, "CV15_DEEPGRAVITY"),
    ("DeepInf GCN", "build_deepinf", "example_input_deepinf", 2018, "CV15_DEEPINF"),
    ("DeepKoCo MLP", "build_deepkoco_mlp", "example_input_deepkoco_mlp", 2022, "CV15_DEEPKOCO_MLP"),
    ("DeepKoCo RNN", "build_deepkoco_rnn", "example_input_deepkoco_rnn", 2022, "CV15_DEEPKOCO_RNN"),
    (
        "DeepLOB original CNN-LSTM",
        "build_deeplob_original",
        "example_input_deeplob_original",
        2018,
        "CV15_DEEPLOB_ORIGINAL",
    ),
    ("DeepLPF U-Net", "build_deeplpf", "example_input_deeplpf", 2020, "CV15_DEEPLPF"),
]
