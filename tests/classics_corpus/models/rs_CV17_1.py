# SOURCE: vendored from CMACH508/DeepTH @ main, SSRSGJYD/NeuralTexture @ master, naivoder/DeformableCapsuleNetwork @ main, milutter/deep_lagrangian_networks @ main, kailiang-zhong/DESCN @ main, YoungGod/DFR @ master, xueyunlong12589/DGCNN @ main
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---- DeepTH graph encoder ----


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        if mask is None:
            return input + self.module(input)
        else:
            return input + self.module(input, mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, mask=None):
        h = q
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            # mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            # compatibility = compatibility * mask
            compatibility[mask[None, :, :, :].expand_as(compatibility) == 0] = -1e10
        attn = torch.softmax(compatibility, dim=-1)
        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)
        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )
        self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class PositionWiseFeedforward(nn.Module):
    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim, bias=True),
        )
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.sub_layers(input)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512, normalization="batch"):
        super(MultiHeadAttentionLayer, self).__init__()
        self.self_attention = SkipConnection(
            MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
        )
        self.norm1 = Normalization(embed_dim, normalization)
        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(embed_dim=embed_dim, feed_forward_dim=feed_forward_hidden)
        )
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x, mask):
        x = self.self_attention(x, mask)
        x = self.norm2(self.positionwise_ff(self.norm1(x)))
        return x


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        node_dim=7,
        normalization="batch",
        feed_forward_hidden=512,
    ):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        x = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
        for layer in self.layers:
            x = layer(x, mask)
        return x, x.mean(dim=1)


# ---- DeepTH network ----


class ValueDecoder(nn.Module):
    def __init__(self, dimension):
        super(ValueDecoder, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, 1)
        )

    def forward(self, x):
        return self.value(x)


class DeConvolution(nn.Module):
    def __init__(self, hidden_dim):
        super(DeConvolution, self).__init__()
        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deConv(x)


class PolicyDecoder(nn.Module):
    def __init__(self, dimension, grid):
        super(PolicyDecoder, self).__init__()
        self.dimension = dimension
        self.out = grid
        self.linear = nn.Sequential(nn.Linear(self.dimension, 2 * 2 * self.out), nn.ReLU())
        self.deconv = nn.Sequential(
            DeConvolution(self.out),
            DeConvolution(self.out // 2),
            DeConvolution(self.out // 4),
            DeConvolution(self.out // 8),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out // 16, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, self.out, 2, 2)
        x = self.conv(self.deconv(x))
        x = x.reshape(-1, self.out**2)
        x = self.softmax(x)
        return x


class Reconstruction(nn.Module):
    def __init__(self, dimension, grid):
        super(Reconstruction, self).__init__()
        self.dimension = dimension
        self.out = grid
        self.linear = nn.Sequential(nn.Linear(self.dimension, 2 * 2 * self.out), nn.ReLU())
        self.deconv = nn.Sequential(
            DeConvolution(self.out),
            DeConvolution(self.out // 2),
            DeConvolution(self.out // 4),
            DeConvolution(self.out // 8),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out // 16, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, self.out, 2, 2)
        x = self.conv(self.deconv(x))
        x = x.reshape(-1, 1, self.out, self.out)
        return x


class Projection(nn.Module):
    def __init__(self, grid):
        super(Projection, self).__init__()
        self.grid = grid
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.grid**2, self.grid * 2),
            nn.ReLU(),
            nn.Linear(self.grid * 2, self.grid),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, self.grid**2)
        x = self.linear(x)
        return x


class Network(nn.Module):
    def __init__(self, dimension, grid):
        super(Network, self).__init__()
        self.dimension = dimension
        self.grid = grid
        self.graphEncoder = GraphAttentionEncoder(
            n_heads=8, embed_dim=self.dimension, n_layers=3, node_dim=7
        )
        self.integratedInput = nn.Sequential(
            nn.Linear(self.dimension * 2, self.dimension), nn.ReLU()
        )
        self.valueDecoder = ValueDecoder(self.dimension)
        self.policyDecoder = PolicyDecoder(self.dimension, self.grid)
        self.reconstrucion = Reconstruction(self.dimension, self.grid)
        self.projection = Projection(self.grid)

    def encoder(self, nodes, adjMatrix):
        nodeEmbedding, graphEmbedding = self.graphEncoder(nodes, adjMatrix)
        return nodeEmbedding, graphEmbedding

    def decoder(self, nodeEmbedding, graphEmbedding, macroID):
        macroID = macroID.reshape(-1, 1).repeat(1, self.dimension)[:, None, :]
        currentEmbedding = nodeEmbedding.gather(dim=1, index=macroID).squeeze(1)
        embedding = self.integratedInput(torch.cat((currentEmbedding, graphEmbedding), dim=1))
        value = self.valueDecoder(embedding)
        policy = self.policyDecoder(embedding)
        return value, policy

    def evaluate(self, nodes, adjMatrix, macroID, mask):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        value, policy = self.decoder(nodeEmbedding, graphEmbedding, macroID)
        mask = mask.reshape(-1, self.grid**2)
        # policy_sample = torch.pow(policy, 0.1)
        # policy_sample += 0.000001
        policy[mask == 1] = 0
        policy = policy / policy.sum(dim=-1, keepdim=True)
        policy = Categorical(policy)
        action = policy.sample()
        logits = policy.log_prob(action)
        entropy = policy.entropy()
        proj = self.projection(self.reconstrucion(graphEmbedding))
        return action, logits, entropy, value, proj

    def playTest(self, nodes, adjMatrix, macroID, mask):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        _, policy = self.decoder(nodeEmbedding, graphEmbedding, macroID)
        mask = mask.reshape(-1, self.grid**2)
        policy[mask == 1] = -1
        action = torch.argmax(policy, dim=-1)
        return action

    def project(self, canvas):
        proj = self.projection(canvas)
        return proj.detach()

    def reconProject(self, nodes, adjMatrix):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        proj = self.projection(self.reconstrucion(graphEmbedding))
        return proj

    def forward(self, nodes, adjMatrix, macroID, action):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        value, policy = self.decoder(nodeEmbedding, graphEmbedding, macroID)
        policy = Categorical(policy)
        logits = policy.log_prob(action)
        entropy = policy.entropy()
        proj = self.projection(self.reconstrucion(graphEmbedding))
        return logits, entropy, value, proj


# ---- DeepTTE ----
def deeptte_normalize(x, key, config):
    mean = config[key + "_mean"]
    std = config[key + "_std"]
    return (x - mean) / std


def deeptte_get_local_seq(full_seq, kernel_size, mean, std):
    seq_len = full_seq.size()[1]
    indices = torch.arange(0, seq_len, device=full_seq.device)
    first_seq = torch.index_select(full_seq, dim=1, index=indices[kernel_size - 1 :])
    second_seq = torch.index_select(full_seq, dim=1, index=indices[: -kernel_size + 1])
    local_seq = first_seq - second_seq
    local_seq = (local_seq - mean) / std
    return local_seq


class DeepTTEAttrNet(nn.Module):
    embed_dims = [("driverID", 24000, 16), ("weekID", 7, 3), ("timeID", 1440, 8)]

    def __init__(self):
        super(DeepTTEAttrNet, self).__init__()
        self.build()

    def build(self):
        for name, dim_in, dim_out in DeepTTEAttrNet.embed_dims:
            self.add_module(name + "_em", nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in DeepTTEAttrNet.embed_dims:
            sz += dim_out
        return sz + 1

    def forward(self, attr, config):
        em_list = []
        for name, dim_in, dim_out in DeepTTEAttrNet.embed_dims:
            embed = getattr(self, name + "_em")
            attr_t = attr[name].view(-1, 1)
            attr_t = torch.squeeze(embed(attr_t))
            em_list.append(attr_t)
        dist = deeptte_normalize(attr["dist"], "dist", config)
        em_list.append(dist.view(-1, 1))
        return torch.cat(em_list, dim=1)


class DeepTTEGeoConvNet(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(DeepTTEGeoConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.build()

    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        lngs = torch.unsqueeze(traj["lngs"], dim=2)
        lats = torch.unsqueeze(traj["lats"], dim=2)
        states = self.state_em(traj["states"].long())
        locs = torch.cat((lngs, lats, states), dim=2)
        locs = F.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)
        local_dist = deeptte_get_local_seq(
            traj["dist_gap"], self.kernel_size, config["dist_gap_mean"], config["dist_gap_std"]
        )
        local_dist = torch.unsqueeze(local_dist, dim=2)
        conv_locs = torch.cat((conv_locs, local_dist), dim=2)
        return conv_locs


class DeepTTESpatioTemporalNet(nn.Module):
    def __init__(
        self, attr_size, kernel_size=3, num_filter=32, pooling_method="attention", rnn="lstm"
    ):
        super(DeepTTESpatioTemporalNet, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.geo_conv = DeepTTEGeoConvNet(kernel_size=kernel_size, num_filter=num_filter)
        if rnn == "lstm":
            self.rnn = nn.LSTM(
                input_size=num_filter + 1 + attr_size,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            )
        elif rnn == "rnn":
            self.rnn = nn.RNN(
                input_size=num_filter + 1 + attr_size,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
            )
        if pooling_method == "attention":
            self.attr2atten = nn.Linear(attr_size, 128)

    def out_size(self):
        return 128

    def mean_pooling(self, hiddens, lens):
        hiddens = torch.sum(hiddens, dim=1, keepdim=False)
        lens = torch.as_tensor(lens, dtype=torch.float32, device=hiddens.device)
        lens = torch.unsqueeze(lens, dim=1)
        hiddens = hiddens / lens
        return hiddens

    def attent_pooling(self, hiddens, lens, attr_t):
        attent = F.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)
        return hiddens

    def forward(self, traj, attr_t, config):
        conv_locs = self.geo_conv(traj, config)
        attr_t = torch.unsqueeze(attr_t, dim=1)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1],))
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)
        lens = [x - self.kernel_size + 1 for x in traj["lens"]]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            conv_locs, lens, batch_first=True, enforce_sorted=False
        )
        packed_hiddens, _ = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)
        if self.pooling_method == "mean":
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)
        if self.pooling_method == "attention":
            return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)
        return packed_hiddens, lens, hiddens[:, -1, :]


class DeepTTEEntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super(DeepTTEEntireEstimator, self).__init__()
        self.input2hid = nn.Linear(input_size, hidden_size)
        self.residuals = nn.ModuleList()
        for _ in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))
        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)
        hidden = F.leaky_relu(self.input2hid(inputs))
        for residual_layer in self.residuals:
            residual = F.leaky_relu(residual_layer(hidden))
            hidden = hidden + residual
        out = self.hid2out(hidden)
        return out


class DeepTTENet(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        num_filter=8,
        pooling_method="attention",
        num_final_fcs=1,
        final_fc_size=16,
    ):
        super(DeepTTENet, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.build()
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find(".bias") != -1:
                param.data.fill_(0)
            elif name.find(".weight") != -1 and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def build(self):
        self.attr_net = DeepTTEAttrNet()
        self.spatio_temporal = DeepTTESpatioTemporalNet(
            attr_size=self.attr_net.out_size(),
            kernel_size=self.kernel_size,
            num_filter=self.num_filter,
            pooling_method=self.pooling_method,
        )
        self.entire_estimate = DeepTTEEntireEstimator(
            input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=self.num_final_fcs,
            hidden_size=self.final_fc_size,
        )

    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr, config)
        _, _, sptm_t = self.spatio_temporal(traj, attr_t, config)
        entire_out = self.entire_estimate(attr_t, sptm_t)
        return entire_out


# ---- NeuralTexture texture ----


class SingleLayerTexture(nn.Module):
    def __init__(self, W, H):
        super(SingleLayerTexture, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W, H))

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y = F.grid_sample(self.layer1.repeat(batch, 1, 1, 1), x)
        return y


class LaplacianPyramid(nn.Module):
    def __init__(self, W, H):
        super(LaplacianPyramid, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W, H))
        self.layer2 = nn.Parameter(torch.FloatTensor(1, 1, W // 2, H // 2))
        self.layer3 = nn.Parameter(torch.FloatTensor(1, 1, W // 4, H // 4))
        self.layer4 = nn.Parameter(torch.FloatTensor(1, 1, W // 8, H // 8))

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y1 = F.grid_sample(self.layer1.repeat(batch, 1, 1, 1), x)
        y2 = F.grid_sample(self.layer2.repeat(batch, 1, 1, 1), x)
        y3 = F.grid_sample(self.layer3.repeat(batch, 1, 1, 1), x)
        y4 = F.grid_sample(self.layer4.repeat(batch, 1, 1, 1), x)
        y = y1 + y2 + y3 + y4
        return y


class Texture(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True):
        super(Texture, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.layer1 = nn.ParameterList()
        self.layer2 = nn.ParameterList()
        self.layer3 = nn.ParameterList()
        self.layer4 = nn.ParameterList()
        if self.use_pyramid:
            self.textures = nn.ModuleList([LaplacianPyramid(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)
                self.layer2.append(self.textures[i].layer2)
                self.layer3.append(self.textures[i].layer3)
                self.layer4.append(self.textures[i].layer4)
        else:
            self.textures = nn.ModuleList([SingleLayerTexture(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)

    def forward(self, x):
        y_i = []
        for i in range(self.feature_num):
            y = self.textures[i](x)
            y_i.append(y)
        y = torch.cat(tuple(y_i), dim=1)
        return y


# ---- NeuralTexture U-Net ----


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, final=False):
        super(up, self).__init__()
        self.concat = concat
        self.final = final
        if self.final:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad
                ),
                nn.InstanceNorm2d(out_ch),
                nn.Tanh(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad
                ),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x1, x2):
        if self.concat:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.down1 = down(input_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up(512, 512, output_pad=1, concat=False)
        self.up2 = up(1024, 512)
        self.up3 = up(768, 256)
        self.up4 = up(384, 128)
        self.up5 = up(192, output_channels, final=True)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return x


# ---- NeuralTexture pipeline ----
class PipeLine(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True):
        super(PipeLine, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.unet = UNet(feature_num, 3)

    def _spherical_harmonics_basis(self, extrinsics):
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float, device=extrinsics.device)
        coff_0 = 1 / (2.0 * math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        sh_bands[:, 0] = coff_0
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (
            extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]
        ) * coff_2
        return sh_bands

    def forward(self, *args):
        if self.view_direction:
            uv_map, extrinsics = args
            x = self.texture(uv_map)
            assert x.shape[1] >= 12
            basis = self._spherical_harmonics_basis(extrinsics)
            basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
            x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis
        else:
            uv_map = args[0]
            x = self.texture(uv_map)
        y = self.unet(x)
        return x[:, 0:3, :, :], y


# ---- Deformable Capsule Network ----
def squash(tensor, dim=-1):
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    scale = (norm**2) / (1 + norm**2)
    return scale * tensor / norm


class DeformConvCapsLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_capsule,
        num_atoms,
        stride=1,
        padding=0,
        routings=3,
    ):
        super(DeformConvCapsLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.stride = stride
        self.padding = padding
        self.routings = routings

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_capsule * num_atoms,
            kernel_size,
            stride,
            padding,
        )
        self.offsets = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding
        )
        self.biases = nn.Parameter(torch.zeros(1, out_channels, num_capsule, num_atoms))

    def forward(self, x):
        batch_size = x.size(0)
        offsets = self.offsets(x)
        offsets = offsets.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        x = self.conv(x)
        votes = x.view(batch_size, self.out_channels, self.num_capsule, self.num_atoms, -1)
        votes = votes.permute(0, 1, 4, 2, 3).contiguous()

        logits = torch.zeros(*votes.size()).to(x.device)
        for i in range(self.routings):
            route = F.softmax(logits, dim=3)
            preactivate = torch.sum(route * votes, dim=2) + self.biases
            activation = squash(preactivate)
            act_replicated = activation.unsqueeze(2).expand_as(votes)
            logits += torch.sum(votes * act_replicated, dim=-1, keepdim=True)
        return activation


class SplitCaps(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_capsule,
        num_atoms,
        kernel_size,
        stride=1,
        padding=0,
        routings=3,
    ):
        super(SplitCaps, self).__init__()
        self.num_classes = num_classes
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms

        self.instantiation_caps = DeformConvCapsLayer(
            in_channels,
            num_capsule,
            kernel_size,
            num_capsule,
            num_atoms,
            stride,
            padding,
            routings,
        )
        self.class_presence_caps = DeformConvCapsLayer(
            in_channels,
            num_classes,
            kernel_size,
            num_capsule,
            num_atoms,
            stride,
            padding,
            routings,
        )

    def forward(self, x):
        instantiation = self.instantiation_caps(x)
        class_presence = self.class_presence_caps(x)
        return instantiation, class_presence


class SERouting(nn.Module):
    def __init__(self, reduction_ratio=4):
        super(SERouting, self).__init__()
        self.reduction_ratio = reduction_ratio

    def forward(self, instantiation, class_presence):
        batch_size, num_capsule, _, num_atoms = instantiation.size()
        combined = torch.cat([instantiation, class_presence], dim=-1)
        excitation = F.relu(nn.Linear(num_atoms * 2, num_atoms // self.reduction_ratio)(combined))
        excitation = torch.sigmoid(
            nn.Linear(num_atoms // self.reduction_ratio, num_atoms * 2)(excitation)
        )
        routed_caps = excitation * combined
        return routed_caps


class DeformCapsNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(DeformCapsNet, self).__init__()
        in_channels = image_size[0]
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.split_caps = SplitCaps(
            128,
            num_classes,
            num_capsule=8,
            num_atoms=16,
            kernel_size=3,
            stride=1,
            padding=1,
            routings=3,
        )
        self.se_routing = SERouting()

    def forward(self, x):
        features = self.backbone(x)
        instantiation, class_presence = self.split_caps(features)
        routed_caps = self.se_routing(instantiation, class_presence)
        return routed_caps


# ---- Deep Lagrangian Network ----
class LowTri:
    def __init__(self, m):
        # Calculate lower triangular matrix indices using numpy
        self._m = m
        self._idx = np.tril_indices(self._m)

    def __call__(self, low_tri_values):
        batch_size = low_tri_values.shape[0]
        self._L = torch.zeros(batch_size, self._m, self._m).type_as(low_tri_values)

        # Assign values to matrix:
        self._L[:batch_size, self._idx[0], self._idx[1]] = low_tri_values[:]
        return self._L[:batch_size]


class SoftplusDer(nn.Module):
    def __init__(self, beta=1.0):
        super(SoftplusDer, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20.0, 20.0)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)

        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):
    def __init__(self):
        super(ReLUDer, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def __init__(self):
        super(LinearDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class CosDer(nn.Module):
    def __init__(self):
        super(CosDer, self).__init__()

    def forward(self, x):
        return -torch.sin(x)


class LagrangianLayer(nn.Module):
    def __init__(self, input_size, n_dof, activation="ReLu"):
        super(LagrangianLayer, self).__init__()

        # Create layer weights and biases:
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))

        # Initialize activation function and its derivative:
        if activation == "ReLu":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()

        else:
            raise ValueError(
                "Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}".format(
                    self.activation
                )
            )

    def forward(self, q, der_prev):
        # Apply Affine Transformation:
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


class DeepLagrangianNetwork(nn.Module):
    def __init__(self, n_dof, **kwargs):
        super(DeepLagrangianNetwork, self).__init__()

        # Read optional arguments:
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)

        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.0))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.0e-5)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":
            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain("relu")
                else:
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0:
                    output_gain = torch.nn.init.calculate_gain("linear")
                else:
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":
            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain("relu")
                else:
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0:
                    output_gain = torch.nn.init.calculate_gain("linear")
                else:
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1.0 and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError(
                "Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {0}".format(
                    self._w_init
                )
            )

        # Compute In- / Output Sizes:
        self.n_dof = n_dof
        self.m = int((n_dof**2 + n_dof) / 2)

        # Compute non-zero elements of L:
        l_output_size = int((self.n_dof**2 + self.n_dof) / 2)
        l_lower_size = l_output_size - self.n_dof

        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract(
            [x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size)
        )

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        # create it once and only apply repeat, this may decrease memory allocation
        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        self.low_tri = LowTri(self.n_dof)

        # Create Network:
        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")

        # Create Input Layer:
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(
                LagrangianLayer(self.n_width, self.n_width, activation=non_linearity)
            )
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)

        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)

        # The diagonal must be non-negative. Therefore, the non-linearity is set to ReLu.
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

    def forward(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[6] + out[7]

        return tau_pred, dEdt

    def _dyn_model(self, q, qd, qdd):
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        # Create initial derivative of dq/dq.
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        # Compute shared network between l & g:
        y, der = self.layers[0](q, der)

        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        # Compute the network heads including the corresponding derivative:
        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag, der_l_diag = self.net_ld(y, der)

        # Compute the potential energy and the gravitational force:
        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze()

        # Assemble l and der_l
        l_diag = l_diag
        low_tri_values = torch.cat((l_diag, l_lower), 1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        # Compute H:
        L = self.low_tri(low_tri_values)
        LT = L.transpose(dim0=1, dim1=2)
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # Calculate dH/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # Calculate dH/dq:
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(
            -1, self.n_dof, self.n_dof, self.n_dof
        )
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + torch.matmul(
            L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3)
        )

        # Compute the Coriolis & Centrifugal forces:
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), torch.matmul(Hdq, qd_4d)).view(
            -1, self.n_dof
        )
        c = Hdt_qd - 1.0 / 2.0 * quad_dq

        # Compute the Torque using the inverse model:
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g

        # Compute kinetic energy T
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = (
            1.0
            / 2.0
            * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qd.view(-1, 1, self.n_dof, 1)).view(
                -1
            )
        )

        # Compute dT/dt:
        qd_H_qdd = torch.matmul(
            qd_4d.transpose(dim0=2, dim1=3), H_qdd.view(-1, 1, self.n_dof, 1)
        ).view(-1)
        qd_Hdt_qd = torch.matmul(
            qd_4d.transpose(dim0=2, dim1=3), Hdt_qd.view(-1, 1, self.n_dof, 1)
        ).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        # Compute dV/dt
        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    def inv_dyn(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        return tau_pred

    def for_dyn(self, q, qd, tau):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]

        # Compute Acceleration, e.g., forward model:
        invH = torch.inverse(H)
        qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        E = out[4] + out[5]
        return E

    def energy_dot(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        dEdt = out[6] + out[7]
        return dEdt

    def cuda(self, device=None):
        # Move the Network to the GPU:
        super(DeepLagrangianNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):
        # Move the Network to the CPU:
        super(DeepLagrangianNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self


# ---- DESCN ----
def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y = torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y


def safe_sqrt(x):
    """Numerically safe version of Pytoch sqrt"""
    return torch.sqrt(torch.clip(x, 1e-9, 1e9))


class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device):
        super(ShareNetwork, self).__init__()
        if cfg.BatchNorm1d == "true":
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.cfg.normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, cfg):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits


class BaseModel4MetaLearner(nn.Module):
    def __init__(self, input_dim, base_dim, cfg, device):
        super(BaseModel4MetaLearner, self).__init__()
        self.DNN = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            # nn.ELU(),
            # nn.Dropout(p=cfg.do_rate),
            # nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, 1),
            # nn.ELU()
            # nn.BatchNorm1d(base_dim),
        )
        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        logit = self.DNN(x)
        return logit


class PrpsyNetwork(nn.Module):
    """propensity network"""

    def __init__(self, base_dim, cfg):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""

    def __init__(self, base_dim, cfg):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit


class ESX(nn.Module):
    """ESX"""

    def __init__(
        self,
        prpsy_network: PrpsyNetwork,
        mu1_network: Mu1Network,
        mu0_network: Mu0Network,
        tau_network: TauNetwork,
        shareNetwork: ShareNetwork,
        cfg,
        device,
    ):
        super(ESX, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.tau_network = tau_network.to(device)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, inputs):
        shared_h = self.shareNetwork(inputs)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1  # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0  # Refer to the naming in TARnet/CFR

        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        return (
            p_prpsy_logit,
            p_estr,
            p_escr,
            tau_logit,
            mu1_logit,
            mu0_logit,
            p_prpsy,
            p_mu1,
            p_mu0,
            p_h1,
            p_h0,
            shared_h,
        )


# ---- DFR feature CAE ----
#########################################
#    1 x 1 conv CAE
#########################################
class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAE, self).__init__()

        layers = []
        layers += [
            nn.Conv2d(
                in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0
            )
        ]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [
            nn.Conv2d(
                (in_channels + 2 * latent_dim) // 2,
                2 * latent_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        ]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
        # of the features for rgb
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [
            nn.Conv2d(
                2 * latent_dim,
                (in_channels + 2 * latent_dim) // 2,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        ]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [
            nn.Conv2d(
                (in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0
            )
        ]
        # layers += [nn.ReLU()]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss


# ---- DGCNN layers ----
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class DGCNNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DGCNNLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


# ---- DGCNN utils ----
def normalize_A(A, lmax=2):
    A = F.relu(A)
    N = A.shape[0]
    A = A * (torch.ones(N, N, device=A.device) - torch.eye(N, N, device=A.device))
    A = A + A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N, N, device=A.device) - torch.matmul(torch.matmul(D, A), D)
    Lnorm = (2 * L / lmax) - torch.eye(N, N, device=A.device)
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1], device=L.device))
        elif i == 1:
            support.append(L)
        else:
            temp = (
                torch.matmul(
                    2 * L,
                    support[-1],
                )
                - support[-2]
            )
            support.append(temp)
    return support


# ---- DGCNN model ----
class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=3):
        # in_channels(int): The feature dimension of each electrode.
        # num_electrodes(int): The number of electrodes.
        # k_adj(int): The number of graph convolutional layers.
        # out_channel(int): The feature dimension of  the graph after GCN.
        # num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = DGCNNLinear(num_electrodes * out_channels, num_classes)
        self.A = nn.Parameter(torch.empty(num_electrodes, num_electrodes))
        nn.init.uniform_(self.A, 0.01, 0.5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # data can also be standardized offline
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)
        return result


# ---- DenseFusion refinement network ----
class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx


class DeepLagrangianTraceAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DeepLagrangianNetwork(n_dof=2, n_width=8, n_depth=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd, qdd = torch.chunk(x, 3, dim=1)
        return self.model(q, qd, qdd)


class _DESCNCfg:
    def __init__(self) -> None:
        self.BatchNorm1d = "false"
        self.do_rate = 0.0
        self.normalization = "none"


def build_deepth() -> Network:
    model = Network(dimension=16, grid=32)
    model.eval()
    return model


def example_input_deepth() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = torch.randn(2, 4, 7)
    adj = torch.ones(2, 4, 4, dtype=torch.bool)
    macro_id = torch.tensor([0, 1], dtype=torch.long)
    action = torch.tensor([0, 3], dtype=torch.long)
    return nodes, adj, macro_id, action


def build_deeptte() -> DeepTTENet:
    model = DeepTTENet()
    model.eval()
    return model


def example_input_deeptte() -> tuple[dict[str, torch.Tensor], dict[str, object], dict[str, float]]:
    attr = {
        "driverID": torch.tensor([1, 2], dtype=torch.long),
        "weekID": torch.tensor([1, 2], dtype=torch.long),
        "timeID": torch.tensor([10, 20], dtype=torch.long),
        "dist": torch.tensor([1.0, 2.0]),
    }
    traj = {
        "lngs": torch.randn(2, 5),
        "lats": torch.randn(2, 5),
        "states": torch.zeros(2, 5),
        "dist_gap": torch.arange(10, dtype=torch.float32).view(2, 5),
        "lens": [5, 5],
    }
    config = {
        "dist_mean": 0.0,
        "dist_std": 1.0,
        "dist_gap_mean": 0.0,
        "dist_gap_std": 1.0,
    }
    return attr, traj, config


def build_deferred_neural_rendering() -> PipeLine:
    model = PipeLine(64, 64, 3, use_pyramid=False, view_direction=False)
    for parameter in model.texture.parameters():
        nn.init.normal_(parameter, mean=0.0, std=0.02)
    model.eval()
    return model


def example_input_deferred_neural_rendering() -> torch.Tensor:
    return torch.rand(1, 64, 64, 2)


def build_deformable_capsules() -> DeformCapsNet:
    model = DeformCapsNet(num_classes=8, image_size=(3, 16, 16))
    model.eval()
    return model


def example_input_deformable_capsules() -> torch.Tensor:
    return torch.randn(1, 3, 16, 16)


def build_delan() -> DeepLagrangianTraceAdapter:
    model = DeepLagrangianTraceAdapter()
    model.eval()
    return model


def example_input_delan() -> torch.Tensor:
    return torch.randn(2, 6)


def build_descn() -> ESX:
    cfg = _DESCNCfg()
    device = torch.device("cpu")
    share = ShareNetwork(input_dim=6, share_dim=8, base_dim=4, cfg=cfg, device=device)
    model = ESX(
        PrpsyNetwork(4, cfg),
        Mu1Network(4, cfg),
        Mu0Network(4, cfg),
        TauNetwork(4, cfg),
        share,
        cfg,
        device,
    )
    model.eval()
    return model


def example_input_descn() -> torch.Tensor:
    return torch.randn(2, 6)


def build_densefusion() -> PoseRefineNet:
    model = PoseRefineNet(num_points=8, num_obj=2)
    model.eval()
    return model


def example_input_densefusion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.randn(1, 8, 3)
    embedding = torch.randn(1, 32, 8)
    obj = torch.tensor([0], dtype=torch.long).view(1, 1)
    return points, embedding, obj


def build_dfr() -> FeatCAE:
    model = FeatCAE(in_channels=8, latent_dim=3, is_bn=True)
    model.eval()
    return model


def example_input_dfr() -> torch.Tensor:
    return torch.randn(2, 8, 4, 4)


def build_dgcnn_eeg() -> DGCNN:
    model = DGCNN(in_channels=4, num_electrodes=5, k_adj=2, out_channels=3, num_classes=3)
    model.eval()
    return model


def example_input_dgcnn_eeg() -> torch.Tensor:
    return torch.randn(2, 5, 4)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepTH", "build_deepth", "example_input_deepth", 2023, "CV17"),
    ("DeepTTE", "build_deeptte", "example_input_deeptte", 2018, "CV17"),
    (
        "Deferred Neural Rendering",
        "build_deferred_neural_rendering",
        "example_input_deferred_neural_rendering",
        2019,
        "CV17",
    ),
    (
        "Deformable Capsules",
        "build_deformable_capsules",
        "example_input_deformable_capsules",
        2022,
        "CV17",
    ),
    ("DeLaN (Deep Lagrangian Network)", "build_delan", "example_input_delan", 2020, "CV17"),
    (
        "DESCN (Deep Entire Space Cross Networks)",
        "build_descn",
        "example_input_descn",
        2022,
        "CV17",
    ),
    ("DenseFusion", "build_densefusion", "example_input_densefusion", 2019, "CV17"),
    ("DFR", "build_dfr", "example_input_dfr", 2020, "CV17"),
    ("DGCNN for EEG emotion", "build_dgcnn_eeg", "example_input_dgcnn_eeg", 2018, "CV17"),
]
