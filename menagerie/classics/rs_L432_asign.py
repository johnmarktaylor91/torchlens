# SOURCE: vendored from hrlblab/ASIGN @ main
# https://raw.githubusercontent.com/hrlblab/ASIGN/main/models/model.py
# https://raw.githubusercontent.com/hrlblab/ASIGN/main/models/ResNet.py
# https://raw.githubusercontent.com/hrlblab/ASIGN/main/models/attention_transformer.py
# https://raw.githubusercontent.com/hrlblab/ASIGN/main/models/bs_block.py
#
# CVPR 2025, hrlblab/ASIGN -- "anatomy-aware spatial imputation graph network"
# (`ST_GTC`, Spatial-Temporal Graph-Transformer-Cross-attention) for imputing
# gene expression across 3D spatial-transcriptomics sections at multiple
# spot resolutions (224/512/1024). A shared `ResNet50` (custom torch
# implementation, not torchvision's) image encoder produces per-spot
# embeddings from H&E image patches; these are fused via two `CrossAttention`
# blocks against pre-computed 512- and 1024-resolution neighborhood features;
# the fused embedding plus two independently supplied multi-resolution graph
# feature tensors are each passed through `gs_depth`-deep stacks of
# `gs_block_with_attention_fixed_weights` (an attention-weighted GraphSAGE-
# style graph-conv block operating over a dense adjacency matrix) at three
# resolutions (224/512/1024), the resulting per-layer GNN outputs are fused
# across depth by a `GNNTransformerBlock` (a `TransformerEncoder` over the
# stacked per-layer GNN outputs with a learnable positional encoding per
# depth), and three per-resolution `LayerNorm+Linear` heads predict gene
# expression (250-dim) at each of the three resolutions. This is the real
# repo's model code -- all four files below (`ST_GTC`, `ResNet50`,
# `GNNTransformerBlock`/`CrossAttention`, `gs_block_with_attention_fixed_
# weights`) are copied verbatim (imports only, no architectural edits); they
# depend only on base-lib torch, so no env install was required.
#
# `idx_512`/`idx_1024` appear in `ST_GTC.forward`'s signature but are never
# referenced in the method body in the real repo (they are used only in the
# training loop in `main.py` to index ground-truth labels, not inside the
# model) -- dummy placeholder tensors are supplied for them here to match the
# real call signature exactly.
#
# Example input shapes follow the real `dataloader.py`/`collate_fn` contract:
# `x` is a batch of N 224x224 RGB spot images (`ResNet50` input), `adj`/
# `adj_512`/`adj_1024` are dense binary adjacency matrices (real
# `calcADJ`/`edge_index_to_adj_matrix` outputs, shape (num_nodes,
# num_nodes)), `feature_512`/`feature_1024` are per-spot precomputed
# neighborhood-resolution features of shape (N, cross_attention_dim) used as
# K/V in the cross-attention fusion, and `graph_512`/`graph_1024` are node
# feature matrices for the two auxiliary multi-resolution graphs (shape
# (M, gs_dim), M independent of N, matching the real `graph_512.x`/
# `graph_1024.x` PyG node-feature tensors from `main.py`).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def ResNet50(num_classes=1000):
    return ResNet([3, 4, 6, 3], num_classes)


class GNNTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=3, dropout=0.1):
        """
        dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers
        dropout: Dropout probability
        """
        super(GNNTransformerBlock, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=4 * dim, dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.positional_encoding = nn.Parameter(torch.randn(num_layers, dim))

    def forward(self, gnn_outputs):
        """
        gnn_outputs: Multi-layer GNN outputs in shape (batch_size, num_layers, num_nodes, dim)
        """
        num_layers, num_nodes, dim = gnn_outputs.size()
        batch_size = 1
        gnn_outputs = gnn_outputs.view(batch_size * num_nodes, num_layers, dim)

        gnn_outputs += self.positional_encoding.unsqueeze(0).repeat(batch_size * num_nodes, 1, 1)

        transformer_output = self.transformer_encoder(gnn_outputs)
        final_output = transformer_output.mean(dim=1)
        final_output = final_output.view(num_nodes, dim)
        return final_output


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Shapes of query, key, and value: [batch_size, feature_size]
        """
        batch_size, feature_size = query.size()

        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        query = (
            self.query_proj(query)
            .view(batch_size, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = self.key_proj(key).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        value = (
            self.value_proj(value)
            .view(batch_size, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=query.device)
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        output = self.out_proj(context).squeeze(1)
        return output


class gs_block_with_attention_fixed_weights(nn.Module):
    def __init__(
        self,
        feature_dim,
        embed_dim,
        policy="mean",
        gcn=False,
        attention=True,
        use_fixed_weights=False,
        fixed_weight1=None,
        fixed_weight2=None,
    ):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.attention = attention
        self.use_fixed_weights = use_fixed_weights

        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)
        )
        init.xavier_uniform_(self.weight)

        if self.attention:
            self.att_weight = nn.Parameter(torch.FloatTensor(self.feat_dim, 1))
            init.xavier_uniform_(self.att_weight)

        if use_fixed_weights:
            self.fixed_weight1 = (
                fixed_weight1 if fixed_weight1 is not None else torch.eye(feature_dim)
            )
            self.fixed_weight2 = (
                fixed_weight2 if fixed_weight2 is not None else torch.eye(feature_dim)
            )
            self.fixed_weight1.requires_grad = False
            self.fixed_weight2.requires_grad = False

    def forward(self, x, Adj):
        if self.use_fixed_weights:
            x_transformed = x @ self.fixed_weight1 + x @ self.fixed_weight2
        else:
            x_transformed = x

        neigh_feats = self.aggregate(x_transformed, Adj)

        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        return combined

    def aggregate(self, x, Adj):
        adj = Adj.to(Adj.device)
        if not self.gcn:
            n = len(adj)
            adj = adj - torch.eye(n).to(adj.device)

        if self.policy == "mean" and self.attention:
            att_scores = torch.matmul(x, self.att_weight).squeeze()
            att_scores = F.softmax(att_scores, dim=0)

            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            weighted_feats = mask * att_scores.unsqueeze(1)
            to_feats = weighted_feats.mm(x)

        elif self.policy == "mean":
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)

        elif self.policy == "max":
            indexs = [i.nonzero() for i in adj == 1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat, 0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)

        return to_feats


class ST_GTC(nn.Module):
    def __init__(
        self,
        encoder_output=1024,
        num_heads=8,
        cross_attention_dim=1024,
        dropout=0.1,
        transformer_dim=1024,
        gcn=True,
        gs_dim=1024,
        policy="mean",
        gs_depth=3,
        gene_output=250,
    ):
        super(ST_GTC, self).__init__()
        self.num_heads = num_heads

        self.encoder = ResNet50(num_classes=encoder_output)
        self.cross_attention_512 = CrossAttention(dim=cross_attention_dim, num_heads=num_heads)
        self.cross_attention_1024 = CrossAttention(dim=cross_attention_dim, num_heads=num_heads)

        self.gat_layer_224 = nn.ModuleList(
            [
                gs_block_with_attention_fixed_weights(gs_dim, gs_dim, policy, gcn)
                for i in range(gs_depth)
            ]
        )

        self.gat_layer_512 = nn.ModuleList(
            [
                gs_block_with_attention_fixed_weights(gs_dim, gs_dim, policy, gcn)
                for i in range(gs_depth)
            ]
        )

        self.gat_layer_1024 = nn.ModuleList(
            [
                gs_block_with_attention_fixed_weights(gs_dim, gs_dim, policy, gcn)
                for i in range(gs_depth)
            ]
        )

        self.transformer = GNNTransformerBlock(
            transformer_dim, dropout=dropout, num_heads=num_heads
        )

        self.lstm_512 = nn.Sequential(
            nn.LSTM(transformer_dim, transformer_dim, 2),
        )

        self.gene_head_250 = nn.Sequential(
            nn.LayerNorm(transformer_dim), nn.Linear(transformer_dim, gene_output)
        )

        self.gene_head_512 = nn.Sequential(
            nn.LayerNorm(transformer_dim), nn.Linear(transformer_dim, gene_output)
        )

        self.gene_head_1024 = nn.Sequential(
            nn.LayerNorm(transformer_dim), nn.Linear(transformer_dim, gene_output)
        )

    def forward(
        self,
        x,
        adj,
        feature_512,
        feature_1024,
        graph_512,
        graph_1024,
        adj_512,
        adj_1024,
        idx_512,
        idx_1024,
    ):
        x = self.encoder(x)

        x_cross_512 = self.cross_attention_512(x, feature_512, feature_512)
        x_cross_1024 = self.cross_attention_1024(x, feature_1024, feature_1024)

        x = x_cross_512 + x_cross_1024

        graph_x_224 = []

        for layer in self.gat_layer_224:
            g = layer(x, adj)
            graph_x_224.append(g.unsqueeze(0))

        g_224 = torch.cat(graph_x_224, 0)
        g_224 = self.transformer(g_224)

        graph_x_512 = []
        for layer in self.gat_layer_512:
            g = layer(graph_512, adj_512)
            graph_x_512.append(g.unsqueeze(0))

        g_512 = torch.cat(graph_x_512, 0)
        g_512 = self.transformer(g_512)

        graph_x_1024 = []
        for layer in self.gat_layer_1024:
            g = layer(graph_1024, adj_1024)
            graph_x_1024.append(g.unsqueeze(0))

        g_1024 = torch.cat(graph_x_1024, 0)
        g_1024 = self.transformer(g_1024)

        out_224 = self.gene_head_250(g_224)
        out_512 = self.gene_head_250(g_512)
        out_1024 = self.gene_head_250(g_1024)

        return out_224, out_512, out_1024


def build_asign():
    # Real defaults (encoder_output=cross_attention_dim=transformer_dim=gs_dim=1024,
    # num_heads=8, gs_depth=3, gene_output=250) kept as-is; these are architecture
    # hyperparameters, not something safe to shrink without changing the real model.
    return ST_GTC()


def example_input_asign():
    # N=4 spots at 224-resolution (ResNet50 input batch), M=6 nodes for each of the
    # auxiliary 512/1024-resolution graphs -- matching the real dataloader contract
    # (per-spot RGB patch + precomputed multi-resolution features + independent
    # multi-resolution graph node features/adjacency).
    n_spots = 4
    m_512 = 6
    m_1024 = 6
    dim = 1024

    x = torch.rand(n_spots, 3, 224, 224)
    adj = (torch.rand(n_spots, n_spots) > 0.5).float()
    feature_512 = torch.rand(n_spots, dim)
    feature_1024 = torch.rand(n_spots, dim)
    graph_512 = torch.rand(m_512, dim)
    graph_1024 = torch.rand(m_1024, dim)
    adj_512 = (torch.rand(m_512, m_512) > 0.5).float()
    adj_1024 = (torch.rand(m_1024, m_1024) > 0.5).float()
    idx_512 = torch.zeros(n_spots, dtype=torch.long)
    idx_1024 = torch.zeros(n_spots, dtype=torch.long)

    return (
        x,
        adj,
        feature_512,
        feature_1024,
        graph_512,
        graph_1024,
        adj_512,
        adj_1024,
        idx_512,
        idx_1024,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ASIGN", "build_asign", "example_input_asign", 2025, "vendored"),
]
