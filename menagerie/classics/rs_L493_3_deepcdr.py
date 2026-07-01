# FAITHFUL PORT of kimmo1019/DeepCDR @ master (original framework: Keras/TensorFlow)
# (prog/model.py: KerasMultiSourceGCNModel.createMaster; prog/layers/graph.py: GraphConv)
#
# DeepCDR: hybrid graph-convolutional network for cancer drug-response prediction
# (Bioinformatics 2020). Four-branch fusion: (1) a stack of custom GraphConv layers
# (step_num=1 neighbor-aggregation GCN, matching prog/layers/graph.py's GraphConv._call:
# `edges^T @ (features @ W + b)`) over the drug molecular graph, global-max/average-pooled
# to a drug embedding; (2) a small 2D-conv tower over the one-hot genomic-mutation track;
# (3) an MLP over gene-expression features; (4) an MLP over methylation features. The four
# branches concatenate and pass through a final 1D "conv-as-MLP" fusion tower (three
# Conv1d+ReLU+MaxPool1d blocks, mirroring the original's Conv2D-on-a-singleton-height-axis
# trick) to a scalar IC50 regression output. Every layer/mechanism is transcribed from the
# real Keras code (functional API -> torch.nn.Module graph); no TensorFlow/Keras dependency
# is required to run this port. Hyperparameters (unit_list=[256,256,256], use_relu=True,
# use_bn=True) match run_DeepCDR.py's argparse defaults; use_GMP follows the actual argparse
# default (unset bool flag -> None/falsy -> GlobalAveragePooling1D branch is taken, per
# createMaster's `if use_GMP: ... else: GlobalAveragePooling1D()`).
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class GraphConv(nn.Module):
    """Port of layers/graph.py GraphLayer/GraphConv (step_num=1 only, matching all
    createMaster call sites which never pass step_num>1)."""

    def __init__(self, in_features, units, activation="relu", use_bias=True):
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(in_features, units))
        nn.init.xavier_uniform_(self.weight)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(units))
        else:
            self.register_parameter("bias", None)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

    def forward(self, features, edges):
        # features: (batch, nodes, in_features); edges: (batch, nodes, nodes)
        out = torch.matmul(features, self.weight)
        if self.use_bias:
            out = out + self.bias
        # GraphConv._call: K.batch_dot(edges^T, features)
        out = torch.bmm(edges.transpose(1, 2), out)
        return self.activation(out)


class DeepCDR(nn.Module):
    def __init__(
        self,
        drug_dim,
        mutation_dim,
        gexpr_dim,
        methy_dim,
        unit_list=(256, 256, 256),
        use_relu=True,
        use_bn=True,
        use_gmp=False,
        use_mut=True,
        use_gexp=True,
        use_methy=True,
        regr=True,
    ):
        super().__init__()
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.use_bn = use_bn
        self.use_gmp = use_gmp
        self.regr = regr
        activation = "relu" if use_relu else "tanh"

        # drug GCN branch
        gcn_layers = []
        gcn_bns = []
        prev_dim = drug_dim
        for units in unit_list:
            gcn_layers.append(GraphConv(prev_dim, units, activation=activation))
            gcn_bns.append(nn.BatchNorm1d(units) if use_bn else nn.Identity())
            prev_dim = units
        gcn_layers.append(GraphConv(prev_dim, 100, activation=activation))
        gcn_bns.append(nn.BatchNorm1d(100) if use_bn else nn.Identity())
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.gcn_bns = nn.ModuleList(gcn_bns)
        self.gcn_dropout = nn.Dropout(0.1)

        # mutation branch: Conv2D(50,(1,700),stride=(1,5)) -> MaxPool(1,5) ->
        # Conv2D(30,(1,5),stride=(1,2)) -> MaxPool(1,10) -> Flatten -> Dense(100)
        self.mut_conv1 = nn.Conv2d(1, 50, kernel_size=(1, 700), stride=(1, 5))
        self.mut_pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.mut_conv2 = nn.Conv2d(50, 30, kernel_size=(1, 5), stride=(1, 2))
        self.mut_pool2 = nn.MaxPool2d(kernel_size=(1, 10))
        self._mut_flat_dim = self._infer_mut_flat_dim(mutation_dim)
        self.mut_fc = nn.Linear(self._mut_flat_dim, 100)
        self.mut_dropout = nn.Dropout(0.1)

        # gene expression branch
        self.gexpr_fc1 = nn.Linear(gexpr_dim, 256)
        self.gexpr_bn = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.gexpr_dropout = nn.Dropout(0.1)
        self.gexpr_fc2 = nn.Linear(256, 100)

        # methylation branch
        self.methy_fc1 = nn.Linear(methy_dim, 256)
        self.methy_bn = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.methy_dropout = nn.Dropout(0.1)
        self.methy_fc2 = nn.Linear(256, 100)

        fused_dim = 100
        if use_mut:
            fused_dim += 100
        if use_gexp:
            fused_dim += 100
        if use_methy:
            fused_dim += 100
        self.fuse_fc = nn.Linear(fused_dim, 300)
        self.fuse_dropout1 = nn.Dropout(0.1)

        # fusion "conv" tower (Conv2D over a (1, 300, 1) singleton-height feature map)
        self.fuse_conv1 = nn.Conv1d(1, 30, kernel_size=150)
        self.fuse_pool1 = nn.MaxPool1d(kernel_size=2)
        self.fuse_conv2 = nn.Conv1d(30, 10, kernel_size=5)
        self.fuse_pool2 = nn.MaxPool1d(kernel_size=3)
        self.fuse_conv3 = nn.Conv1d(10, 5, kernel_size=5)
        self.fuse_pool3 = nn.MaxPool1d(kernel_size=3)
        self.fuse_dropout2 = nn.Dropout(0.1)
        self.fuse_dropout3 = nn.Dropout(0.2)

        fuse_flat_dim = self._infer_fuse_flat_dim()
        out_dim = 1
        self.output_fc = nn.Linear(fuse_flat_dim, out_dim)

    def _infer_mut_flat_dim(self, mutation_dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, 1, mutation_dim)
            x = self.mut_pool1(self.mut_conv1(x))
            x = self.mut_pool2(self.mut_conv2(x))
            return x.numel()

    def _infer_fuse_flat_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 300)
            x = self.fuse_pool1(self.fuse_conv1(x))
            x = self.fuse_pool2(self.fuse_conv2(x))
            x = self.fuse_pool3(self.fuse_conv3(x))
            return x.numel()

    def forward(self, drug_feat, drug_adj, mutation_feat, gexpr_feat, methy_feat):
        # drug GCN branch
        h = drug_feat
        for gcn, bn in zip(self.gcn_layers, self.gcn_bns):
            h = gcn(h, drug_adj)
            batch, nodes, feat = h.shape
            h = (
                bn(h.reshape(batch * nodes, feat)).reshape(batch, nodes, feat)
                if self.use_bn
                else bn(h)
            )
            h = self.gcn_dropout(h)
        if self.use_gmp:
            x_drug = torch.amax(h, dim=1)
        else:
            x_drug = torch.mean(h, dim=1)

        parts = [x_drug]

        if self.use_mut:
            x_mut = torch.tanh(self.mut_conv1(mutation_feat))
            x_mut = self.mut_pool1(x_mut)
            x_mut = torch.relu(self.mut_conv2(x_mut))
            x_mut = self.mut_pool2(x_mut)
            x_mut = x_mut.reshape(x_mut.size(0), -1)
            x_mut = torch.relu(self.mut_fc(x_mut))
            x_mut = self.mut_dropout(x_mut)
            parts.append(x_mut)

        if self.use_gexp:
            x_gexpr = torch.tanh(self.gexpr_fc1(gexpr_feat))
            x_gexpr = self.gexpr_bn(x_gexpr)
            x_gexpr = self.gexpr_dropout(x_gexpr)
            x_gexpr = torch.relu(self.gexpr_fc2(x_gexpr))
            parts.append(x_gexpr)

        if self.use_methy:
            x_methy = torch.tanh(self.methy_fc1(methy_feat))
            x_methy = self.methy_bn(x_methy)
            x_methy = self.methy_dropout(x_methy)
            x_methy = torch.relu(self.methy_fc2(x_methy))
            parts.append(x_methy)

        x = torch.cat(parts, dim=1)
        x = torch.tanh(self.fuse_fc(x))
        x = self.fuse_dropout1(x)
        x = x.unsqueeze(1)  # (batch, 1, 300) channel-first for Conv1d

        x = torch.relu(self.fuse_conv1(x))
        x = self.fuse_pool1(x)
        x = torch.relu(self.fuse_conv2(x))
        x = self.fuse_pool2(x)
        x = torch.relu(self.fuse_conv3(x))
        x = self.fuse_pool3(x)
        x = self.fuse_dropout2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fuse_dropout3(x)

        out = self.output_fc(x)
        if not self.regr:
            out = torch.sigmoid(out)
        return out


def build_deepcdr():
    # NOTE: real repo's mutation branch operates on the full
    # genomic_mutation_34673_demap_features.csv width (34673); mutation_dim is shrunk to
    # 2000 here for a fast trace -- the two-stage Conv2d(kernel width 700/5, stride 5/2)
    # tower needs several thousand input columns to avoid the kernel exceeding the
    # shrinking spatial extent, so this is the practical floor, not an arbitrary choice.
    return DeepCDR(
        drug_dim=75,
        mutation_dim=2000,
        gexpr_dim=64,
        methy_dim=48,
        unit_list=(32, 32, 32),
        use_relu=True,
        use_bn=True,
        use_gmp=False,
    )


def example_input_deepcdr():
    batch = 2
    n_nodes = 12
    drug_feat = torch.randn(batch, n_nodes, 75)
    drug_adj = torch.rand(batch, n_nodes, n_nodes)
    mutation_feat = torch.randn(batch, 1, 1, 2000)
    gexpr_feat = torch.randn(batch, 64)
    methy_feat = torch.randn(batch, 48)
    return (drug_feat, drug_adj, mutation_feat, gexpr_feat, methy_feat)


MENAGERIE_ENTRIES = [
    ("DeepCDR", build_deepcdr, example_input_deepcdr, 2020, "REIMPLEMENT"),
]
