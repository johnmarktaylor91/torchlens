# SOURCE: vendored from https://github.com/lrcfmd/Site-Net @ master (modules.py)
#
# `modules.py` below is verbatim upstream code: the crystal-structure attention encoder
# (`SiteNetEncoder`, `SiteNetAttentionBlock`, and their helper layers/norms). It has zero
# dependency beyond torch + torch_scatter.
#
# The `SiteNetWrapper` class further down is NOT part of `modules.py`. It reproduces, at
# the tensor level, the real architecture wiring found in `SiteNet.__init__` /
# `SiteNet.shared_step` in the upstream `lightning_module.py` (learnt elemental embedding
# -> SiteNetEncoder -> linear decoder -> TReLU), but is written as a plain `nn.Module`
# instead of inheriting `pytorch_lightning.LightningModule`. This is necessary because the
# real `SiteNet`/`collate_fn` pull in `pytorch_lightning`, `pymatgen`, `matminer`, and the
# repo's own `h5_handler` module purely for training-time data loading/featurization glue
# that is not installed here and not part of the neural-net architecture. The synthetic
# `example_input_sitenet()` below constructs the same `Batch_Mask`/`Attention_Mask` tensor
# shapes that upstream `collate_fn` produces for a single small crystal (a handful of
# sites, one "image"/supercell repeat), without needing pymatgen/h5_handler at all.

import torch
from torch import nn
import torch.nn.functional as F
from itertools import tee
from torch.nn import LayerNorm, InstanceNorm1d
import numpy as np
from torch_scatter import scatter_std, scatter_mean, segment_coo, segment_csr
from torch_scatter.composite import scatter_std  # noqa: F811 (redundant re-import in real source too, kept for fidelity)

################################
# Helper Functions and classes #
################################
# epsilon value for giving numerical stability to asymptotic function
eps = 0.0000008


# Creates a sequential module of linear layers + activation function + normalization for the interaction and site features, used in the attention block
class set_seq_af_norm(nn.Module):
    def __init__(self, layers, af, norm):
        super().__init__()
        layer_list = layers
        first_iter = layer_list[:-1]
        second_iter = layer_list[1:]
        self.linear_layers = nn.ModuleList(nn.Linear(i, j) for i, j in zip(first_iter, second_iter))
        self.af_modules = nn.ModuleList(af() for i in second_iter)
        self.norm_layers = nn.ModuleList(norm(i) for i in second_iter)

    def forward(self, x):
        for i, j, k in zip(self.linear_layers, self.af_modules, self.norm_layers):
            x = i(x)
            x = j(x)
            x = k(x)
        return x


# Convinience class for defining independant perceptrons with activation functions and norms inside the model, used in the attention block
class pairwise_seq_af_norm(nn.Module):
    def __init__(self, layers, af, norm):
        super().__init__()
        layer_list = layers
        first_iter = layer_list[:-1]
        second_iter = layer_list[1:]
        self.linear_layers = nn.ModuleList(nn.Linear(i, j) for i, j in zip(first_iter, second_iter))
        self.af_modules = nn.ModuleList(af() for i in second_iter)
        self.norm_layers = nn.ModuleList(norm(i) for i in second_iter)

    def forward(self, x):
        for i, j, k in zip(self.linear_layers, self.af_modules, self.norm_layers):
            x = i(x)
            x = j(x)
            x = k(x)
        return x


# Sets all but the highest k attention coefficients to negative infinity prior to softmax
# Just performs softmax along the requested dimension otherwise
# Not used in paper, but made available regardless
def k_softmax(x, dim, k):
    if k != -1:
        top_k = x.topk(k, dim=dim, sorted=False)[1]
        mask = torch.zeros_like(x)
        mask.scatter_(dim, top_k, True)
        mask = mask.bool()
        x[~mask] = float("-infinity")
    x = F.softmax(x, dim=dim)
    return x


# Simple transparent module for use with the 3 part framework above
class FakeModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


# Allows iterating through a list with a length 2 sliding window
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# The mish activation function
class mish(nn.Module):
    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x


class pairwise_norm(nn.Module):
    def __init__(self, dim, norm_func):
        super().__init__()
        self.Norm = norm_func(dim)

    def forward(self, x):
        # Batch mask is B*N,E
        return self.Norm(x)


class set_norm(nn.Module):
    def __init__(self, dim, norm_func):
        super().__init__()
        self.Norm = norm_func(dim)

    def forward(self, x):
        # Batch mask is B*N,E
        return self.Norm(x)


#######################
# Helper Dictionaries #
#######################

# Dictionaries of normalization modules
pairwise_norm_dict = {
    "layer": lambda _: pairwise_norm(_, LayerNorm),
    "none": FakeModule,
}

set_norm_dict = {
    "layer": lambda _: set_norm(_, LayerNorm),
    "none": FakeModule,
}
norm_dict = {
    "layer": LayerNorm,
    "instance": InstanceNorm1d,
    "none": FakeModule,
}
# Dictionary of activation modules
af_dict = {"relu": nn.ReLU, "mish": mish, "none": FakeModule}


class SiteNetAttentionBlock(nn.Module):
    def __init__(
        self,
        site_dim,
        interaction_dim,
        heads=4,
        af="relu",
        set_norm="batch",
        tdot=False,
        k_softmax=-1,
        attention_hidden_layers=[256, 256],
    ):
        super().__init__()
        # Number of attention heads
        self.heads = heads
        # Site feature vector length per attention head
        self.site_dim = site_dim
        # Interaction feature length per attention head
        self.interaction_dim = interaction_dim
        # Activation function to use in hidden layers
        self.af = af_dict[af]()
        # K softmax value, -1 and unused in the paper
        self.k_softmax = k_softmax
        # Hidden layers for calculating the attention weights (g^W)
        self.ije_to_multihead = pairwise_seq_af_norm(
            [2 * site_dim * heads + interaction_dim * heads, *attention_hidden_layers],
            af_dict[af],
            pairwise_norm_dict[set_norm],
        )
        # Final layer to generate the attention weights, no activation function or normalization is used
        self.pre_softmax_linear = nn.Linear(attention_hidden_layers[-1], heads)
        # Maps the bond feautres to the new interaction features (g^I)
        self.ije_to_Interaction_Features = pairwise_seq_af_norm(
            [(site_dim * 2 + interaction_dim) * heads, interaction_dim * heads],
            af_dict[af],
            pairwise_norm_dict[set_norm],
        )
        # Maps the bond features to the attention features (g^A)
        self.ije_to_attention_features = pairwise_seq_af_norm(
            [(site_dim * 2 + interaction_dim) * heads, site_dim],
            af_dict[af],
            pairwise_norm_dict[set_norm],
        )
        # Linear layer on new site features prior to the next attention block / pooling
        self.global_linear = set_seq_af_norm(
            [site_dim * heads, site_dim * heads], af_dict[af], set_norm_dict[set_norm]
        )

    @staticmethod
    def head_reshape(x, heads):
        return x.reshape(*x.shape[:-1], x.shape[-1] // heads, heads)

    def forward(
        self, x, Interaction_Features, Attention_Mask, Batch_Mask, cutoff_mask=None, m=None
    ):
        # Construct the Bond Features x_ije
        x_i = x[Batch_Mask["attention_i"], :]
        x_j = x[Batch_Mask["attention_j"], :]
        x_ije = torch.cat([x_i, x_j, Interaction_Features], axis=2)
        # Construct the Attention Weights
        multi_headed_attention_weights = self.pre_softmax_linear(
            self.ije_to_multihead(x_ije)
        )  # g^W
        multi_headed_attention_weights[Attention_Mask] = float(
            "-infinity"
        )  # Necessary to avoid interfering with the softmax
        if cutoff_mask is not None:
            multi_headed_attention_weights[cutoff_mask] = float("-infinity")
        # Perform softmax on j
        multi_headed_attention_weights = k_softmax(
            multi_headed_attention_weights, 1, self.k_softmax
        )  # K_softmax is unused in the paper, ability to disable message passing beyond the highest N coefficients, dynamic graph
        # Compute the attention weights and perform attention
        x = torch.einsum(
            "ijk,ije->iek",
            multi_headed_attention_weights,
            self.ije_to_attention_features(x_ije),  # g^F
        )
        # Compute the new site features with g^S and append to the global summary
        x = self.global_linear(
            torch.reshape(
                x,
                [
                    x.shape[0],
                    x.shape[1] * x.shape[2],
                ],
            )
        )  # g^S
        m = (
            torch.cat([m, x], dim=1) if m != None else x
        )  # Keep running total of the site features  # noqa: E711 (kept for fidelity)
        # Compute the new interaction features with g^I
        New_interaction_Features = self.ije_to_Interaction_Features(x_ije)  # g^I
        return x, New_interaction_Features, m


class SiteNetEncoder(nn.Module):
    def __init__(
        self,
        embedding_size=100,
        site_feature_size=1,
        attention_blocks=4,
        attention_heads=4,
        site_dim_per_head=64,
        pre_pool_layers=[256, 256],
        post_pool_layers=[256, 256],
        activation_function="relu",
        sym_func="mean",
        set_norm="none",
        lin_norm="none",
        interaction_feature_size=3,
        attention_dim_interaction=64,
        tdot=False,
        attention_hidden_layers=[256, 256],
        k_softmax=-1,
        distance_cutoff=-1,
        **kwargs,
    ):
        super().__init__()
        # Site Layers
        self.full_elem_token_size = embedding_size + site_feature_size + 1
        self.site_featurization = nn.Linear(
            self.full_elem_token_size, site_dim_per_head * attention_heads
        )
        self.site_featurization_norm = set_norm_dict[set_norm](site_dim_per_head * attention_heads)
        # Interaction Layers
        self.interaction_featurization = nn.Linear(
            interaction_feature_size,
            attention_dim_interaction * attention_heads,
        )
        self.interaction_featurization_norm = pairwise_norm_dict[set_norm](
            attention_dim_interaction * attention_heads
        )
        self.distance_cutoff = distance_cutoff

        # Attention Layers
        self.Attention_Blocks = nn.ModuleList(
            SiteNetAttentionBlock(
                site_dim_per_head,
                attention_dim_interaction,
                af=activation_function,
                heads=attention_heads,
                set_norm=set_norm,
                tdot=tdot,
                k_softmax=k_softmax,
                attention_hidden_layers=attention_hidden_layers,
            )
            for i in range(attention_blocks)
        )
        # Pooling Layers
        self.sym_func = sym_func
        self.pre_pool_layers = nn.ModuleList(
            nn.Linear(i, j)
            for i, j in pairwise(
                (
                    site_dim_per_head * attention_blocks * attention_heads,
                    *pre_pool_layers,
                )
            )
        )
        self.pre_pool_layers_norm = nn.ModuleList(
            set_norm_dict[set_norm](i) for i in pre_pool_layers
        )
        if sym_func == "mean" or sym_func == "max":
            self.post_pool_layers = nn.ModuleList(
                nn.Linear(i, j) for i, j in pairwise((pre_pool_layers[-1], *post_pool_layers))
            )
            self.post_pool_layers_norm = nn.ModuleList(
                norm_dict[lin_norm](i) for i in post_pool_layers
            )
        self.af = af_dict[activation_function]()

    def forward(self, Site_Features, Interaction_Features, Attention_Mask, Batch_Mask):
        # Interaction Feature Dimensions (i,j,Batch,Embedding)
        # Site Feature Dimensions (i,Batch,Embedding)
        # Attention Mask Dimensions (Batch,i), True for padding, False for data

        # Compute Optional attention mask for cut off mode
        if self.distance_cutoff >= 0:
            cutoff_mask = torch.gt(Interaction_Features[:, :, 0], self.distance_cutoff)
        else:
            cutoff_mask = None

        # We concat the site feautre outputs as we go to global_summary, intialized here so it exists
        global_summary = None
        # Bring Interaction Features to dimensionality expected by the attention blocks
        Interaction_Features = self.interaction_featurization_norm(
            self.af(self.interaction_featurization(Interaction_Features))
        )
        # Bring Site Features to dimensionality expected by the attention blocks
        Site_Features = self.site_featurization_norm(
            self.af(self.site_featurization(Site_Features))
        )
        # Apply the attention blocks and build up the summary site feature vectors accordingly
        for layer in self.Attention_Blocks:
            Site_Features, Interaction_Features, global_summary = layer(
                Site_Features,
                Interaction_Features,
                Attention_Mask,
                Batch_Mask,
                cutoff_mask=cutoff_mask,
                m=global_summary,
            )
        # Apply the pre pooling layers
        for pre_pool_layer, pre_pool_layer_norm in zip(
            self.pre_pool_layers, self.pre_pool_layers_norm
        ):
            global_summary = pre_pool_layer_norm(self.af(pre_pool_layer(global_summary)))

        # Apply the symettric aggregation function to get the global representation
        # segment_csr takes the mean or max across the whole batch
        if self.sym_func == "mean":
            Global_Representation = segment_csr(global_summary, Batch_Mask["CSR"], reduce="mean")
        elif self.sym_func == "max":
            Global_Representation = segment_csr(global_summary, Batch_Mask["CSR"], reduce="max")
        else:
            raise Exception()
        # Apply the post pooling layers
        for post_pool_layer, post_pool_layer_norm in zip(
            self.post_pool_layers, self.post_pool_layers_norm
        ):
            Global_Representation = post_pool_layer_norm(
                self.af(post_pool_layer(Global_Representation))
            )
        return Global_Representation


class TReLU(torch.autograd.Function):
    """Transparent ReLU: clamps negative outputs to zero but keeps a linear gradient.

    Verbatim (tensor-level) port of `TReLU` in upstream `lightning_module.py`.
    """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SiteNetWrapper(nn.Module):
    """Plain nn.Module reproduction of `SiteNet.__init__` / `SiteNet.shared_step` tensor
    wiring from upstream `lightning_module.py` (learnt elemental embedding -> SiteNetEncoder
    -> linear decoder -> TReLU), using the config from `config/PaperParams_small.yaml`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config["post_pool_layers"][-1], 1)
        self.encoder = SiteNetEncoder(**config)
        self.Elemental_Embeddings = nn.Embedding(
            200,
            config["embedding_size"],
            max_norm=1,
            scale_grad_by_freq=False,
        )

    def input_handler(self, atomic_number, features):
        Atomic_Embedding = self.Elemental_Embeddings(atomic_number)
        return torch.cat([Atomic_Embedding, *features], dim=1)

    def forward(
        self,
        Atomic_ID,
        Site_Feature,
        Oxidation_State,
        Interaction_Features,
        Attention_Mask,
        Batch_Mask,
    ):
        x = self.input_handler(Atomic_ID, [Site_Feature, Oxidation_State])
        Global_Embedding = self.encoder.forward(x, Interaction_Features, Attention_Mask, Batch_Mask)
        prediction = TReLU.apply(self.decoder(Global_Embedding))
        return prediction


MENAGERIE_ZOO = "vendored-pytorch"

_SITENET_CONFIG = dict(
    embedding_size=92,
    site_feature_size=8,
    attention_blocks=2,
    attention_heads=3,
    site_dim_per_head=30,
    pre_pool_layers=[94],
    post_pool_layers=[200, 200, 200],
    activation_function="mish",
    sym_func="mean",
    set_norm="layer",
    lin_norm="layer",
    interaction_feature_size=2,
    attention_dim_interaction=16,
    attention_hidden_layers=[225],
    k_softmax=-1,
    distance_cutoff=-1,
)


def build_sitenet():
    return SiteNetWrapper(_SITENET_CONFIG)


def example_input_sitenet():
    # A single small crystal with 4 sites (a primitive cell repeated once, i.e. one image),
    # matching the tensor shapes produced by the real `collate_fn` in `lightning_module.py`
    # for a one-crystal batch: site features indexed 0..3 attend densely over each other
    # (j-axis == i-axis, no padding needed since there is only one crystal/image in the
    # batch), and `Batch_Mask["CSR"]` marks the single-batch segment boundary for pooling.
    n_sites = 4
    site_feature_size = _SITENET_CONFIG["site_feature_size"]
    interaction_feature_size = _SITENET_CONFIG["interaction_feature_size"]

    Atomic_ID = torch.randint(1, 92, (n_sites,), dtype=torch.long)
    Site_Feature = torch.randn(n_sites, site_feature_size)
    Oxidation_State = torch.randn(n_sites, 1)
    Interaction_Features = torch.randn(n_sites, n_sites, interaction_feature_size)
    Attention_Mask = torch.zeros(n_sites, n_sites, dtype=torch.bool)

    attention_i = torch.arange(n_sites).unsqueeze(1).expand(n_sites, n_sites).long()
    attention_j = torch.arange(n_sites).unsqueeze(0).expand(n_sites, n_sites).long()
    Batch_Mask = {
        "COO": torch.zeros(n_sites, dtype=torch.long),
        "CSR": torch.tensor([0, n_sites], dtype=torch.long),
        "attention_i": attention_i,
        "attention_j": attention_j,
    }

    return (
        Atomic_ID,
        Site_Feature,
        Oxidation_State,
        Interaction_Features,
        Attention_Mask,
        Batch_Mask,
    )


MENAGERIE_ENTRIES = [
    ("Site-Net", "build_sitenet", "example_input_sitenet", 2023, "SOURCE_AVAILABLE"),
]
