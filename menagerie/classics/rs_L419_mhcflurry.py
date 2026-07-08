# SOURCE: vendored from openvax/mhcflurry @ 7d156f0e79191b60110079bb9757f83490235120
# https://raw.githubusercontent.com/openvax/mhcflurry/master/mhcflurry/pytorch_layers.py
# https://raw.githubusercontent.com/openvax/mhcflurry/master/mhcflurry/class1_neural_network.py
#
# O'Donnell, Rubinsteyn, Laserson 2020 (Cell Systems) "MHCflurry 2.0", PyTorch port of
# the pan-allele class-I peptide-MHC binding-affinity predictor. `Class1NeuralNetworkModel`
# (`mhcflurry/class1_neural_network.py`) is the real PyTorch `nn.Module`: a peptide branch
# of stacked `LocallyConnected1D` layers (an unshared-weight 1D convolution matching Keras'
# `LocallyConnected1D`, implemented via `unfold`+`einsum`) followed by peptide dense layers,
# an allele branch (a frozen `nn.Embedding` seeded from precomputed allele similarity
# vectors + allele dense layers), a peptide/allele merge (multiply or concatenate), and a
# main dense stack (feedforward or "with-skip-connections" DenseNet-style topology) ending
# in a sigmoid output layer producing normalized (0,1) binding-affinity predictions. Only
# the pan-allele `forward()` entry point (dict input with 'peptide' and 'allele' keys, the
# production inference path used by `Class1AffinityPredictor`) is exercised here; the
# calibration fast-path methods (`forward_peptide_stage`, `forward_from_peptide_stage`,
# `_forward_compact_cartesian`, cartesian-batch helpers) are also copied verbatim for
# fidelity but are not part of the traced call. `LocallyConnected1D`/`get_activation` are
# copied verbatim from the real `pytorch_layers.py`; `KERAS_BATCH_NORM_EPSILON`/
# `KERAS_BATCH_NORM_MOMENTUM` are the real module-level constants from
# `class1_neural_network.py`. No architectural changes were made. This trace uses the
# plain 3D fp32 peptide-vector input path (`peptide_input_is_indices=False`), which is the
# same forward computation as the (N, L) index-input path minus one leading embedding
# lookup (`_peptide_torch_encoding_table`, a data file the real repo loads from disk and
# out of scope for this self-contained module) that expands indices to the identical (N,
# L, V) tensor `forward()` consumes either way.

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    """
    Map activation name string to a PyTorch activation function.

    Parameters
    ----------
    name : str
        Activation name: "tanh", "sigmoid", "relu", "linear", or ""

    Returns
    -------
    callable or None
        Activation function, or None for no activation
    """
    if not name or name == "linear":
        return None
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "relu":
        return F.relu
    else:
        raise ValueError(f"Unknown activation: {name}")


class LocallyConnected1D(nn.Module):
    """
    A locally connected 1D layer (unshared convolution).

    Unlike Conv1D, this layer uses different filter weights at each position
    in the input sequence. This is equivalent to Keras' LocallyConnected1D.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (filters)
    input_length : int
        Length of the input sequence
    kernel_size : int
        Size of the convolution kernel
    activation : str
        Activation function name
    """

    def __init__(self, in_channels, out_channels, input_length, kernel_size, activation="tanh"):
        super(LocallyConnected1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.output_length = input_length - kernel_size + 1

        # Weight shape: (output_length, out_channels, in_channels * kernel_size)
        self.weight = nn.Parameter(
            torch.randn(self.output_length, out_channels, in_channels * kernel_size)
        )
        # Bias shape: (output_length, out_channels)
        self.bias = nn.Parameter(torch.zeros(self.output_length, out_channels))

        self._activation = get_activation(activation)

        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, in_channels)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, output_length, out_channels)
        """
        batch_size = x.size(0)

        # Use unfold to extract patches and match Keras flatten order.
        # x_unfolded shape: (batch, output_length, in_channels, kernel_size)
        x_unfolded = x.unfold(1, self.kernel_size, 1)
        # Keras flattens patches with kernel positions first, then channels.
        x_unfolded = x_unfolded.permute(0, 1, 3, 2)
        # Reshape to (batch, output_length, kernel_size * in_channels)
        x_unfolded = x_unfolded.reshape(
            batch_size, self.output_length, self.kernel_size * self.in_channels
        )

        # Apply locally connected weights via einsum
        # x_unfolded: (batch, output_length, in_channels * kernel_size)
        # weight: (output_length, out_channels, in_channels * kernel_size)
        # result: (batch, output_length, out_channels)
        output = torch.einsum("boi,ofi->bof", x_unfolded, self.weight) + self.bias

        if self._activation is not None:
            output = self._activation(output)

        return output


KERAS_BATCH_NORM_EPSILON = 1e-3
KERAS_BATCH_NORM_MOMENTUM = 0.01


class Class1NeuralNetworkModel(nn.Module):
    """
    PyTorch module for Class1 neural network.
    """

    def __init__(
        self,
        peptide_encoding_shape,
        allele_representations=None,
        locally_connected_layers=None,
        peptide_dense_layer_sizes=None,
        allele_dense_layer_sizes=None,
        layer_sizes=None,
        peptide_allele_merge_method="multiply",
        peptide_allele_merge_activation="",
        activation="tanh",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=False,
        dense_layer_l1_regularization=0.001,
        dense_layer_l2_regularization=0.0,
        topology="feedforward",
        num_outputs=1,
        init="glorot_uniform",
        peptide_input_is_indices=False,
        peptide_input_vector_encoding_name=None,
    ):
        super(Class1NeuralNetworkModel, self).__init__()

        self.peptide_encoding_shape = peptide_encoding_shape
        if peptide_input_vector_encoding_name is None and peptide_input_is_indices:
            peptide_input_vector_encoding_name = "BLOSUM62"
        self.peptide_input_vector_encoding_name = peptide_input_vector_encoding_name
        self.peptide_input_is_indices = peptide_input_vector_encoding_name is not None
        if self.peptide_input_is_indices:
            raise NotImplementedError(
                "index-input path requires the on-disk amino-acid encoding "
                "table (_peptide_torch_encoding_table); this staging module "
                "only exercises the plain fp32 vector-input path"
            )
        self.has_allele = allele_representations is not None
        self.peptide_allele_merge_method = peptide_allele_merge_method
        self.peptide_allele_merge_activation = peptide_allele_merge_activation
        self.dropout_probability = dropout_probability
        self.topology = topology
        self.num_outputs = num_outputs
        self.activation_name = activation
        self.output_activation_name = output_activation

        if locally_connected_layers is None:
            locally_connected_layers = []
        if peptide_dense_layer_sizes is None:
            peptide_dense_layer_sizes = []
        if allele_dense_layer_sizes is None:
            allele_dense_layer_sizes = []
        if layer_sizes is None:
            layer_sizes = [32]

        # Build locally connected layers
        self.lc_layers = nn.ModuleList()
        input_length = peptide_encoding_shape[0]
        in_channels = peptide_encoding_shape[1]

        for i, lc_params in enumerate(locally_connected_layers):
            filters = lc_params.get("filters", 8)
            kernel_size = lc_params.get("kernel_size", 3)
            lc_activation = lc_params.get("activation", "tanh")

            lc_layer = LocallyConnected1D(
                in_channels=in_channels,
                out_channels=filters,
                input_length=input_length,
                kernel_size=kernel_size,
                activation=lc_activation,
            )
            self.lc_layers.append(lc_layer)
            in_channels = filters
            input_length = lc_layer.output_length

        # Flattened size after locally connected layers
        self.flatten_size = input_length * in_channels

        # Peptide dense layers
        self.peptide_dense_layers = nn.ModuleList()
        peptide_layer_input = self.flatten_size
        for i, size in enumerate(peptide_dense_layer_sizes):
            layer = nn.Linear(peptide_layer_input, size)
            self.peptide_dense_layers.append(layer)
            peptide_layer_input = size

        # Batch normalization after peptide processing (early)
        self.batch_norm_early = None
        if batch_normalization:
            self.batch_norm_early = nn.BatchNorm1d(
                peptide_layer_input,
                eps=KERAS_BATCH_NORM_EPSILON,
                momentum=KERAS_BATCH_NORM_MOMENTUM,
            )

        # Allele embedding and processing
        self.allele_embedding = None
        self.allele_dense_layers = nn.ModuleList()
        allele_output_size = 0

        if self.has_allele:
            num_alleles = allele_representations.shape[0]
            embedding_dim = numpy.prod(allele_representations.shape[1:])

            self.allele_embedding = nn.Embedding(
                num_embeddings=num_alleles, embedding_dim=embedding_dim
            )
            # Set embedding weights and freeze
            self.allele_embedding.weight.data = torch.from_numpy(
                allele_representations.reshape(num_alleles, -1).astype(numpy.float32)
            )
            self.allele_embedding.weight.requires_grad = False

            allele_layer_input = embedding_dim
            for i, size in enumerate(allele_dense_layer_sizes):
                layer = nn.Linear(allele_layer_input, size)
                self.allele_dense_layers.append(layer)
                allele_layer_input = size
            allele_output_size = allele_layer_input

        # Compute merged size
        if self.has_allele:
            if peptide_allele_merge_method == "concatenate":
                merged_size = peptide_layer_input + allele_output_size
            elif peptide_allele_merge_method == "multiply":
                # Both must have the same size for multiply
                merged_size = peptide_layer_input
            else:
                raise ValueError(f"Unknown merge method: {peptide_allele_merge_method}")
        else:
            merged_size = peptide_layer_input

        # Merge activation
        self.merge_activation = get_activation(peptide_allele_merge_activation)

        # Main dense layers
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # For DenseNet topology, track input sizes for skip connections
        self.merged_size = merged_size
        current_size = merged_size
        prev_sizes = []  # Track previous layer output sizes for skip connections

        for i, size in enumerate(layer_sizes):
            if topology == "with-skip-connections" and i > 0:
                if i == 1:
                    current_size = merged_size + prev_sizes[-1]
                else:
                    current_size = prev_sizes[-2] + prev_sizes[-1]

            layer = nn.Linear(current_size, size)
            self.dense_layers.append(layer)

            if batch_normalization:
                self.batch_norms.append(
                    nn.BatchNorm1d(
                        size,
                        eps=KERAS_BATCH_NORM_EPSILON,
                        momentum=KERAS_BATCH_NORM_MOMENTUM,
                    )
                )
            else:
                self.batch_norms.append(None)

            if dropout_probability > 0:
                drop_prob = max(0.0, 1.0 - dropout_probability)
                if drop_prob > 0:
                    self.dropouts.append(nn.Dropout(p=drop_prob))
                else:
                    self.dropouts.append(None)
            else:
                self.dropouts.append(None)

            prev_sizes.append(size)
            current_size = size

        # Output layer
        self.output_layer = nn.Linear(current_size, num_outputs)

        # Activation functions
        self.activation = get_activation(activation)
        self.output_activation = get_activation(output_activation)

        # Initialize weights
        self._initialize_weights(init)

    def _initialize_weights(self, init):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init == "glorot_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init == "glorot_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init == "he_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif init == "he_normal":
                    nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs):
        """
        Forward pass.

        Parameters
        ----------
        inputs : dict
            Dictionary with 'peptide' and optionally 'allele' keys

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, num_outputs)
        """
        if self.has_allele != ("allele" in inputs):
            raise ValueError(
                "Class1NeuralNetworkModel input mismatch: "
                f"network has_allele={self.has_allele} but "
                f"received allele key={('allele' in inputs)}"
            )

        peptide = inputs["peptide"]

        if self.peptide_input_is_indices and peptide.dim() == 2:
            peptide = torch.nn.functional.embedding(peptide.long(), self.peptide_embedding_table)

        # Locally connected layers
        x = peptide
        for lc_layer in self.lc_layers:
            x = lc_layer(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Peptide dense layers
        for layer in self.peptide_dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

        # Early batch normalization
        if self.batch_norm_early is not None:
            x = self.batch_norm_early(x)

        # Allele processing and merge
        if self.has_allele:
            allele_idx = inputs["allele"].long()
            if allele_idx.dim() > 1:
                allele_idx = allele_idx.squeeze(-1)
            allele_embed = self.allele_embedding(allele_idx)

            for layer in self.allele_dense_layers:
                allele_embed = layer(allele_embed)
                if self.activation is not None:
                    allele_embed = self.activation(allele_embed)

            allele_embed = allele_embed.reshape(allele_embed.size(0), -1)

            if self.peptide_allele_merge_method == "concatenate":
                x = torch.cat([x, allele_embed], dim=-1)
            elif self.peptide_allele_merge_method == "multiply":
                x = x * allele_embed

            if self.merge_activation is not None:
                x = self.merge_activation(x)

        # Main dense layers (with optional skip connections for DenseNet topology)
        prev_outputs = []
        merged_input = x

        for i, layer in enumerate(self.dense_layers):
            if self.topology == "with-skip-connections" and i > 0:
                if i == 1:
                    x = torch.cat([merged_input, prev_outputs[-1]], dim=-1)
                else:
                    x = torch.cat([prev_outputs[-2], prev_outputs[-1]], dim=-1)

            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)

            prev_outputs.append(x)

        # Output
        output = self.output_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_mhcflurry():
    # peptide_encoding_shape=(15, 21): a length-15 peptide window x 21-dim one-hot
    # amino-acid vector encoding (the real repo's fixed-length peptide encodings run
    # 8-15 residues; 15 is the standard max window). locally_connected_layers +
    # layer_sizes mirror the real "with-skip-connections" pan-allele production
    # architecture shape (shrunk for a fast trace); allele_representations is a small
    # random (num_alleles, allele_repr_dim) matrix standing in for the real
    # precomputed allele similarity vectors loaded from the repo's data files.
    allele_representations = numpy.random.randn(6, 1, 214).astype(numpy.float32)
    return Class1NeuralNetworkModel(
        peptide_encoding_shape=(15, 21),
        allele_representations=allele_representations,
        locally_connected_layers=[
            {"filters": 8, "kernel_size": 3, "activation": "tanh"},
        ],
        peptide_dense_layer_sizes=[],
        # allele_dense_layer_sizes projects the 214-dim allele embedding down to
        # 104 so it matches the peptide branch's flatten_size ((15-3+1)*8=104) --
        # required by the real code's multiply-merge, which needs equal widths.
        allele_dense_layer_sizes=[104],
        layer_sizes=[16, 16, 16],
        peptide_allele_merge_method="multiply",
        activation="tanh",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=False,
        topology="with-skip-connections",
        num_outputs=1,
        peptide_input_is_indices=False,
    )


def example_input_mhcflurry():
    batch = 4
    peptide = torch.randn(batch, 15, 21)
    allele = torch.randint(0, 6, (batch,))
    return ({"peptide": peptide, "allele": allele},)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MHCflurry", "build_mhcflurry", "example_input_mhcflurry", 2020, "vendored"),
]
