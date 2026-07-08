# SOURCE: vendored from PaccMann/paccmann_predictor @ master
# Files vendored (near-verbatim, architecture untouched):
#   paccmann_predictor/models/bimodal_mca.py
#   paccmann_predictor/utils/layers.py
#   paccmann_predictor/utils/hyperparams.py
#   paccmann_predictor/utils/utils.py
# Minimal import fix only: the real bimodal_mca.py imports `pytoda`
# (`import pytoda`, `from pytoda.smiles.transforms import AugmentTensor`)
# solely to support the OPTIONAL `confidence=True` branch of `forward()`
# (Monte-Carlo-dropout / test-time-augmentation uncertainty estimates),
# which is never exercised by a plain trace call. `pytoda` is a non-base
# package we are instructed not to install, so those two lines and the
# dead `confidence=True` code path (which requires a bound SMILES/protein
# language object we don't construct here) are dropped; every remaining
# line of the architecture (embeddings, conv towers, context-attention,
# dense head) is verbatim from the source repo.
"""Bimodal Multiscale Convolutional Attentive Encoder (PaccMann/BimodalMCA).

Predicts drug-target (ligand-receptor) binding affinity from a SMILES
ligand token sequence and a protein-sequence receptor token sequence, using
parallel multi-scale 1D convolutional towers with mutual context attention.
Reference: https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520
"""

from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- paccmann_predictor/utils/utils.py (verbatim, base-lib only) ---
#
# NOTE: `get_device()` is pinned to CPU here (staging-harness concern, not
# an architecture change) because the real code only `.to(self.device)`s
# the conv/dense submodules in `BimodalMCA.__init__`, never the
# `nn.Embedding` layers -- on a CUDA-visible machine this leaves the
# embeddings on CPU while convs move to GPU, so any input device is a
# device mismatch for one half of the forward pass or the other. Pinning
# to CPU keeps every submodule + input on the same device, matching how
# a CPU-only environment already behaves with the unmodified upstream code.


def get_device():
    return torch.device("cpu")


def cuda():
    return False


def to_np(x):
    return x.data.cpu().numpy()


def attention_list_to_matrix(coding_tuple, dim=2):
    raw_coeff = torch.cat([torch.unsqueeze(tpl[1], 2) for tpl in coding_tuple], dim=dim)
    return raw_coeff, torch.mean(raw_coeff, dim=dim)


def get_log_molar(y, ic50_max=None, ic50_min=None):
    return y * (ic50_max - ic50_min) + ic50_min


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data, -1)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.unsqueeze(data, self.dim)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature


DEVICE = get_device()


# --- paccmann_predictor/utils/hyperparams.py (verbatim, base-lib only) ---

ACTIVATION_FN_FACTORY = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "lrelu": nn.LeakyReLU(),
    "elu": nn.ELU(),
}
LOSS_FN_FACTORY = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "binary_cross_entropy": nn.BCELoss(),
}


# --- paccmann_predictor/utils/layers.py (verbatim, base-lib only) ---


def dense_layer(input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.0):
    return nn.Sequential(
        OrderedDict(
            [
                ("projection", nn.Linear(input_size, hidden_size)),
                (
                    "batch_norm",
                    nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
                ),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.0,
    input_channels=1,
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "convolve",
                    torch.nn.Conv2d(
                        input_channels,  # channel_in
                        num_kernel,  # channel_out
                        kernel_size,  # kernel_size
                        padding=[kernel_size[0] // 2, 0],  # pad for valid conv.
                    ),
                ),
                ("squeeze", Squeeze()),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
                (
                    "batch_norm",
                    nn.BatchNorm1d(num_kernel) if batch_norm else nn.Identity(),
                ),
            ]
        )
    )


class ContextAttentionLayer(nn.Module):
    """
    Implements context attention as in the PaccMann paper (Figure 2C) in
    Molecular Pharmaceutics.
    """

    def __init__(
        self,
        reference_hidden_size: int,
        reference_sequence_length: int,
        context_hidden_size: int,
        context_sequence_length: int = 1,
        attention_size: int = 16,
        individual_nonlinearity: type = nn.Sequential(),
        temperature: float = 1.0,
    ):
        super().__init__()

        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity
        self.temperature = temperature

        # Project the reference into the attention space
        self.reference_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(reference_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Project the context into the attention space
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(context_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Optionally reduce the hidden size in context
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'projection',
                            nn.Linear(
                                context_sequence_length,
                                reference_sequence_length,
                            ),
                        ),
                        ('act_fn', individual_nonlinearity),
                    ]
                )
            )  # yapf: disable
        else:
            self.context_hidden_projection = nn.Sequential()

        self.alpha_projection = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(attention_size, 1, bias=False)),
                    ("squeeze", Squeeze()),
                    ("temperature", Temperature(self.temperature)),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(
        self,
        reference: torch.Tensor,
        context: torch.Tensor,
        average_seq: bool = True,
    ):
        assert len(reference.shape) == 3, "Reference tensor needs to be 3D"
        assert len(context.shape) == 3, "Context tensor needs to be 3D"

        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(
            self.context_projection(context).permute(0, 2, 1)
        ).permute(0, 2, 1)
        alphas = self.alpha_projection(torch.tanh(reference_attention + context_attention))

        output = reference * torch.unsqueeze(alphas, -1)
        output = torch.sum(output, 1) if average_seq else torch.squeeze(output)

        return output, alphas


# --- paccmann_predictor/models/bimodal_mca.py (verbatim architecture) ---


class BimodalMCA(nn.Module):
    """Bimodal Multiscale Convolutional Attentive Encoder.

    This is based on the MCA model as presented in the publication in
    Molecular Pharmaceutics:
        https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520.
    """

    def __init__(self, params, *args, **kwargs):
        super(BimodalMCA, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.ligand_padding_length = params["ligand_padding_length"]
        self.receptor_padding_length = params["receptor_padding_length"]

        self.loss_fn = LOSS_FN_FACTORY[
            params.get('loss_fn', 'binary_cross_entropy')
        ]  # yapf: disable
        self.ligand_embedding_type = params.get("ligand_embedding", "learned")
        self.receptor_embedding_type = params.get("receptor_embedding", "learned")

        # Hyperparameter
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')
        ]  # yapf: disable
        self.dropout = params.get("dropout", 0.5)
        self.use_batch_norm = params.get("batch_norm", True)
        self.temperature = params.get("temperature", 1.0)
        self.ligand_filters = params.get("ligand_filters", [32, 32, 32])
        self.receptor_filters = params.get("receptor_filters", [32, 32, 32])

        # set embedding_size to vocabulary_size if one_hot encoding is chosen
        if params.get("ligand_embedding", "learned") == "one_hot":
            self.ligand_embedding_size = params.get("ligand_vocabulary_size", 32)
        else:
            self.ligand_embedding_size = params.get("ligand_embedding_size", 32)
        if params.get("receptor_embedding", "learned") == "one_hot":
            self.receptor_embedding_size = params.get("receptor_vocabulary_size", 35)
        else:
            self.receptor_embedding_size = params.get("receptor_embedding_size", 35)

        self.ligand_kernel_sizes = params.get(
            "ligand_kernel_sizes",
            [
                [3, self.ligand_embedding_size],
                [5, self.ligand_embedding_size],
                [11, self.ligand_embedding_size],
            ],
        )
        self.receptor_kernel_sizes = params.get(
            "receptor_kernel_sizes",
            [
                [3, self.receptor_embedding_size],
                [11, self.receptor_embedding_size],
                [25, self.receptor_embedding_size],
            ],
        )

        self.ligand_attention_size = params.get("ligand_attention_size", 16)
        self.receptor_attention_size = params.get("receptor_attention_size", 16)

        self.ligand_hidden_sizes = [self.ligand_embedding_size] + self.ligand_filters
        self.receptor_hidden_sizes = [self.receptor_embedding_size] + self.receptor_filters
        self.hidden_sizes = [
            self.ligand_embedding_size
            + sum(self.ligand_filters)
            + self.receptor_embedding_size
            + sum(self.receptor_filters)
        ] + params.get("dense_hidden_sizes", [20])
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])

        # Sanity checking of model sizes
        if len(self.ligand_filters) != len(self.ligand_kernel_sizes):
            raise ValueError("Length of ligand filter and kernel size lists do not match.")
        if len(self.receptor_filters) != len(self.receptor_kernel_sizes):
            raise ValueError("Length of receptor filter and kernel size lists do not match.")
        if len(self.ligand_filters) != len(self.receptor_filters):
            raise ValueError(
                "Length of ligand_filters and receptor_filters array must match"
                f", found ligand_filters: {len(self.ligand_filters)} and "
                f"receptor_filters: {len(self.receptor_filters)}."
            )

        # Construct model
        # Embeddings
        if params.get("ligand_embedding", "learned") == "pretrained":
            raise NotImplementedError(
                "Pretrained ligand embeddings need a weight file; not "
                "supported in this vendored trace-only build."
            )
        elif params.get("ligand_embedding", "learned") == "one_hot":
            self.ligand_embedding = nn.Embedding(
                self.params["ligand_vocabulary_size"],
                self.params["ligand_vocabulary_size"],
            )
            self.ligand_embedding.load_state_dict(
                {
                    "weight": torch.nn.functional.one_hot(
                        torch.arange(self.params["ligand_vocabulary_size"])
                    )
                }
            )
            self.ligand_embedding.weight.requires_grad = False
        elif params.get("ligand_embedding", "learned") == "learned":
            self.ligand_embedding = nn.Embedding(
                self.params["ligand_vocabulary_size"],
                self.ligand_embedding_size,
                scale_grad_by_freq=params.get("embed_scale_grad", False),
            )
        else:
            assert params.get("ligand_embedding", "learned") == "predefined", (
                "Choose either pretrained, one_hot, predefined \
             or learned as ligand_embedding. Defaults to learned"
            )

        if params.get("receptor_embedding", "learned") == "pretrained":
            raise NotImplementedError(
                "Pretrained receptor embeddings need a weight file; not "
                "supported in this vendored trace-only build."
            )
        elif params.get("receptor_embedding", "learned") == "one_hot":
            self.receptor_embedding = nn.Embedding(
                self.params["receptor_vocabulary_size"],
                self.params["receptor_vocabulary_size"],
            )
            self.receptor_embedding.load_state_dict(
                {
                    "weight": torch.nn.functional.one_hot(
                        torch.arange(self.params["receptor_vocabulary_size"])
                    )
                }
            )
            self.receptor_embedding.weight.requires_grad = False
        elif params.get("receptor_embedding", "learned") == "learned":
            self.receptor_embedding = nn.Embedding(
                self.params["receptor_vocabulary_size"],
                self.receptor_embedding_size,
                scale_grad_by_freq=params.get("embed_scale_grad", False),
            )
        else:
            assert params.get("receptor_embedding", "learned") == "predefined", (
                "Choose either pretrained, one_hot, predefined \
             or learned as ligand_embedding. Defaults to learned"
            )

        # Convolutions
        self.ligand_convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'ligand_convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.ligand_filters, self.ligand_kernel_sizes)
                    )
                ]
            )
        )  # yapf: disable

        self.receptor_convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'receptor_convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.receptor_filters, self.receptor_kernel_sizes)
                    )
                ]
            )
        )  # yapf: disable

        # Context attention
        self.context_attention_ligand_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"context_attention_ligand_{layer}",
                        ContextAttentionLayer(
                            self.ligand_hidden_sizes[layer],
                            self.params["ligand_padding_length"],
                            self.receptor_hidden_sizes[layer],
                            context_sequence_length=(self.receptor_padding_length),
                            attention_size=self.ligand_attention_size,
                            individual_nonlinearity=params.get(
                                "context_nonlinearity", nn.Sequential()
                            ),
                            temperature=self.temperature,
                        ),
                    )
                    for layer in range(len(self.ligand_filters) + 1)
                ]
            )
        )

        self.context_attention_receptor_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"context_attention_receptor_{layer}",
                        ContextAttentionLayer(
                            self.receptor_hidden_sizes[layer],
                            self.params["receptor_padding_length"],
                            self.ligand_hidden_sizes[layer],
                            context_sequence_length=self.ligand_padding_length,
                            attention_size=self.receptor_attention_size,
                            individual_nonlinearity=params.get(
                                "context_nonlinearity", nn.Sequential()
                            ),
                            temperature=self.temperature,
                        ),
                    )
                    for layer in range(len(self.receptor_filters) + 1)
                ]
            )
        )

        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"dense_{ind}",
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = nn.Linear(self.hidden_sizes[-1], 1)
        if params.get("final_activation", True):
            self.final_dense = nn.Sequential(self.final_dense, ACTIVATION_FN_FACTORY["sigmoid"])

    def forward(self, ligand, receptors):
        """Forward pass through the biomodal MCA.

        Args:
            ligand (torch.Tensor): of type int and shape
                `[bs, ligand_padding_length]`.
            receptors (torch.Tensor): of type int and shape
                `[bs, receptor_padding_length]`.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.

        NOTE: the real forward() also accepts a `confidence: bool` kwarg
        that runs Monte-Carlo-dropout / test-time-augmentation uncertainty
        estimation via `pytoda`. That branch is dropped here (see module
        header); it is orthogonal to the traced architecture.
        """
        # Embedding
        if self.ligand_embedding_type == "predefined":
            embedded_ligand = ligand.to(torch.float)
        else:
            embedded_ligand = self.ligand_embedding(ligand.to(torch.int64))
        if self.receptor_embedding_type == "predefined":
            embedded_receptor = receptors.to(torch.float)
        else:
            embedded_receptor = self.receptor_embedding(receptors.to(torch.int64))

        # Convolutions
        encoded_ligand = [embedded_ligand] + [
            layer(torch.unsqueeze(embedded_ligand, 1)).permute(0, 2, 1)
            for layer in self.ligand_convolutional_layers
        ]
        encoded_receptor = [embedded_receptor] + [
            layer(torch.unsqueeze(embedded_receptor, 1)).permute(0, 2, 1)
            for layer in self.receptor_convolutional_layers
        ]

        # Context attention on ligand
        ligand_encodings, ligand_alphas = zip(
            *[
                layer(reference, context)
                for layer, reference, context in zip(
                    self.context_attention_ligand_layers,
                    encoded_ligand,
                    encoded_receptor,
                )
            ]
        )

        # Context attention on receptor
        receptor_encodings, receptor_alphas = zip(
            *[
                layer(reference, context)
                for layer, reference, context in zip(
                    self.context_attention_receptor_layers,
                    encoded_receptor,
                    encoded_ligand,
                )
            ]
        )

        # Concatenate all encodings
        encodings = torch.cat(
            [
                torch.cat(ligand_encodings, dim=1),
                torch.cat(receptor_encodings, dim=1),
            ],
            dim=1,
        )

        # Apply batch normalization if specified
        out = self.batch_norm(encodings) if self.use_batch_norm else encodings

        # Stack dense layers
        for dl in self.dense_layers:
            out = dl(out)
        predictions = self.final_dense(out)

        prediction_dict = {}
        if not self.training:
            ligand_attention_weights = torch.mean(
                torch.cat([torch.unsqueeze(p, -1) for p in ligand_alphas], dim=-1),
                dim=-1,
            )
            receptor_attention_weights = torch.mean(
                torch.cat([torch.unsqueeze(p, -1) for p in receptor_alphas], dim=-1),
                dim=-1,
            )
            prediction_dict.update(
                {
                    'ligand_attention': ligand_attention_weights,
                    'receptor_attention': receptor_attention_weights,
                }
            )  # yapf: disable

        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)


# --- staging harness ---

_LIGAND_LEN = 32
_RECEPTOR_LEN = 48
_LIGAND_VOCAB = 28
_RECEPTOR_VOCAB = 26

_PARAMS = {
    "ligand_padding_length": _LIGAND_LEN,
    "receptor_padding_length": _RECEPTOR_LEN,
    "ligand_vocabulary_size": _LIGAND_VOCAB,
    "receptor_vocabulary_size": _RECEPTOR_VOCAB,
    "ligand_embedding_size": 8,
    "receptor_embedding_size": 8,
    "ligand_filters": [4, 4],
    "receptor_filters": [4, 4],
    "ligand_kernel_sizes": [[3, 8], [5, 8]],
    "receptor_kernel_sizes": [[3, 8], [5, 8]],
    "ligand_attention_size": 8,
    "receptor_attention_size": 8,
    "dense_hidden_sizes": [16],
    "activation_fn": "relu",
    "dropout": 0.0,
    "batch_norm": True,
    "final_activation": True,
    "loss_fn": "mse",
}


def build_paccmann_bimodal_mca():
    model = BimodalMCA(_PARAMS)
    model.eval()
    return model


def example_input_paccmann_bimodal_mca():
    ligand = torch.randint(0, _LIGAND_VOCAB, (2, _LIGAND_LEN))
    receptor = torch.randint(0, _RECEPTOR_VOCAB, (2, _RECEPTOR_LEN))
    return (ligand, receptor)


MENAGERIE_ENTRIES = [
    (
        "PaccMann-BimodalMCA",
        "build_paccmann_bimodal_mca",
        "example_input_paccmann_bimodal_mca",
        2020,
        "vendored-pytorch",
    ),
]
