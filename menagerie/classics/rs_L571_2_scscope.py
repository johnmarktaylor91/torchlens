# FAITHFUL PORT of AltschulerWu-Lab/scScope @ master (original framework: TensorFlow 1.x)
# Original: scscope/scscope/large_scale_processing.py (Inference(), Cal_Loss())
#   + scscope/scscope/ops.py (_variable_with_weight_decay / _variable_on_cpu)
# The original graph-mode TF1 code (tf.placeholder / tf.variable_scope /
# tf.get_variable / tf.Session) cannot run in this base torch env and TF1's
# legacy graph API is not reasonably installable alongside torch here, so the
# architecture is transcribed faithfully into base-env torch below: every
# weight matrix, bias, and control-flow branch in Inference() is preserved.
"""scScope: a deep recurrent autoencoder for large-scale single-cell RNA-seq.

scScope imputes dropout events in scRNA-seq data via a T-step recurrent
autoencoder. Step 0 removes an experimental-batch linear effect and encodes
the (batch-corrected) input through an optional encoder MLP into a latent
code, then decodes to a reconstruction. Steps 1..T-1 feed the previous
reconstruction through a small 2-layer "imputation" network that fills in
only the originally-zero input entries, and re-run the same encode/decode
stack (recurrent weight sharing, matching the original `re_use=True`
variable-scope reuse).
Reference: https://github.com/AltschulerWu-Lab/scScope
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


# --- ported from scscope/scscope/large_scale_processing.py: Inference() ---


class ScScope(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_code_dim,
        exp_batch_dim=1,
        T=2,
        encoder_layers=(),
        decoder_layers=(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_code_dim = latent_code_dim
        self.exp_batch_dim = exp_batch_dim
        self.T = T

        # batch_effect_removal: single linear layer, no bias (matmul only in
        # the original), weights initialized to zero (stddev=0 in the
        # original `_variable_with_weight_decay` call).
        self.batch_effect_weight = nn.Parameter(torch.zeros(exp_batch_dim, input_dim))

        # inference/encoder stack (shared across all T recurrent steps).
        encoder_dims = [input_dim] + list(encoder_layers)
        self.encoder_layers = nn.ModuleList(
            [nn.Linear(encoder_dims[i], encoder_dims[i + 1]) for i in range(len(encoder_layers))]
        )
        latent_input_dim = encoder_dims[-1]

        # latent feature layer.
        self.latent_layer = nn.Linear(latent_input_dim, latent_code_dim)

        # inference/decoder stack (shared across all T recurrent steps).
        decoder_dims = [latent_code_dim] + list(decoder_layers)
        self.decoder_layers = nn.ModuleList(
            [nn.Linear(decoder_dims[i], decoder_dims[i + 1]) for i in range(len(decoder_layers))]
        )
        decoder_last_dim = decoder_dims[-1]

        # reconstruction layer.
        self.reconstruction_layer = nn.Linear(decoder_last_dim, input_dim)

        # recurrent feedback ("impute_layer") network, only created if T > 1
        # (matches the original `if i == 1:` one-time weight creation).
        if T > 1:
            self.feedback_1 = nn.Linear(input_dim, 64)
            self.feedback_2 = nn.Linear(64, input_dim)

        self.relu = nn.ReLU()

    def _encode_decode(self, input_vec):
        h = input_vec
        for layer in self.encoder_layers:
            h = self.relu(layer(h))
        latent_code = self.relu(self.latent_layer(h))

        d = latent_code
        for layer in self.decoder_layers:
            d = self.relu(layer(d))
        output = self.relu(self.reconstruction_layer(d))
        return latent_code, output

    def forward(self, input_d, exp_batch_idx):
        batch_effect_removal_layer = exp_batch_idx @ self.batch_effect_weight

        latent_code_list = []
        output_list = []
        output = None
        for i in range(self.T):
            if i == 0:
                input_vec = self.relu(input_d - batch_effect_removal_layer)
            else:
                intermediate = self.relu(self.feedback_1(output))
                imputation_layer = (1 - torch.sign(input_d)) * self.feedback_2(intermediate)
                input_vec = self.relu(imputation_layer + input_d - batch_effect_removal_layer)

            latent_code, output = self._encode_decode(input_vec)
            latent_code_list.append(latent_code)
            output_list.append(output)

        return output_list, latent_code_list, batch_effect_removal_layer


# --- staging harness ---

_INPUT_DIM = 64
_LATENT_CODE_DIM = 8
_EXP_BATCH_DIM = 1


def build_scscope():
    model = ScScope(
        input_dim=_INPUT_DIM,
        latent_code_dim=_LATENT_CODE_DIM,
        exp_batch_dim=_EXP_BATCH_DIM,
        T=2,
        encoder_layers=(32,),
        decoder_layers=(32,),
    )
    model.eval()
    return model


def example_input_scscope():
    # (gene expression matrix, one-hot/zero experimental-batch indicator),
    # matching the real Inference(input_d, ..., exp_batch_idx, ...) call.
    return torch.rand(4, _INPUT_DIM), torch.zeros(4, _EXP_BATCH_DIM)


MENAGERIE_ENTRIES = [
    (
        "scScope",
        "build_scscope",
        "example_input_scscope",
        2019,
        "ported-pytorch",
    ),
]
