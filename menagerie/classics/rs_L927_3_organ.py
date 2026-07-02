# FAITHFUL PORT of gablg1/ORGAN @ 1e4494127f90ff2cb07842619a7c4559bc8899e9
#   (repo: https://github.com/gablg1/ORGAN, organ/generator.py, class
#   `Generator`) (original framework: TensorFlow 1.x graph-mode, tf.placeholder
#   + tf.variable_scope + tf.while_loop -- TF1.x cannot be installed/run
#   alongside this environment's torch stack, and this task's rules forbid
#   installing extra frameworks for a vendor, so the real code is transcribed
#   faithfully into base-env torch rather than run in-place).
# ORGAN (Guimaraes, Sanchez-Lengeling, Outeiral, Farias & Aspuru-Guzik,
# "Objective-Reinforced Generative Adversarial Networks (ORGAN) for
# Sequence Generation Models", arXiv:1705.10843, 2017) wraps SeqGAN
# (Yu, Zhang, Wang & Yu, AAAI 2017) with domain-specific objective rewards
# (drug-likeness / music theory scores) mixed into the GAN reward signal.
# The generator itself IS SeqGAN's generator verbatim: a hand-rolled LSTM
# cell (`create_recurrent_unit`, explicit Wi/Ui/bi, Wf/Uf/bf, Wog/Uog/bog,
# Wc/Uc/bc gate weights -- not `tf.nn.rnn_cell.LSTMCell`) followed by a
# linear output-vocabulary head (`create_output_unit`, Wo/bo), run either
# autoregressively (`_g_recurrence`, sampling from `tf.multinomial` each
# step) or teacher-forced over a real input token sequence
# (`_pretrain_recurrence`, producing `g_predictions`/`g_logits` -- this is
# the supervised-pretraining forward pass, and the one ported here as the
# traceable `forward`). Every gate equation, the LSTM state-tuple
# stack/unstack convention, and the embedding-lookup-then-recurrence
# structure are transcribed 1:1 from `create_recurrent_unit`/
# `create_output_unit`/`_pretrain_recurrence`; only the TF1.x placeholder/
# session/optimizer/summary machinery (`pretrain_step`, `generator_step`,
# tensorboard summaries -- training-loop plumbing, not architecture) is
# dropped, and TF's `tf.split`+`tf.stack` embedding-lookup preprocessing of
# `self.x` into `self.processed_x` is replaced with the equivalent single
# `nn.Embedding` lookup + `unbind(dim=1)` per-timestep loop.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ORGANGeneratorLSTMCell(nn.Module):
    """Hand-rolled LSTM cell, transcribed from `create_recurrent_unit`.
    Input/hidden weight matrices for input(i), forget(f), output(og), and
    candidate-memory(c) gates -- identical gate equations to the real
    TF1.x `unit(x, hidden_memory_tm1)` closure."""

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        # NOTE: the real repo declares `Ui` with shape [emb_dim, hidden_dim]
        # (matching Wi) even though it is matmul'd against the
        # hidden_dim-wide `previous_hidden_state` -- only valid when
        # emb_dim == hidden_dim, which is the real repo's own default
        # config. Transcribed as-is (not "fixed" to [hidden_dim,
        # hidden_dim]) to stay faithful to the actual upstream code.
        self.Wi = nn.Parameter(torch.empty(emb_dim, hidden_dim))
        self.Ui = nn.Parameter(torch.empty(emb_dim, hidden_dim))
        self.bi = nn.Parameter(torch.zeros(hidden_dim))

        self.Wf = nn.Parameter(torch.empty(emb_dim, hidden_dim))
        self.Uf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bf = nn.Parameter(torch.zeros(hidden_dim))

        self.Wog = nn.Parameter(torch.empty(emb_dim, hidden_dim))
        self.Uog = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bog = nn.Parameter(torch.zeros(hidden_dim))

        self.Wc = nn.Parameter(torch.empty(emb_dim, hidden_dim))
        self.Uc = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bc = nn.Parameter(torch.zeros(hidden_dim))

        for p in (self.Wi, self.Ui, self.Wf, self.Uf, self.Wog, self.Uog, self.Wc, self.Uc):
            nn.init.normal_(p, std=0.1)  # matches real init_matrix (tf.random_normal, stddev=0.1)

    def forward(self, x, h_prev, c_prev):
        # Input Gate
        i = torch.sigmoid(x @ self.Wi + h_prev @ self.Ui + self.bi)
        # Forget Gate
        f = torch.sigmoid(x @ self.Wf + h_prev @ self.Uf + self.bf)
        # Output Gate
        o = torch.sigmoid(x @ self.Wog + h_prev @ self.Uog + self.bog)
        # New Memory Cell
        c_ = torch.tanh(x @ self.Wc + h_prev @ self.Uc + self.bc)
        # Final Memory cell
        c = f * c_prev + i * c_
        # Current Hidden state
        h = o * torch.tanh(c)
        return h, c


class Generator(nn.Module):
    """ORGAN/SeqGAN generator, transcribed from organ/generator.py.

    Supervised-pretraining forward pass (`_pretrain_recurrence`): teacher-forced
    LSTM recurrence over an input token sequence, producing per-timestep
    output-vocabulary logits/probabilities -- this is `pretrain_step`'s
    computational graph (`g_predictions`/`g_logits`) minus the TF session/
    placeholder/optimizer machinery.
    """

    def __init__(self, num_emb, emb_dim, hidden_dim, sequence_length, start_token=0):
        super().__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token

        self.g_embeddings = nn.Parameter(torch.empty(num_emb, emb_dim))
        nn.init.normal_(self.g_embeddings, std=0.1)  # init_matrix

        self.g_recurrent_unit = ORGANGeneratorLSTMCell(emb_dim, hidden_dim)

        self.Wo = nn.Parameter(torch.empty(hidden_dim, num_emb))
        nn.init.normal_(self.Wo, std=0.1)
        self.bo = nn.Parameter(torch.zeros(num_emb))

    def g_output_unit(self, h):
        # hidden_state x Wo + bo -> vocab logits (create_output_unit's `unit`)
        return h @ self.Wo + self.bo

    def forward(self, x):
        """Teacher-forced pretraining pass (`_pretrain_recurrence`).

        Args:
            x (torch.LongTensor): token ids, shape (batch, sequence_length),
                not including the start token (matches the real `self.x`
                placeholder).

        Returns:
            g_predictions (torch.Tensor): softmax probabilities per step,
                shape (batch, sequence_length, num_emb).
            g_logits (torch.Tensor): pre-softmax logits per step, same shape.
        """
        batch_size = x.shape[0]
        device = x.device

        # processed_x: seq_length x batch_size x emb_dim (tf.split + embedding_lookup)
        processed_x = self.g_embeddings[x]  # (batch, seq_len, emb_dim)
        processed_x = processed_x.transpose(0, 1)  # (seq_len, batch, emb_dim)

        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        start_tokens = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
        x_t = self.g_embeddings[start_tokens]  # (batch, emb_dim)

        predictions = []
        logits = []
        for t in range(self.sequence_length):
            h, c = self.g_recurrent_unit(x_t, h, c)
            o_t = self.g_output_unit(h)  # (batch, num_emb) logits
            predictions.append(torch.softmax(o_t, dim=-1))
            logits.append(o_t)
            x_t = processed_x[t]  # teacher forcing: next real token's embedding

        g_predictions = torch.stack(predictions, dim=1)  # (batch, seq_len, num_emb)
        g_logits = torch.stack(logits, dim=1)  # (batch, seq_len, num_emb)
        return g_predictions, g_logits


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------

_NUM_EMB = 32  # small SMILES/char vocabulary
# Real repo's `create_recurrent_unit` declares `Ui` with shape
# `[emb_dim, hidden_dim]` (matching Wi) but matmuls it against
# `previous_hidden_state`, which has shape `hidden_dim` -- a latent
# shape-coupling in the upstream code that only type-checks when
# emb_dim == hidden_dim. That equality is exactly the real repo's own
# default config (`organ/__init__.py`: GEN_EMB_DIM = GEN_HIDDEN_DIM = 32),
# so this staging entry mirrors it rather than picking independent widths.
_EMB_DIM = 16
_HIDDEN_DIM = 16
_SEQ_LEN = 12
_BATCH_SIZE = 2


def build_organ():
    return Generator(
        num_emb=_NUM_EMB,
        emb_dim=_EMB_DIM,
        hidden_dim=_HIDDEN_DIM,
        sequence_length=_SEQ_LEN,
        start_token=0,
    )


def example_input_organ():
    torch.manual_seed(0)
    x = torch.randint(low=0, high=_NUM_EMB, size=(_BATCH_SIZE, _SEQ_LEN), dtype=torch.long)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "ORGAN",
        build_organ,
        example_input_organ,
        2017,
        "CODE",
    ),
]
