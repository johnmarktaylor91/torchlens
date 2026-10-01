# SOURCE: vendored from snakeztc/NeuralDialog-LaRL @ master
# (latent_dialog/models_task.py::SysPerfectBD2Cat,
#  latent_dialog/enc2dec/encoders.py, latent_dialog/enc2dec/decoders.py,
#  latent_dialog/enc2dec/base_modules.py, latent_dialog/nn_lib.py,
#  latent_dialog/base_models.py, latent_dialog/criterions.py)
#
# LaRL (Latent Action Reinforcement Learning) -- Zhao, Xie, Eskenazi, "Rethinking
# Action Spaces for Reinforcement Learning in End-to-end Dialog Agents with
# Latent Variable Models", NAACL 2019 oral. Real architecture: a bidirectional
# GRU utterance encoder feeding a `Hidden2Discrete` linear projection that
# parameterizes a *categorical* latent dialogue-action distribution
# (`y_size` independent `k_size`-way categoricals), a Gumbel-Softmax
# connector (`GumbelConnector`) that draws a differentiable one-hot sample
# from that categorical latent, a learned linear "z-embedding" that turns
# the sampled discrete latent action into the attention-decoder's initial
# hidden state, and an attention-augmented GRU response decoder
# (`DecoderRNN`/`Attention`) that reconstructs the system response
# conditioned on that latent action. This categorical-latent-action
# bottleneck (in place of the continuous VAE latents used by prior latent
# dialog-act work) is LaRL's own architectural contribution -- it is not a
# stock encoder-decoder or VAE, so this is vendored (rung 2) rather than
# recipe'd.
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `SysPerfectBD2Cat.__init__` originally took a `corpus` object (with
#     `corpus.vocab`, `corpus.vocab_dict`, `corpus.bs_size`, `corpus.db_size`)
#     built by LaRL's MultiWOZ data pipeline; the traced entry constructs a
#     tiny synthetic `_TinyCorpus`/`_TinyConfig` with the same attributes
#     (small vocab, `bs_size`/`db_size` belief-state/database-vector widths)
#     instead of loading the real MultiWOZ corpus, per the menagerie "tiny
#     config, random init" convention. `SysPerfectBD2Cat`,
#     `RnnUttEncoder`/`EncoderRNN`/`BaseRNN`, `Hidden2Discrete`,
#     `GumbelConnector`, `DecoderRNN`/`Attention`, `NLLEntropy`,
#     `CatKLLoss`, `Entropy` are copied verbatim from the source.
#   - `forward()` is copied verbatim for `mode=TEACH_FORCE` (the standard
#     supervised-training / teacher-forcing forward pass, which is what a
#     plain eager capture exercises); `forward_rl()` (LaRL's separate
#     REINFORCE self-play rollout method, an autoregressive sampling loop
#     with no fixed computation graph) is dropped as dead code for this
#     entry, matching the menagerie convention of tracing the model's
#     standard forward/training computation rather than a
#     policy-gradient rollout loop.
#   - `DecoderRNN.forward()`/`forward_step()` are copied verbatim; the
#     `mode != TEACH_FORCE` free-running/beam-search branches (also
#     autoregressive sampling loops using `.topk()`/`th.no_grad()`
#     BOS-token seeding, not exercised when `mode=TEACH_FORCE`) and the
#     `write()`/`forward_rl()`/`_step()` RL-only decode helpers are dropped
#     as dead code for the traced path.
#   - `BaseModel.np2var` (numpy -> `Variable` + dtype cast) is reproduced
#     verbatim from `latent_dialog/base_models.py` so `SysPerfectBD2Cat`'s
#     inherited `self.np2var(...)` calls work unchanged; `get_optimizer`/
#     `get_clf_optimizer`/`backward`/`extract_short_ctx`/`flatten_context`
#     (training-loop and MultiWOZ-context-window helpers, not used by the
#     forward computation graph) are dropped.
#   - `INT`/`LONG`/`FLOAT`/`cast_type`/`Pack` are reproduced verbatim from
#     `latent_dialog/utils.py` (pure dtype-casting/dict-subclass helpers,
#     no external deps); the NLTK tokenizer helpers in the same file
#     (unused by the model) are dropped.
#   - `self.log_uniform_y`/`self.eye` are kept as plain (non-`Variable`)
#     tensors -- `torch.autograd.Variable` is a no-op alias in modern torch,
#     so wrapping is dropped throughout in favor of plain tensors/ops with
#     identical numerics.

import torch as th
import torch.nn as nn
import torch.nn.functional as F

INT = 0
LONG = 1
FLOAT = 2

TEACH_FORCE = "teacher_forcing"
GEN = "gen"


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(th.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(th.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(th.IntTensor)
        elif dtype == LONG:
            var = var.type(th.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


# ---------------------------------------------------------------------------
# latent_dialog/enc2dec/base_modules.py
# ---------------------------------------------------------------------------


class BaseRNN(nn.Module):
    KEY_ATTN_SCORE = "attention_score"
    KEY_SEQUENCE = "sequence"

    def __init__(
        self,
        input_dropout_p,
        rnn_cell,
        input_size,
        hidden_size,
        num_layers,
        output_dropout_p,
        bidirectional,
    ):
        super(BaseRNN, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == "lstm":
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell Type: {0}".format(rnn_cell))
        self.rnn = self.rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=output_dropout_p,
            bidirectional=bidirectional,
        )

        if rnn_cell.lower() == "lstm":
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)


# ---------------------------------------------------------------------------
# latent_dialog/enc2dec/encoders.py
# ---------------------------------------------------------------------------


class EncoderRNN(BaseRNN):
    def __init__(
        self,
        input_dropout_p,
        rnn_cell,
        input_size,
        hidden_size,
        num_layers,
        output_dropout_p,
        bidirectional,
        variable_lengths,
    ):
        super(EncoderRNN, self).__init__(
            input_dropout_p=input_dropout_p,
            rnn_cell=rnn_cell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dropout_p=output_dropout_p,
            bidirectional=bidirectional,
        )
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals=None):
        if goals is not None:
            batch_size, max_ctx_len, ctx_nhid = input_var.size()
            goals = goals.view(goals.size(0), 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, 1).view(batch_size, max_ctx_len, -1)
            input_var = th.cat([input_var, goals_rep], dim=2)

        embedded = self.input_dropout(input_var)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        feat_size,
        goal_nhid,
        rnn_cell,
        utt_cell_size,
        num_layers,
        input_dropout_p,
        output_dropout_p,
        bidirectional,
        variable_lengths,
        use_attn,
        embedding=None,
    ):
        super(RnnUttEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(
            input_dropout_p=input_dropout_p,
            rnn_cell=rnn_cell,
            input_size=embedding_dim + feat_size + goal_nhid,
            hidden_size=utt_cell_size,
            num_layers=num_layers,
            output_dropout_p=output_dropout_p,
            bidirectional=bidirectional,
            variable_lengths=variable_lengths,
        )

        self.utt_cell_size = utt_cell_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.utt_cell_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, goals=None):
        batch_size, max_ctx_len, max_utt_len = utterances.size()
        flat_words = utterances.view(-1, max_utt_len)
        word_embeddings = self.embedding(flat_words)
        flat_mask = th.sign(flat_words).float()
        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            word_embeddings = th.cat([word_embeddings, flat_feats], dim=2)

        if goals is not None:
            goals = goals.view(goals.size(0), 1, 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, max_utt_len, 1).view(
                batch_size * max_ctx_len, max_utt_len, -1
            )
            word_embeddings = th.cat([word_embeddings, goals_rep], dim=2)

        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim() - 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True) + 1e-10)).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = th.sum(utt_embedded, dim=1)
        else:
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_len, self.output_size)
        return (
            utt_embedded,
            word_embeddings.contiguous().view(batch_size, max_ctx_len * max_utt_len, -1),
            enc_outs.contiguous().view(batch_size, max_ctx_len * max_utt_len, -1),
        )


# ---------------------------------------------------------------------------
# latent_dialog/nn_lib.py
# ---------------------------------------------------------------------------


class Hidden2Discrete(nn.Module):
    def __init__(self, input_size, y_size, k_size, is_lstm=False, has_bias=True):
        super(Hidden2Discrete, self).__init__()
        self.y_size = y_size
        self.k_size = k_size
        latent_size = self.k_size * self.y_size
        if is_lstm:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
            self.p_c = nn.Linear(input_size, latent_size, bias=has_bias)
        else:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)

        self.is_lstm = is_lstm

    def forward(self, inputs):
        if self.is_lstm:
            h, c = inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)
            logits = self.p_h(h) + self.p_c(c)
        else:
            logits = self.p_h(inputs)
        logits = logits.view(-1, self.k_size)
        log_qy = F.log_softmax(logits, dim=1)
        return logits, log_qy


class GumbelConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = th.rand(logits.size())
        sample = -th.log(-th.log(u + eps) + eps)
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim() - 1)

    def forward(self, logits, temperature=1.0, hard=False, return_max_id=False):
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = th.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(th.zeros(y.size()), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y


# ---------------------------------------------------------------------------
# latent_dialog/enc2dec/decoders.py
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dec_cell_size, ctx_cell_size, attn_mode, project):
        super(Attention, self).__init__()
        self.dec_cell_size = dec_cell_size
        self.ctx_cell_size = ctx_cell_size
        self.attn_mode = attn_mode
        if project:
            self.linear_out = nn.Linear(dec_cell_size + ctx_cell_size, dec_cell_size)
        else:
            self.linear_out = None

        if attn_mode == "general":
            self.dec_w = nn.Linear(dec_cell_size, ctx_cell_size)
        elif attn_mode == "cat":
            self.dec_w = nn.Linear(dec_cell_size, dec_cell_size)
            self.attn_w = nn.Linear(ctx_cell_size, dec_cell_size)
            self.query_w = nn.Linear(dec_cell_size, 1)

    def forward(self, output, context):
        batch_size = output.size(0)
        max_ctx_len = context.size(1)

        if self.attn_mode == "dot":
            attn = th.bmm(output, context.transpose(1, 2))
        elif self.attn_mode == "general":
            mapped_output = self.dec_w(output)
            attn = th.bmm(mapped_output, context.transpose(1, 2))
        elif self.attn_mode == "cat":
            mapped_output = self.dec_w(output)
            mapped_attn = self.attn_w(context)
            tiled_output = mapped_output.unsqueeze(2).repeat(1, 1, max_ctx_len, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = th.tanh(tiled_output + tiled_attn)
            attn = self.query_w(fc1).squeeze(-1)
        else:
            raise ValueError("Unknown attention mode")

        attn = F.softmax(attn.view(-1, max_ctx_len), dim=1).view(batch_size, -1, max_ctx_len)
        mix = th.bmm(attn, context)
        combined = th.cat((mix, output), dim=2)
        if self.linear_out is None:
            return combined, attn
        else:
            output = th.tanh(
                self.linear_out(combined.view(-1, self.dec_cell_size + self.ctx_cell_size))
            ).view(batch_size, -1, self.dec_cell_size)
            return output, attn


class DecoderRNN(BaseRNN):
    def __init__(
        self,
        input_dropout_p,
        rnn_cell,
        input_size,
        hidden_size,
        num_layers,
        output_dropout_p,
        bidirectional,
        vocab_size,
        use_attn,
        ctx_cell_size,
        attn_mode,
        sys_id,
        eos_id,
        use_gpu,
        max_dec_len,
        embedding=None,
    ):
        super(DecoderRNN, self).__init__(
            input_dropout_p=input_dropout_p,
            rnn_cell=rnn_cell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dropout_p=output_dropout_p,
            bidirectional=bidirectional,
        )

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.embedding = embedding

        self.use_attn = use_attn
        if self.use_attn:
            self.attention = Attention(
                dec_cell_size=hidden_size,
                ctx_cell_size=ctx_cell_size,
                attn_mode=attn_mode,
                project=True,
            )

        self.dec_cell_size = hidden_size
        self.output_size = vocab_size
        self.project = nn.Linear(self.dec_cell_size, self.output_size)
        self.log_softmax = F.log_softmax

        self.sys_id = sys_id
        self.eos_id = eos_id
        self.use_gpu = use_gpu
        self.max_dec_len = max_dec_len

    def forward(
        self,
        batch_size,
        dec_inputs,
        dec_init_state,
        attn_context,
        mode,
        gen_type,
        beam_size,
        goal_hid=None,
    ):
        ret_dict = dict()

        if self.use_attn:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if mode == GEN:
            dec_inputs = None

        decoder_input = dec_inputs
        decoder_hidden_state = dec_init_state

        if mode == TEACH_FORCE:
            prob_outputs, decoder_hidden_state, attn = self.forward_step(
                input_var=decoder_input,
                hidden_state=decoder_hidden_state,
                encoder_outputs=attn_context,
                goal_hid=goal_hid,
            )
        else:
            raise NotImplementedError("Only TEACH_FORCE mode is vendored for this menagerie entry")

        ret_dict[DecoderRNN.KEY_SEQUENCE] = []

        return prob_outputs, decoder_hidden_state, ret_dict

    def forward_step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var)

        if goal_hid is not None:
            goal_hid = goal_hid.view(goal_hid.size(0), 1, goal_hid.size(1))
            goal_rep = goal_hid.repeat(1, output_seq_len, 1)
            embedded = th.cat([embedded, goal_rep], dim=2)

        embedded = self.input_dropout(embedded)

        output, hidden_s = self.rnn(embedded, hidden_state)

        attn = None
        if self.use_attn:
            output, attn = self.attention(output, encoder_outputs)

        logits = self.project(output.contiguous().view(-1, self.dec_cell_size))
        prediction = self.log_softmax(logits, dim=logits.dim() - 1).view(
            batch_size, output_seq_len, -1
        )
        return prediction, hidden_s, attn


# ---------------------------------------------------------------------------
# latent_dialog/criterions.py
# ---------------------------------------------------------------------------


class NLLEntropy(nn.Module):
    def __init__(self, padding_idx, avg_type):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = avg_type

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        pred = net_output.reshape(-1, net_output.size(-1))
        target = labels.reshape(-1)

        if self.avg_type is None:
            loss = F.nll_loss(pred, target, reduction="sum", ignore_index=self.padding_idx)
        elif self.avg_type == "seq":
            loss = F.nll_loss(pred, target, reduction="sum", ignore_index=self.padding_idx)
            loss = loss / batch_size
        elif self.avg_type == "real_word":
            loss = F.nll_loss(pred, target, ignore_index=self.padding_idx, reduction="none")
            loss = loss.view(-1, net_output.size(1))
            loss = th.sum(loss, dim=1)
            word_cnt = th.sum(th.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = th.mean(loss)
        elif self.avg_type == "word":
            loss = F.nll_loss(pred, target, reduction="mean", ignore_index=self.padding_idx)
        else:
            raise ValueError("Unknown average type")

        return loss


class CatKLLoss(nn.Module):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """qy * log(q(y)/p(y))"""
        qy = th.exp(log_qy)
        y_kl = th.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return th.mean(y_kl)
        else:
            return th.sum(y_kl) / batch_size


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """-qy log(qy)"""
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = th.exp(log_qy)
        h_q = th.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return th.mean(h_q)
        else:
            return th.sum(h_q) / batch_size


# ---------------------------------------------------------------------------
# latent_dialog/base_models.py (trimmed: only what SysPerfectBD2Cat needs)
# ---------------------------------------------------------------------------


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        # NOTE (menagerie vendoring): original took a raw numpy array and
        # called `th.from_numpy(inputs)`; the traced entry point below
        # feeds tensors directly (torchlens normalizes numpy positional
        # trace inputs to tensors before `forward()` is invoked), so this
        # accepts both without changing the cast/dtype semantics below.
        var = inputs if th.is_tensor(inputs) else th.from_numpy(inputs)
        return cast_type(var, dtype, self.use_gpu)

    def forward(self, *inputs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# latent_dialog/models_task.py :: SysPerfectBD2Cat
# ---------------------------------------------------------------------------


class SysPerfectBD2Cat(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict["<s>"]
        self.eos_id = self.vocab_dict["</s>"]
        self.pad_id = self.vocab_dict["<pad>"]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=config.embed_size,
            feat_size=0,
            goal_nhid=0,
            rnn_cell=config.utt_rnn_cell,
            utt_cell_size=config.utt_cell_size,
            num_layers=config.num_layers,
            input_dropout_p=config.dropout,
            output_dropout_p=config.dropout,
            bidirectional=config.bi_utt_cell,
            variable_lengths=False,
            use_attn=config.enc_use_attn,
            embedding=self.embedding,
        )

        self.c2z = Hidden2Discrete(
            self.utt_encoder.output_size + self.db_size + self.bs_size,
            config.y_size,
            config.k_size,
            is_lstm=False,
        )
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = Hidden2Discrete(
                    self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                    config.y_size,
                    config.k_size,
                    is_lstm=False,
                )
            else:
                self.xc2z = Hidden2Discrete(
                    self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False
                )

        self.decoder = DecoderRNN(
            input_dropout_p=config.dropout,
            rnn_cell=config.dec_rnn_cell,
            input_size=config.embed_size,
            hidden_size=config.dec_cell_size,
            num_layers=config.num_layers,
            output_dropout_p=config.dropout,
            bidirectional=False,
            vocab_size=self.vocab_size,
            use_attn=config.dec_use_attn,
            ctx_cell_size=config.dec_cell_size,
            attn_mode=config.dec_attn_mode,
            sys_id=self.bos_id,
            eos_id=self.eos_id,
            use_gpu=config.use_gpu,
            max_dec_len=config.max_dec_len,
            embedding=self.embedding,
        )

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = th.log(th.ones(1) / config.k_size)
        self.eye = th.eye(self.config.y_size).unsqueeze(0)
        self.beta = self.config.beta if hasattr(self.config, "beta") else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def forward(
        self, data_feed, mode, clf=False, gen_type="greedy", use_py=None, return_latent=False
    ):
        ctx_lens = data_feed["context_lens"]  # (batch_size, )
        short_ctx_utts = self.np2var(data_feed["contexts"], LONG)
        out_utts = self.np2var(data_feed["outputs"], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed["bs"], FLOAT)  # (batch_size, bs_size)
        db_label = self.np2var(data_feed["db"], FLOAT)  # (batch_size, db_size)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode == GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(
                sample_y.view(1, -1, self.config.y_size * self.config.k_size)
            )
            attn_context = None

        if self.config.dec_rnn_cell == "lstm":
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(
            batch_size=batch_size,
            dec_inputs=dec_inputs,
            dec_init_state=dec_init_state,
            attn_context=attn_context,
            mode=mode,
            gen_type=gen_type,
            beam_size=self.config.beam_size,
        )
        if mode == GEN:
            ret_dict["sample_z"] = sample_y
            ret_dict["log_qy"] = log_qy
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(
                log_qy, unit_average=True
            )
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result["pi_kl"] = pi_kl
            result["diversity"] = th.mean(p)
            result["nll"] = self.nll(dec_outputs, labels)
            result["b_pr"] = b_pr
            result["mi"] = mi
            return result


# ---------------------------------------------------------------------------
# Menagerie entry point
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 64
_EMBED = 16
_UTT_CELL = 16
_DEC_CELL = 16
_Y_SIZE = 4
_K_SIZE = 4
_BS_SIZE = 6
_DB_SIZE = 6
_BATCH = 2
_CTX_LEN = 8
_OUT_LEN = 6


class _TinyConfig:
    use_gpu = False
    k_size = _K_SIZE
    y_size = _Y_SIZE
    simple_posterior = True
    contextual_posterior = False
    embed_size = _EMBED
    utt_rnn_cell = "gru"
    utt_cell_size = _UTT_CELL
    num_layers = 1
    dropout = 0.0
    bi_utt_cell = True
    enc_use_attn = False
    dec_rnn_cell = "gru"
    dec_cell_size = _DEC_CELL
    dec_use_attn = True
    dec_attn_mode = "cat"
    max_dec_len = _OUT_LEN
    beam_size = 1
    avg_type = "seq"
    beta = 0.001
    use_pr = 1.0
    use_mi = False
    use_diversity = False


class _TinyCorpus:
    vocab = ["<pad>", "<s>", "</s>"] + [f"w{i}" for i in range(_VOCAB_SIZE - 3)]
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    bs_size = _BS_SIZE
    db_size = _DB_SIZE


class _LaRLEntry(nn.Module):
    """Traces `SysPerfectBD2Cat.forward(data_feed, mode=TEACH_FORCE)` --
    the categorical-latent-action dialogue policy + attention response
    decoder."""

    def __init__(self):
        super().__init__()
        self.model = SysPerfectBD2Cat(_TinyCorpus(), _TinyConfig())

    def forward(self, contexts, context_lens, outputs, bs, db):
        data_feed = {
            "contexts": contexts,
            "context_lens": context_lens,
            "outputs": outputs,
            "bs": bs,
            "db": db,
        }
        return self.model(data_feed, mode=TEACH_FORCE)


def build_larl():
    m = _LaRLEntry()
    m.eval()
    return m


def example_input_larl():
    contexts = th.randint(1, _VOCAB_SIZE, (_BATCH, _CTX_LEN))
    context_lens = th.full((_BATCH,), _CTX_LEN, dtype=th.int64)
    outputs = th.randint(1, _VOCAB_SIZE, (_BATCH, _OUT_LEN))
    bs = th.rand(_BATCH, _BS_SIZE)
    db = th.rand(_BATCH, _DB_SIZE)
    return (contexts, context_lens, outputs, bs, db)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "LaRL (Latent Action RL, Categorical)",
        build_larl,
        example_input_larl,
        2019,
        "vendored-pytorch",
    ),
]
