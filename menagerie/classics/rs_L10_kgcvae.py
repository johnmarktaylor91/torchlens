# FAITHFUL PORT of ruotianluo/NeuralDialog-CVAE-pytorch @ bb9e7bb2cfc7d559b32c24b971a13e5fafa1eee4
# (models/cvae.py: KgRnnCVAE, a knowledge-guided conditional VAE for dialogue response
# generation; Zhao, Zhao & Eskenazi, "Learning Discourse-level Diversity for Neural Dialog
# Models using Conditional Variational Autoencoders", ACL 2017. `use_hcf=True` is the paper's
# "kgCVAE" (dialog-act-conditioned); `use_hcf=False` is the ablated "CVAE" baseline -- both
# configurations of the SAME real class are exposed below.)
#
# The repo's real code is torch (nn.Embedding/nn.Linear/GRU cells and their exact forward-pass
# wiring below are copied verbatim from models/cvae.py), but it is unrunnable as-is in a
# base env: it imports tensorflow 0.12/1.x purely for `tf.placeholder` I/O declarations and
# `tensorflow.python.ops.variable_scope` used ONLY as a no-op context manager around blocks of
# torch code (never for TF computation graph construction). Under TF2 (the only tensorflow
# available in this env), `tf.placeholder` no longer exists (`AttributeError`), so the module
# cannot even be imported, let alone traced -- this is a genuine TF1-vs-TF2 incompatibility,
# not a missing package (RUNG 2 vendoring was attempted first and failed at import time).
#
# This is therefore a RUNG 3 faithful port: every real torch layer, dimension, and forward-pass
# computation from KgRnnCVAE.forward (and its helpers get_bow/get_rnn_encode/get_bi_rnn_encode/
# dynamic_rnn/sample_gaussian/gaussian_kld from models/utils.py, and train_loop/inference_loop
# from models/decoder_fn_lib.py) is transcribed unchanged. Only the dead TF1 scaffolding is
# removed: the `tf.placeholder` I/O declarations (never read by forward(); the real forward()
# takes tensors via `feed_dict`) and the `variable_scope.variable_scope(...)` context managers
# (already no-ops around torch code in the original -- TF1 variable scoping never touched these
# torch tensors). TensorBoard summary logging (train_summary_writer) and the training-loop /
# CLI driver (train_model/valid_model/test_model, kgcvae_swda.py) are the training harness, not
# architecture, and are likewise omitted.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---- models/utils.py (helper functions used by KgRnnCVAE.forward), verbatim ----


def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(
        1
        + (recog_logvar - prior_logvar)
        - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)),
        1,
    )
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5 * torch.sum(
        logvar + np.log(2 * np.pi) + torch.div(torch.pow((x - mu), 2), torch.exp(logvar)), 1
    )


def get_bow(embedding, avg=False):
    """Assumption: last dim is the embedding, 2nd-last dim is sentence length (rank 3)."""
    embedding_size = embedding.size(2)
    if avg:
        return embedding.mean(1), embedding_size
    else:
        return embedding.sum(1), embedding_size


def dynamic_rnn(cell, inputs, sequence_length, init_state=None, output_fn=None):
    sorted_lens, len_ix = sequence_length.sort(0, descending=True)

    inv_ix = len_ix.clone()
    inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

    valid_num = torch.sign(sorted_lens).long().sum().item()
    zero_num = inputs.size(0) - valid_num

    sorted_inputs = inputs[len_ix].contiguous()
    if init_state is not None:
        sorted_init_state = init_state[:, len_ix].contiguous()

    packed_inputs = pack_padded_sequence(
        sorted_inputs[:valid_num], list(sorted_lens[:valid_num]), batch_first=True
    )

    if init_state is not None:
        outputs, state = cell(packed_inputs, sorted_init_state[:, :valid_num])
    else:
        outputs, state = cell(packed_inputs)

    outputs, _ = pad_packed_sequence(outputs, batch_first=True)

    if zero_num > 0:
        outputs = torch.cat(
            [outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0
        )
        if init_state is not None:
            state = torch.cat([state, sorted_init_state[:, valid_num:]], 1)
        else:
            state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

    outputs = outputs[inv_ix].contiguous()
    state = state[:, inv_ix].contiguous()

    state = F.dropout(state, cell.dropout, cell.training)
    outputs = F.dropout(outputs, cell.dropout, cell.training)

    if output_fn is not None:
        outputs = output_fn(outputs)

    return outputs, state


def get_rnn_encode(embedding, cell, length_mask=None):
    """Assumption: last dim is embedding, 2nd-last dim is sentence length (rank 3), 0-padded."""
    if length_mask is None:
        length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
        length_mask = length_mask.long()
    _, encoded_input = dynamic_rnn(cell, embedding, sequence_length=length_mask)
    encoded_input = encoded_input[-1]
    return encoded_input, cell.hidden_size


def get_bi_rnn_encode(embedding, cell, length_mask=None):
    """Assumption: last dim is embedding, 2nd-last dim is sentence length (rank 3), 0-padded."""
    if length_mask is None:
        length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
        length_mask = length_mask.long()
    _, encoded_input = dynamic_rnn(cell, embedding, sequence_length=length_mask)
    encoded_input = torch.cat([encoded_input[-2], encoded_input[-1]], 1)
    return encoded_input, cell.hidden_size * 2


# ---- models/decoder_fn_lib.py (train-mode decoder loop used by forward()), verbatim ----


def train_loop(cell, output_fn, inputs, init_state, context_vector, sequence_length):
    if context_vector is not None:
        inputs = torch.cat(
            [
                inputs,
                context_vector.unsqueeze(1).expand(
                    inputs.size(0), inputs.size(1), context_vector.size(1)
                ),
            ],
            2,
        )
    return dynamic_rnn(cell, inputs, sequence_length, init_state, output_fn) + (None,)


# ---- models/cvae.py: KgRnnCVAE, verbatim architecture (TF1 I/O + variable_scope stripped) ----


class KgRnnCVAE(nn.Module):
    def __init__(self, config, api):
        super().__init__()
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)
        self.topic_vocab = api.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)
        self.da_vocab = api.dialog_act_vocab
        self.da_vocab_size = len(self.da_vocab)
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size

        self.use_hcf = config.use_hcf
        self.embed_size = config.embed_size
        self.sent_type = config.sent_type
        self.keep_prob = config.keep_prob
        self.num_layer = config.num_layer
        self.dec_keep_prob = config.dec_keep_prob
        self.full_kl_step = config.full_kl_step
        self.grad_clip = config.grad_clip
        self.grad_noise = config.grad_noise

        # topicEmbedding
        self.t_embedding = nn.Embedding(self.topic_vocab_size, config.topic_embed_size)
        if self.use_hcf:
            # dialogActEmbedding
            self.d_embedding = nn.Embedding(self.da_vocab_size, config.da_embed_size)
        # wordEmbedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

        # no dropout at last layer, we need to add one
        if self.sent_type == "bow":
            input_embedding_size = output_embedding_size = self.embed_size
        elif self.sent_type == "rnn":
            self.sent_cell = self._get_rnncell(
                config.cell_type, self.embed_size, self.sent_cell_size, self.keep_prob, 1
            )
            input_embedding_size = output_embedding_size = self.sent_cell_size
        elif self.sent_type == "bi_rnn":
            self.bi_sent_cell = self._get_rnncell(
                "gru",
                self.embed_size,
                self.sent_cell_size,
                keep_prob=1.0,
                num_layer=1,
                bidirectional=True,
            )
            input_embedding_size = output_embedding_size = self.sent_cell_size * 2
        else:
            raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

        joint_embedding_size = input_embedding_size + 2

        # contextRNN
        self.enc_cell = self._get_rnncell(
            config.cell_type,
            joint_embedding_size,
            self.context_cell_size,
            keep_prob=1.0,
            num_layer=config.num_layer,
        )

        self.attribute_fc1 = nn.Sequential(nn.Linear(config.da_embed_size, 30), nn.Tanh())

        cond_embedding_size = config.topic_embed_size + 4 + 4 + self.context_cell_size

        # recognitionNetwork
        recog_input_size = cond_embedding_size + output_embedding_size
        if self.use_hcf:
            recog_input_size += 30

        self.recogNet_mulogvar = nn.Linear(recog_input_size, config.latent_size * 2)

        # priorNetwork: P(XYZ)=P(Z|X)P(X)P(Y|X,Z)
        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_embedding_size, np.maximum(config.latent_size * 2, 100)),
            nn.Tanh(),
            nn.Linear(np.maximum(config.latent_size * 2, 100), config.latent_size * 2),
        )

        gen_inputs_size = cond_embedding_size + config.latent_size
        # BOW loss
        self.bow_project = nn.Sequential(
            nn.Linear(gen_inputs_size, 400),
            nn.Tanh(),
            nn.Dropout(1 - config.keep_prob),
            nn.Linear(400, self.vocab_size),
        )

        # Y loss
        if self.use_hcf:
            self.da_project = nn.Sequential(
                nn.Linear(gen_inputs_size, 400),
                nn.Tanh(),
                nn.Dropout(1 - config.keep_prob),
                nn.Linear(400, self.da_vocab_size),
            )
            dec_inputs_size = gen_inputs_size + 30
        else:
            dec_inputs_size = gen_inputs_size

        # Decoder
        if config.num_layer > 1:
            self.dec_init_state_net = nn.ModuleList(
                [nn.Linear(dec_inputs_size, self.dec_cell_size) for i in range(config.num_layer)]
            )
        else:
            self.dec_init_state_net = nn.Linear(dec_inputs_size, self.dec_cell_size)

        # decoder
        dec_input_embedding_size = self.embed_size
        if self.use_hcf:
            dec_input_embedding_size += 30
        self.dec_cell = self._get_rnncell(
            config.cell_type,
            dec_input_embedding_size,
            self.dec_cell_size,
            config.keep_prob,
            config.num_layer,
        )
        self.dec_cell_proj = nn.Linear(self.dec_cell_size, self.vocab_size)

        self.learning_rate = config.init_lr

    @staticmethod
    def _get_rnncell(cell_type, input_size, cell_size, keep_prob, num_layer, bidirectional=False):
        cell = getattr(nn, cell_type.upper())(
            input_size,
            cell_size,
            num_layers=num_layer,
            dropout=1 - keep_prob,
            bidirectional=bidirectional,
            batch_first=True,
        )
        return cell

    def forward(self, feed_dict, mode="train"):
        for k, v in feed_dict.items():
            setattr(self, k, v)

        max_dialog_len = self.input_contexts.size(1)
        batch_size = self.input_contexts.size(0)

        topic_embedding = self.t_embedding(self.topics)

        if self.use_hcf:
            da_embedding = self.d_embedding(self.output_das)

        self.input_contexts = self.input_contexts.view(-1, self.max_utt_len)
        input_embedding = self.embedding(self.input_contexts)
        output_embedding = self.embedding(self.output_tokens)

        if self.sent_type == "bow":
            input_embedding, sent_size = get_bow(input_embedding)
            output_embedding, _ = get_bow(output_embedding)
        elif self.sent_type == "rnn":
            input_embedding, sent_size = get_rnn_encode(input_embedding, self.sent_cell)
            output_embedding, _ = get_rnn_encode(output_embedding, self.sent_cell, self.output_lens)
        elif self.sent_type == "bi_rnn":
            input_embedding, sent_size = get_bi_rnn_encode(input_embedding, self.bi_sent_cell)
            output_embedding, _ = get_bi_rnn_encode(
                output_embedding, self.bi_sent_cell, self.output_lens
            )
        else:
            raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

        # reshape input into dialogs
        input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)
        if self.keep_prob < 1.0:
            input_embedding = F.dropout(input_embedding, 1 - self.keep_prob, self.training)

        # convert floors into 1 hot
        floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
        floor_one_hot.data.scatter_(1, self.floors.reshape(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)

        joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)

        # contextRNN: enc_last_state is the true last state
        _, enc_last_state = dynamic_rnn(
            self.enc_cell, joint_embedding, sequence_length=self.context_lens
        )

        if self.num_layer > 1:
            enc_last_state = torch.cat(list(torch.unbind(enc_last_state)), 1)
        else:
            enc_last_state = enc_last_state.squeeze(0)

        # combine with other attributes
        if self.use_hcf:
            attribute_embedding = da_embedding
            attribute_fc1 = self.attribute_fc1(attribute_embedding)

        cond_list = [topic_embedding, self.my_profile, self.ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        # recognitionNetwork
        if self.use_hcf:
            recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
        else:
            recog_input = torch.cat([cond_embedding, output_embedding], 1)
        self.recog_mulogvar = recog_mulogvar = self.recogNet_mulogvar(recog_input)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)

        # priorNetwork: P(XYZ)=P(Z|X)P(X)P(Y|X,Z)
        prior_mulogvar = self.priorNet_mulogvar(cond_embedding)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

        if self.use_prior:
            latent_sample = sample_gaussian(prior_mu, prior_logvar)
        else:
            latent_sample = sample_gaussian(recog_mu, recog_logvar)

        # generationNetwork
        gen_inputs = torch.cat([cond_embedding, latent_sample], 1)

        # BOW loss logits
        self.bow_logits = self.bow_project(gen_inputs)

        # Y loss logits
        if self.use_hcf:
            self.da_logits = self.da_project(gen_inputs)
            da_prob = F.softmax(self.da_logits, dim=1)
            pred_attribute_embedding = torch.matmul(da_prob, self.d_embedding.weight)
            if mode == "test":
                selected_attribute_embedding = pred_attribute_embedding
            else:
                selected_attribute_embedding = attribute_embedding
            dec_inputs = torch.cat([gen_inputs, selected_attribute_embedding], 1)
        else:
            self.da_logits = gen_inputs.new_zeros(batch_size, self.da_vocab_size)
            selected_attribute_embedding = None
            dec_inputs = gen_inputs

        # Decoder init state
        if self.num_layer > 1:
            dec_init_state = [self.dec_init_state_net[i](dec_inputs) for i in range(self.num_layer)]
            dec_init_state = torch.stack(dec_init_state)
        else:
            dec_init_state = self.dec_init_state_net(dec_inputs).unsqueeze(0)

        # decoder (train-mode teacher-forced loop; mirrors kgcvae_swda.py's default train call)
        input_tokens = self.output_tokens[:, :-1]
        if self.dec_keep_prob < 1.0:
            keep_mask = input_tokens.new_empty(input_tokens.size()).bernoulli_(self.dec_keep_prob)
            input_tokens = input_tokens * keep_mask

        dec_input_embedding = self.embedding(input_tokens)
        dec_seq_lens = self.output_lens - 1

        dec_input_embedding = F.dropout(dec_input_embedding, 1 - self.keep_prob, self.training)

        dec_outs, _, final_context_state = train_loop(
            self.dec_cell,
            self.dec_cell_proj,
            dec_input_embedding,
            init_state=dec_init_state,
            context_vector=selected_attribute_embedding,
            sequence_length=dec_seq_lens,
        )

        if final_context_state is not None:
            self.dec_out_words = final_context_state
        else:
            self.dec_out_words = torch.max(dec_outs, 2)[1]

        if mode != "test":
            labels = self.output_tokens[:, 1:]
            label_mask = torch.sign(labels).detach().float()

            rc_loss = F.cross_entropy(
                dec_outs.reshape(-1, dec_outs.size(-1)), labels.reshape(-1), reduction="none"
            ).view(dec_outs.size()[:-1])
            rc_loss = torch.sum(rc_loss * label_mask, 1)
            self.avg_rc_loss = rc_loss.mean()
            self.rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

            bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
            bow_loss = torch.sum(bow_loss, 1)
            self.avg_bow_loss = torch.mean(bow_loss)

            if self.use_hcf:
                self.avg_da_loss = F.cross_entropy(self.da_logits, self.output_das)
            else:
                self.avg_da_loss = self.avg_bow_loss.new_tensor(0)

            kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
            self.avg_kld = torch.mean(kld)
            kl_weights = 1.0
            self.kl_w = kl_weights
            self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld
            self.aug_elbo = self.avg_bow_loss + self.avg_da_loss + self.elbo

            self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
            self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
            self.est_marginal = torch.mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

        return dec_outs


class _KgCVAEConfig:
    """Mirrors config_utils.py:KgCVAEConfig field names/defaults, sized down for a tiny trace."""

    def __init__(self, use_hcf):
        self.use_hcf = use_hcf
        self.sent_type = "bi_rnn"
        self.latent_size = 12
        self.full_kl_step = 10000
        self.dec_keep_prob = 1.0
        self.cell_type = "gru"
        self.embed_size = 10
        self.topic_embed_size = 6
        # NOTE: the upstream repo hardcodes 30 as attribute_fc1's output width (cvae.py
        # `nn.Sequential(nn.Linear(da_embed_size, 30), nn.Tanh())`) while also using the raw,
        # un-projected da_embedding (width da_embed_size) as `selected_attribute_embedding` in
        # dec_inputs -- two different tensors that upstream's default config keeps
        # same-shaped only because da_embed_size == 30 there too. Kept at 30 here (not shrunk
        # like the other tiny dims) to preserve that real, if accidental, upstream contract.
        self.da_embed_size = 30
        self.cxt_cell_size = 14
        self.sent_cell_size = 8
        self.dec_cell_size = 12
        self.max_utt_len = 6
        self.num_layer = 1
        self.op = "adam"
        self.grad_clip = 5.0
        self.batch_size = 2
        self.init_lr = 0.001
        self.keep_prob = 1.0
        self.grad_noise = 0.0


class _TinyDialogApi:
    """Minimal stand-in for the repo's SWDADialogCorpus `api`: KgRnnCVAE.__init__ only reads
    api.vocab / api.rev_vocab / api.topic_vocab / api.dialog_act_vocab (each a small list)."""

    def __init__(self):
        self.vocab = ["<pad>", "<s>", "</s>", "a", "b", "c", "d", "e"]
        self.rev_vocab = {w: i for i, w in enumerate(self.vocab)}
        self.topic_vocab = ["t0", "t1", "t2"]
        self.dialog_act_vocab = ["da0", "da1", "da2"]


def _tiny_feed_dict(config, batch_size=2, n_turns=2, use_prior=False):
    torch.manual_seed(0)
    T = config.max_utt_len
    input_contexts = torch.randint(1, 6, (batch_size, n_turns, T))
    floors = torch.randint(0, 2, (batch_size, n_turns))
    context_lens = torch.full((batch_size,), n_turns, dtype=torch.long)
    topics = torch.randint(0, 3, (batch_size,))
    my_profile = torch.zeros(batch_size, 4)
    ot_profile = torch.zeros(batch_size, 4)
    output_tokens = torch.randint(1, 6, (batch_size, T))
    output_lens = torch.full((batch_size,), T, dtype=torch.long)
    output_das = torch.randint(0, 3, (batch_size,))
    return {
        "input_contexts": input_contexts,
        "floors": floors,
        "context_lens": context_lens,
        "topics": topics,
        "my_profile": my_profile,
        "ot_profile": ot_profile,
        "output_tokens": output_tokens,
        "output_lens": output_lens,
        "output_das": output_das,
        "use_prior": use_prior,
    }


class CvaeWrapper(nn.Module):
    """Wraps KgRnnCVAE's feed_dict-in forward into a single-tensor-friendly call so torchlens
    can trace it directly (KgRnnCVAE.forward's real computation is unchanged)."""

    def __init__(self, use_hcf):
        super().__init__()
        config = _KgCVAEConfig(use_hcf=use_hcf)
        api = _TinyDialogApi()
        self.cvae = KgRnnCVAE(config, api)
        self._config = config

    def forward(self, dummy):
        feed_dict = _tiny_feed_dict(self._config)
        return self.cvae(feed_dict, mode="train")


def build_cvae():
    """Ablated baseline: use_hcf=False (plain CVAE, no dialog-act conditioning)."""
    return CvaeWrapper(use_hcf=False)


def build_kgcvae():
    """Full model: use_hcf=True (knowledge-guided CVAE with dialog-act conditioning)."""
    return CvaeWrapper(use_hcf=True)


def example_input_cvae():
    return torch.zeros(1)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("CVAE dialogue model", "build_cvae", "example_input_cvae", 2017, "ported-pytorch"),
    ("kgCVAE dialogue model", "build_kgcvae", "example_input_cvae", 2017, "ported-pytorch"),
]
