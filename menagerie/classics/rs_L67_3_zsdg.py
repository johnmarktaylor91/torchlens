# SOURCE: vendored from snakeztc/NeuralDialog-ZSDG @ master
# Files: zsdg/models/models.py (ZeroShotHRED class), zsdg/models/model_bases.py,
#        zsdg/enc2dec/base_modules.py, zsdg/enc2dec/encoders.py, zsdg/enc2dec/decoders.py
#        (Attention + DecoderRNN only), zsdg/nn_lib.py, zsdg/criterions.py
# https://github.com/snakeztc/NeuralDialog-ZSDG
#
# Minimal changes from the original source (2018-era pre-1.0 PyTorch code,
# SIGDIAL 2018 best paper):
#   - `zsdg.dataset.corpora`'s PAD/BOS/EOS/BOD are plain string constants;
#     inlined here rather than importing the whole `corpora.py` module,
#     which requires nltk/pandas to build real vocabularies from the
#     SimDial/Stanford datasets (irrelevant to model construction/forward).
#   - `zsdg.utils`'s `Pack`, `INT`/`LONG`/`FLOAT`, `cast_type` are vendored
#     verbatim; the tokenizer helpers in that file (`get_tokenize`, nltk
#     import) are dropped since they are never called by the model classes.
#   - `DecoderPointerGen` (only used by `PtrHRED`/`ZeroShotPtrHRED`, not by
#     `ZeroShotHRED`) is omitted; `Attention` + `DecoderRNN` are vendored
#     verbatim.
#   - No PyTorch-API fixes were needed: `F.tanh`/`F.sigmoid` and
#     `Variable(..., volatile=True)` are still valid (deprecated no-ops) on
#     the installed torch, and this trace path (`mode=TEACH_FORCE`,
#     `inputs=dec_inputs` always provided) never reaches the
#     `bos_var`/`volatile` branch anyway.
#   - `BaseModel.np2var()` calls the real repo's `torch.from_numpy()` on
#     each data_feed field, so the real forward expects plain NumPy arrays.
#     TorchLens's tracer auto-coerces raw NumPy leaves in the traced input
#     tuple into tensors before forward() runs; the harness `ZSDGForward`
#     wrapper below converts back to NumPy at the boundary (`.numpy()`)
#     before calling into the vendored code -- a harness/trace-plumbing
#     adaptation, not an architecture change.
#   - Added build_zsdg_zero_shot_hred() / example_input_zsdg_zero_shot_hred()
#     harness constructing a tiny corpus/config stand-in (matching
#     model/configs.py's argparse defaults for utt_cell_size/dec_cell_size/
#     etc.) and driving the real ZeroShotHRED.forward() teacher-forcing path.
#
# Architecture (unmodified from source): ZSDG (Zero-Shot Dialogue Generation)
# is a hierarchical encoder-decoder with cross-domain latent actions. A
# shared RnnUttEncoder embeds utterances; a context RNN (EncoderRNN) folds
# utterance embeddings across turns; a domain-agnostic "policy" network maps
# context state to a latent action vector shared across domains (the
# "zero-shot" cross-domain transfer mechanism); an attention DecoderRNN
# (LSTM/GRU with 'cat' attention over encoder outputs) generates the
# response conditioned on the latent action via a LinearConnector. Trained
# with Action Matching: an L2 distance between the response's own latent
# encoding (`out_embedded`) and the context-derived latent action.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


# ---------------------------------------------------------------------------
# zsdg/dataset/corpora.py (special-token string constants only)
# ---------------------------------------------------------------------------

PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"
BOD = "<d>"


# ---------------------------------------------------------------------------
# zsdg/utils.py (subset actually used by the model classes; verbatim)
# ---------------------------------------------------------------------------

INT = 0
LONG = 1
FLOAT = 2


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


# ---------------------------------------------------------------------------
# zsdg/criterions.py (verbatim)
# ---------------------------------------------------------------------------


class L2Loss(_Loss):
    def forward(self, state_a, state_b):
        if type(state_a) is tuple:
            losses = 0.0
            for s_a, s_b in zip(state_a, state_b):
                losses += torch.pow(s_a - s_b, 2)
        else:
            losses = torch.pow(state_a - state_b, 2)
        return torch.mean(losses)


class NLLEntropy(_Loss):
    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = config.avg_type

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT, config.use_gpu)

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        if self.avg_type is None:
            loss = F.nll_loss(
                input, target, size_average=False, ignore_index=self.padding_idx, weight=self.weight
            )
        elif self.avg_type == "seq":
            loss = F.nll_loss(
                input, target, size_average=False, ignore_index=self.padding_idx, weight=self.weight
            )
            loss = loss / batch_size
        elif self.avg_type == "real_word":
            loss = F.nll_loss(
                input,
                target,
                size_average=True,
                ignore_index=self.padding_idx,
                weight=self.weight,
                reduce=False,
            )
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == "word":
            loss = F.nll_loss(
                input, target, size_average=True, ignore_index=self.padding_idx, weight=self.weight
            )
        else:
            raise ValueError("Unknown avg type")

        return loss


# ---------------------------------------------------------------------------
# zsdg/enc2dec/base_modules.py (verbatim)
# ---------------------------------------------------------------------------


class BaseRNN(nn.Module):
    SYM_MASK = PAD
    SYM_EOS = EOS

    KEY_ATTN_SCORE = "attention_score"
    KEY_LENGTH = "length"
    KEY_SEQUENCE = "sequence"
    KEY_LATENT = "latent"
    KEY_RECOG_LATENT = "recog_latent"
    KEY_POLICY = "policy"
    KEY_G = "g"
    KEY_PTR_SOFTMAX = "ptr_softmax"
    KEY_PTR_CTX = "ptr_context"

    def __init__(
        self,
        vocab_size,
        input_size,
        hidden_size,
        input_dropout_p,
        dropout_p,
        n_layers,
        rnn_cell,
        bidirectional,
    ):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == "lstm":
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )
        if rnn_cell.lower() == "lstm":
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

    def gumbel_max(self, log_probs):
        """
        Obtain a sample from the Gumbel max. Not this is not differentibale.
        :param log_probs: [batch_size x vocab_size]
        :return: [batch_size x 1] selected token IDs
        """
        sample = torch.Tensor(log_probs.size()).uniform_(0, 1)
        sample = cast_type(Variable(sample), FLOAT, self.use_gpu)

        # compute the gumbel sample
        matrix_u = -1.0 * torch.log(-1.0 * torch.log(sample))
        gumbel_log_probs = log_probs + matrix_u
        max_val, max_ids = torch.max(gumbel_log_probs, dim=-1, keepdim=True)
        return max_ids

    def repeat_state(self, state, batch_size, times):
        new_s = state.repeat(1, 1, times)
        return new_s.view(-1, batch_size * times, self.hidden_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# zsdg/enc2dec/encoders.py (verbatim)
# ---------------------------------------------------------------------------


class EncoderRNN(BaseRNN):
    def __init__(
        self,
        input_size,
        hidden_size,
        input_dropout_p=0,
        dropout_p=0,
        n_layers=1,
        rnn_cell="gru",
        variable_lengths=False,
        bidirection=False,
    ):
        super(EncoderRNN, self).__init__(
            -1, input_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell, bidirection
        )

        self.variable_lengths = variable_lengths
        self.output_size = hidden_size * 2 if bidirection else hidden_size

    def forward(self, input_var, input_lengths=None, init_state=None):
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
        utt_cell_size,
        dropout,
        rnn_cell="gru",
        bidirection=True,
        use_attn=False,
        embedding=None,
        vocab_size=None,
        embed_dim=None,
        feat_size=0,
    ):
        super(RnnUttEncoder, self).__init__()
        self.bidirection = bidirection
        self.utt_cell_size = utt_cell_size

        if embedding is None:
            self.embed_size = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = embedding
            self.embed_size = embedding.embedding_dim

        self.rnn = EncoderRNN(
            self.embed_size + feat_size,
            utt_cell_size,
            0.0,
            dropout,
            rnn_cell=rnn_cell,
            variable_lengths=False,
            bidirection=bidirection,
        )

        self.multipler = 2 if bidirection else 1
        self.output_size = self.utt_cell_size * self.multipler
        self.use_attn = use_attn
        self.feat_size = feat_size
        if use_attn:
            self.key_w = nn.Linear(self.utt_cell_size * self.multipler, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, return_all=False):
        batch_size = int(utterances.size()[0])
        max_ctx_lens = int(utterances.size()[1])
        max_utt_len = int(utterances.size()[2])

        # repeat the init state
        if init_state is not None:
            init_state = init_state.repeat(1, max_ctx_lens, 1)

        # get word embeddings
        flat_words = utterances.view(-1, max_utt_len)
        words_embeded = self.embedding(flat_words)

        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            words_embeded = torch.cat([words_embeded, flat_feats], dim=2)

        enc_outs, enc_last = self.rnn(words_embeded, init_state=init_state)

        if self.use_attn:
            fc1 = F.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim() - 1).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = torch.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_lens, self.output_size)

        if return_all:
            return utt_embedded, enc_outs, enc_last, attn
        else:
            return utt_embedded


# ---------------------------------------------------------------------------
# zsdg/enc2dec/decoders.py (Attention + DecoderRNN only; verbatim)
# ---------------------------------------------------------------------------

TEACH_FORCE = "teacher_forcing"
TEACH_GEN = "teacher_gen"
GEN = "gen"


class Attention(nn.Module):
    def __init__(self, dec_size, attn_size, mode, project=False):
        super(Attention, self).__init__()
        self.mask = None
        self.mode = mode
        self.attn_size = attn_size
        self.dec_size = dec_size

        if project:
            self.linear_out = nn.Linear(dec_size + attn_size, dec_size)
        else:
            self.linear_out = None

        if mode == "general":
            self.attn_w = nn.Linear(dec_size, attn_size)
        elif mode == "cat":
            self.dec_w = nn.Linear(dec_size, dec_size)
            self.attn_w = nn.Linear(attn_size, dec_size)
            self.query_w = nn.Linear(dec_size, 1)

    def forward(self, output, context):
        """
        :param output: [batch, out_len, dec_size]
        :param context: [batch, in_len, attn_size]
        :return: output, attn
        """
        batch_size = output.size(0)
        input_size = context.size(1)

        # batch, out_len, in_len
        if self.mode == "dot":
            attn = torch.bmm(output, context.transpose(1, 2))
        elif self.mode == "general":
            mapped_output = self.attn_w(output)
            attn = torch.bmm(mapped_output, context.transpose(1, 2))
        elif self.mode == "cat":
            mapped_attn = self.attn_w(context)
            mapped_out = self.dec_w(output)
            tiled_out = mapped_out.unsqueeze(2).repeat(1, 1, input_size, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = F.tanh(tiled_attn + tiled_out)
            attn = self.query_w(fc1).squeeze(-1)
        else:
            raise ValueError("Unknown attention")

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float("inf"))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim)
        #  -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        if self.linear_out is None:
            return combined, attn
        else:
            # output -> (batch, out_len, dim)
            output = F.tanh(
                self.linear_out(combined.view(-1, self.dec_size + self.attn_size))
            ).view(batch_size, -1, self.dec_size)
            return output, attn


class DecoderRNN(BaseRNN):
    def __init__(
        self,
        vocab_size,
        max_len,
        input_size,
        hidden_size,
        sos_id,
        eos_id,
        n_layers=1,
        rnn_cell="lstm",
        input_dropout_p=0,
        dropout_p=0,
        use_attention=False,
        attn_mode="cat",
        attn_size=None,
        use_gpu=True,
        embedding=None,
        output_size=None,
        tie_output_embed=False,
    ):
        super(DecoderRNN, self).__init__(
            vocab_size,
            input_size,
            hidden_size,
            input_dropout_p,
            dropout_p,
            n_layers,
            rnn_cell,
            False,
        )

        self.output_size = vocab_size if output_size is None else output_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.use_gpu = use_gpu

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, self.input_size)
        else:
            self.embedding = embedding

        if use_attention:
            self.attention = Attention(self.hidden_size, attn_size, attn_mode, project=True)

        if tie_output_embed:
            self.project = lambda x: x * self.embedding.weight.transpose(0, 1)
        else:
            self.project = nn.Linear(self.hidden_size, self.output_size)
        self.function = F.log_softmax

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = output.contiguous()
        logits = self.project(output.view(-1, self.hidden_size))
        predicted_softmax = self.function(logits, dim=logits.dim() - 1).view(
            batch_size, output_size, -1
        )
        return predicted_softmax, hidden, attn

    def forward(
        self,
        batch_size,
        inputs=None,
        init_state=None,
        attn_context=None,
        mode=TEACH_FORCE,
        gen_type="greedy",
        beam_size=4,
    ):
        # sanity checks
        ret_dict = dict()

        if self.use_attention:
            # calculate initial attention
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if mode == GEN:
            inputs = None

        if gen_type != "beam":
            beam_size = 1

        if inputs is not None:
            decoder_input = inputs
        else:
            # prepare the BOS inputs
            bos_var = Variable(torch.LongTensor([self.sos_id]), volatile=True)
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size * beam_size, 1)

        if mode == GEN and gen_type == "beam":
            # if beam search, repeat the initial states of the RNN
            if self.rnn_cell is nn.LSTM:
                h, c = init_state
                decoder_hidden = (
                    self.repeat_state(h, batch_size, beam_size),
                    self.repeat_state(c, batch_size, beam_size),
                )
            else:
                decoder_hidden = self.repeat_state(init_state, batch_size, beam_size)
        else:
            decoder_hidden = init_state

        decoder_outputs = []  # a list of logprob
        sequence_symbols = []  # a list word ids
        back_pointers = []  # a list of parent beam ID
        lengths = np.array([self.max_length] * batch_size * beam_size)

        def decode(step, cum_sum, step_output, step_attn):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)

            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if gen_type == "greedy":
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == "sample":
                symbols = self.gumbel_max(step_output_slice)
            elif gen_type == "beam":
                if step == 0:
                    seq_score = step_output_slice.view(batch_size, -1)
                    seq_score = seq_score[:, 0 : self.output_size]
                else:
                    seq_score = cum_sum + step_output_slice
                    seq_score = seq_score.view(batch_size, -1)

                top_v, top_id = seq_score.topk(beam_size)

                back_ptr = top_id.div(self.output_size).view(-1, 1)
                symbols = top_id.fmod(self.output_size).view(-1, 1)
                cum_sum = top_v.view(-1, 1)
                back_pointers.append(back_ptr)
            else:
                raise ValueError("Unsupported decoding mode")

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return cum_sum, symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph
        if mode == TEACH_FORCE:
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, attn_context
            )

            # in teach forcing mode, we don't need symbols.
            decoder_outputs = decoder_output

        else:
            # do free running here
            cum_sum = None
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(
                    decoder_input, decoder_hidden, attn_context
                )

                cum_sum, symbols = decode(di, cum_sum, decoder_output, step_attn)
                decoder_input = symbols

            decoder_outputs = torch.cat(decoder_outputs, dim=1)

            if gen_type == "beam":
                # do back tracking here to recover the 1-best according to
                # beam search.
                final_seq_symbols = []
                cum_sum = cum_sum.view(-1, beam_size)
                max_seq_id = cum_sum.topk(1)[1].data.cpu().view(-1).numpy()
                rev_seq_symbols = sequence_symbols[::-1]
                rev_back_ptrs = back_pointers[::-1]

                for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
                    symbol2ds = symbols.view(-1, beam_size)
                    back2ds = back_ptrs.view(-1, beam_size)

                    selected_symbols = []
                    selected_parents = []
                    for b_id in range(batch_size):
                        selected_parents.append(back2ds[b_id, max_seq_id[b_id]])
                        selected_symbols.append(symbol2ds[b_id, max_seq_id[b_id]])

                    final_seq_symbols.append(torch.cat(selected_symbols).unsqueeze(1))
                    max_seq_id = torch.cat(selected_parents).data.cpu().numpy()
                sequence_symbols = final_seq_symbols[::-1]

        # save the decoded sequence symbols and sequence length
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict


# ---------------------------------------------------------------------------
# zsdg/nn_lib.py (LinearConnector only, used by ZeroShotHRED; verbatim)
# ---------------------------------------------------------------------------


class LinearConnector(nn.Module):
    def __init__(self, input_size, output_size, is_lstm, has_bias=True):
        super(LinearConnector, self).__init__()
        if is_lstm:
            self.linear_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.linear_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h = self.linear_h(inputs).unsqueeze(0)
            c = self.linear_c(inputs).unsqueeze(0)
            return (h, c)
        else:
            return self.linear(inputs).unsqueeze(0)

    def get_w(self):
        if self.is_lstm:
            return self.linear_h.weight
        else:
            return self.linear.weight


class Hidden2Feat(nn.Module):
    def __init__(self, input_size, output_size, is_lstm, has_bias=True):
        super(Hidden2Feat, self).__init__()
        if is_lstm:
            self.linear_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.linear_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h = self.linear_h(inputs[0].squeeze(0))
            c = self.linear_c(inputs[1].squeeze(0))
            return h + c
        else:
            return self.linear(inputs.squeeze(0))


# ---------------------------------------------------------------------------
# zsdg/models/model_bases.py (BaseModel; only the members ZeroShotHRED
# actually uses -- np2var, _remove_padding; verbatim)
# ---------------------------------------------------------------------------


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(torch.from_numpy(inputs)), dtype, self.use_gpu)

    def forward(self, *input):
        raise NotImplementedError

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for key, l in loss.items():  # noqa: E741 (kept for parity with original repo)
            if l is not None:
                total_loss += l
        return total_loss

    def _gather_last_out(self, rnn_outs, lens):
        time_dimension = 1
        len_vars = self.np2var(np.array(lens), LONG)
        len_vars = len_vars.view(-1, 1).expand(len(lens), rnn_outs.size(2)).unsqueeze(1)
        slices = rnn_outs.gather(time_dimension, len_vars - 1)
        return slices.squeeze(time_dimension)

    def _remove_padding(self, feats, words):
        """ "
        :param feats: batch_size x num_words x feats
        :param words: batch_size x num_words
        :return: the same input without padding
        """
        if feats is None:
            return None, None

        batch_size = words.size(0)
        valid_mask = torch.sign(words).float()
        batch_lens = torch.sum(valid_mask, dim=1)
        max_word_num = torch.max(batch_lens)
        padded_lens = (max_word_num - batch_lens).cpu().data.numpy()
        valid_words = []
        valid_feats = []

        for b_id in range(batch_size):
            valid_idxs = valid_mask[b_id].nonzero().view(-1)
            valid_row_words = torch.index_select(words[b_id], 0, valid_idxs)
            valid_row_feat = torch.index_select(feats[b_id], 0, valid_idxs)

            padded_len = int(padded_lens[b_id])
            valid_row_words = F.pad(valid_row_words, (0, padded_len))
            valid_row_feat = F.pad(valid_row_feat, (0, 0, 0, padded_len))

            valid_words.append(valid_row_words.unsqueeze(0))
            valid_feats.append(valid_row_feat.unsqueeze(0))

        feats = torch.cat(valid_feats, dim=0)
        words = torch.cat(valid_words, dim=0)
        return feats, words


# ---------------------------------------------------------------------------
# zsdg/models/models.py -- PtrBase.compute_loss + ZeroShotHRED (verbatim)
# ---------------------------------------------------------------------------


class PtrBase(BaseModel):
    def compute_loss(self, dec_outs, dec_ctx, labels):
        rnn_loss = self.nll_loss(dec_outs, labels)
        g = dec_ctx.get("g")
        if g is not None:
            ptr_softmax = dec_ctx["ptr_softmax"]
            flat_ptr = ptr_softmax.view(-1, self.vocab_size)
            label_mask = labels.view(-1, 1) == self.rev_vocab[PAD]
            label_ptr = flat_ptr.gather(1, labels.view(-1, 1))
            not_in_ctx = label_ptr == 0
            mix_ptr = torch.cat([label_ptr, g.view(-1, 1)], dim=1).gather(1, not_in_ctx.long())
            attention_loss = -1.0 * torch.log(mix_ptr.clamp(min=1e-10))
            attention_loss.masked_fill_(label_mask, 0)

            valid_cnt = (label_mask.size(0) - torch.sum(label_mask).float()).clamp(min=1e-10)
            avg_attn_loss = torch.sum(attention_loss) / valid_cnt
        else:
            avg_attn_loss = None

        return Pack(nll=rnn_loss, attn_loss=avg_attn_loss)


class ZeroShotHRED(PtrBase):
    def __init__(self, corpus, config):
        super(ZeroShotHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)

        self.utt_encoder = RnnUttEncoder(
            config.utt_cell_size,
            config.dropout,
            use_attn=config.utt_type == "rnn_attn",
            vocab_size=self.vocab_size,
            embedding=self.embedding,
            feat_size=1,
        )

        self.ctx_encoder = EncoderRNN(
            self.utt_encoder.output_size,
            config.ctx_cell_size,
            0.0,
            config.dropout,
            config.num_layer,
            config.rnn_cell,
            variable_lengths=False,
            bidirection=config.bi_ctx_cell,
        )

        self.policy = Hidden2Feat(
            self.ctx_encoder.output_size, config.dec_cell_size, is_lstm=config.rnn_cell == "lstm"
        )
        self.utt_policy = lambda x: x

        self.connector = LinearConnector(
            config.dec_cell_size, config.dec_cell_size, is_lstm=config.rnn_cell == "lstm"
        )

        self.attn_size = self.ctx_encoder.output_size

        self.decoder = DecoderRNN(
            self.vocab_size,
            config.max_dec_len,
            config.embed_size,
            config.dec_cell_size,
            self.go_id,
            self.eos_id,
            n_layers=1,
            rnn_cell=config.rnn_cell,
            input_dropout_p=config.dropout,
            dropout_p=config.dropout,
            use_attention=config.use_attn,
            attn_size=self.ctx_encoder.output_size,
            attn_mode=config.attn_type,
            use_gpu=config.use_gpu,
        )

        self.nll_loss = NLLEntropy(self.pad_id, config)
        self.l2_loss = L2Loss()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.distance + loss.nll
        return total_loss

    def forward(self, data_feed, mode, gen_type="greedy", return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        # optional fields
        ctx_lens = data_feed.get("context_lens")
        ctx_utts = self.np2var(data_feed.get("contexts"), LONG)
        ctx_confs = self.np2var(data_feed.get("context_confs"), FLOAT)
        out_acts = self.np2var(data_feed.get("output_actions"), LONG)

        # required fields
        out_utts = self.np2var(data_feed["outputs"], LONG)
        batch_size = len(data_feed["outputs"])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)

        # forward pass
        out_embedded, out_outs, _, _ = self.utt_encoder(
            out_utts.unsqueeze(1), out_confs, return_all=True
        )
        out_embedded = self.utt_policy(out_embedded.squeeze(1))

        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(
                out_acts.unsqueeze(1), out_confs, return_all=True
            )
            act_embedded = act_embedded.squeeze(1)

            # create attention contexts
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)

            # create decoder initial states
            latent_action = self.policy(ctx_last)

            # create attention contexts
            ctx_outs = (
                ctx_outs.unsqueeze(2)
                .repeat(1, 1, ctx_utts.size(2), 1)
                .view(batch_size, -1, self.ctx_encoder.output_size)
            )
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_inputs = ctx_outs + utt_outs  # batch_size x num_word x attn_size
            attn_words = ctx_utts.view(batch_size, -1)  # batch_size x num_words

        dec_init_state = self.connector(latent_action)

        # mask out PAD words in the attention inputs
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(
            batch_size,
            dec_inputs,
            dec_init_state,
            attn_context=attn_inputs,
            mode=mode,
            gen_type=gen_type,
            beam_size=self.config.beam_size,
        )
        if mode == GEN:
            return dec_ctx, labels
        else:
            rnn_loss = self.nll_loss(dec_outs, labels)
            loss_pack = Pack(nll=rnn_loss)
            if return_latent:
                loss_pack["latent_actions"] = latent_action

            loss_pack["distance"] = self.l2_loss(out_embedded, latent_action)
            return loss_pack


# ---------------------------------------------------------------------------
# Menagerie harness
# ---------------------------------------------------------------------------


class ZSDGConfig:
    """Minimal stand-in for the argparse Config built by simdial-zsdg.py /
    stanford-zsdg.py, with only the attributes ZeroShotHRED reads. Values
    mirror those scripts' real `--utt_cell_size`/`--ctx_cell_size`/
    `--dec_cell_size`/`--embed_size`/`--rnn_cell`/`--attn_type`/`--avg_type`
    defaults, shrunk to menagerie-tiny sizes."""

    def __init__(self):
        self.embed_size = 16
        self.utt_cell_size = 16
        # RnnUttEncoder defaults bidirection=True (see ZeroShotHRED.__init__
        # below, which never overrides it), so utt_encoder.output_size =
        # utt_cell_size * 2. attn_inputs = ctx_outs + utt_outs requires
        # ctx_encoder.output_size to match, i.e. ctx_cell_size = 2 *
        # utt_cell_size when bi_ctx_cell=False -- same ratio as the real
        # repo's argparse defaults (utt_cell_size=256, ctx_cell_size=512).
        self.ctx_cell_size = 32
        self.dec_cell_size = 32
        self.bi_ctx_cell = False
        self.max_dec_len = 8
        self.num_layer = 1
        self.use_attn = True
        self.attn_type = "cat"
        self.rnn_cell = "lstm"
        self.dropout = 0.0
        self.use_gpu = False
        self.avg_type = "word"
        self.beam_size = 1
        self.utt_type = "rnn"


class ZSDGCorpus:
    """Minimal stand-in for the real Corpus object (built from a
    pickled/tokenized dataset in the original repo); ZeroShotHRED only
    reads `.vocab` / `.rev_vocab`."""

    def __init__(self, vocab_size=48):
        self.vocab = [f"tok{i}" for i in range(vocab_size)]
        self.vocab[0], self.vocab[1], self.vocab[2] = PAD, BOS, EOS
        self.rev_vocab = {tok: idx for idx, tok in enumerate(self.vocab)}


class ZSDGForward(nn.Module):
    """Wraps ZeroShotHRED to return only the decoder log-prob tensor (the
    NLL/L2 loss terms are real training-loss scalars, not activations)."""

    def __init__(self, config, corpus):
        super().__init__()
        self.model = ZeroShotHRED(corpus, config)

    def forward(self, contexts, context_confs, context_lens, outputs):
        # ZeroShotHRED.np2var() calls the real repo's torch.from_numpy() on
        # each field (see model_bases.BaseModel.np2var above), so it expects
        # NumPy arrays. TorchLens auto-coerces raw NumPy leaves found in the
        # traced input tuple into tensors before forward() runs; converting
        # back here is a harness-only adaptation (mirrors what any caller
        # holding tensor batches would do before invoking the real numpy-
        # typed API), not an architecture change.
        data_feed = {
            "contexts": contexts.numpy(),
            "context_confs": context_confs.numpy(),
            "context_lens": context_lens,
            "outputs": outputs.numpy(),
        }
        dec_outs, dec_last, dec_ctx = self._forward_and_capture(data_feed)
        return dec_outs

    def _forward_and_capture(self, data_feed):
        # Re-implements ZeroShotHRED.forward's ctx-branch inline so we can
        # surface the raw decoder tensor (mode=TEACH_FORCE path only).
        model = self.model
        ctx_lens = data_feed["context_lens"]
        ctx_utts = model.np2var(data_feed["contexts"], LONG)
        ctx_confs = model.np2var(data_feed["context_confs"], FLOAT)
        out_utts = model.np2var(data_feed["outputs"], LONG)
        batch_size = len(data_feed["outputs"])
        out_confs = model.np2var(np.ones((batch_size, 1)), FLOAT)

        out_embedded, out_outs, _, _ = model.utt_encoder(
            out_utts.unsqueeze(1), out_confs, return_all=True
        )

        utt_embedded, utt_outs, _, _ = model.utt_encoder(ctx_utts, ctx_confs, return_all=True)
        ctx_outs, ctx_last = model.ctx_encoder(utt_embedded, ctx_lens)
        latent_action = model.policy(ctx_last)

        ctx_outs = (
            ctx_outs.unsqueeze(2)
            .repeat(1, 1, ctx_utts.size(2), 1)
            .view(batch_size, -1, model.ctx_encoder.output_size)
        )
        utt_outs = utt_outs.contiguous().view(batch_size, -1, model.utt_encoder.output_size)
        attn_inputs = ctx_outs + utt_outs
        attn_words = ctx_utts.view(batch_size, -1)

        dec_init_state = model.connector(latent_action)
        attn_inputs, attn_words = model._remove_padding(attn_inputs, attn_words)

        dec_inputs = out_utts[:, 0:-1]

        dec_outs, dec_last, dec_ctx = model.decoder(
            batch_size,
            dec_inputs,
            dec_init_state,
            attn_context=attn_inputs,
            mode=TEACH_FORCE,
            gen_type="greedy",
            beam_size=model.config.beam_size,
        )
        return dec_outs, dec_last, dec_ctx


def build_zsdg_zero_shot_hred():
    config = ZSDGConfig()
    corpus = ZSDGCorpus(vocab_size=48)
    return ZSDGForward(config, corpus)


def example_input_zsdg_zero_shot_hred():
    batch_size = 2
    n_ctx_turns = 3
    utt_len = 6
    out_len = 7
    vocab_size = 48

    contexts = torch.randint(4, vocab_size, (batch_size, n_ctx_turns, utt_len)).long()
    context_confs = torch.ones((batch_size, n_ctx_turns), dtype=torch.float32)
    context_lens = [n_ctx_turns] * batch_size
    outputs = torch.randint(4, vocab_size, (batch_size, out_len)).long()

    return (contexts, context_confs, context_lens, outputs)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "ZSDG (ZeroShotHRED)",
        "build_zsdg_zero_shot_hred",
        "example_input_zsdg_zero_shot_hred",
        2018,
        "vendored",
    ),
]
