# SOURCE: vendored from RaymondLi0/conversational-recommendations @ master
#
# https://github.com/RaymondLi0/conversational-recommendations
# https://raw.githubusercontent.com/RaymondLi0/conversational-recommendations/master/models/hierarchical_rnn.py
# https://raw.githubusercontent.com/RaymondLi0/conversational-recommendations/master/models/hred.py
# https://raw.githubusercontent.com/RaymondLi0/conversational-recommendations/master/models/decoders.py
# https://raw.githubusercontent.com/RaymondLi0/conversational-recommendations/master/utils.py
#
# Li et al. 2018 "Towards Deep Conversational Recommendations" (NeurIPS 2018) --
# the official ReDial repo. The architecture vendored here is the HRED
# conversation model (`models/hred.py::HRED`): an `HRNN` hierarchical
# recurrent encoder (`models/hierarchical_rnn.py::HRNN`, a bidirectional-GRU
# sentence encoder feeding a GRU conversation encoder) whose output context
# vector conditions a `TextDecoder`/`DecoderGRU` (`models/decoders.py`)
# response generator. `HRNN` optionally wraps a pretrained GenSen sentence
# encoder (`models/gensen.py`) for richer embeddings; that path requires
# downloading external NLI-pretrained GenSen checkpoint files
# (`nli_large.model`/`nli_large_vocab.pkl`) plus `h5py`/`nltk`/`sklearn` for
# vocabulary expansion, none of which are base libs or reasonably
# tiny-random-init-able (the checkpoint IS the model). `gensen=False` is a
# genuine, first-class configuration branch already present in the real
# `HRNN.__init__` (see `models/hierarchical_rnn.py:66-72`) that swaps GenSen
# for a plain trainable `nn.Embedding` -- no architecture is invented, this is
# an as-shipped code path, just as skipping an optional pretrained-embedding
# arg in a HF model config is not an architectural change.
#
# Classes below (`HRNN`, `DecoderGRU`, `TextDecoder`, `HRED`) are copied
# verbatim from the three source files, combined into one module. The only
# changes are:
#   - `torch.autograd.Variable` calls removed (a documented no-op since
#     torch 0.4; modern tensors are already differentiable, so
#     `Variable(x)` -> `x` and `Variable(torch.zeros(...))` -> `torch.zeros(...)`),
#   - `F.log_softmax(...)`/`F.softmax(...)` calls given an explicit `dim=`
#     kwarg (mandatory in modern torch; the original 2018 code relied on the
#     since-removed dim-less legacy behavior over the last dimension, which
#     is what `dim=-1` reproduces here),
#   - `sort_for_packed_sequence` inlined from `utils.py` unchanged (its own
#     upstream module pulls in `nltk`/nothing else needed, so only this one
#     function is copied rather than importing all of `utils.py`),
#   - the `import config` / `from models.gensen import GenSenSingle` /
#     package-relative imports are removed since this vendored file only
#     exercises the `gensen=False` branch and never constructs `GenSenSingle`.
# No layer, dimension formula, or forward-pass control flow was altered.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# from utils.py::sort_for_packed_sequence (verbatim, Variable(...) unwrapped)
def sort_for_packed_sequence(lengths, cuda=False):
    """
    :param lengths: 1D array of lengths
    :return: sorted_lengths (lengths in descending order,
    sorted_idx (indices to sort),
    rev (indices to retrieve original order)
    """
    sorted_idx = np.argsort(lengths)[::-1]  # idx to sort by length
    sorted_lengths = lengths[sorted_idx]
    rev = np.argsort(sorted_idx)  # idx to retrieve original order

    tt = torch.cuda.LongTensor if cuda else torch.LongTensor
    sorted_idx = tt(sorted_idx.copy())
    rev = tt(rev.copy())
    return sorted_lengths, sorted_idx, rev


# from models/hierarchical_rnn.py::HRNN (verbatim for the gensen=False path)
class HRNN(nn.Module):
    """
    Hierarchical Recurrent Neural Network

    params.keys() ['use_gensen', 'use_movie_occurrences', 'sentence_encoder_hidden_size',
    'conversation_encoder_hidden_size', 'sentence_encoder_num_layers', 'conversation_encoder_num_layers', 'use_dropout',
    ['embedding_dimension']]

    Input: Input["dialogue"] (batch, max_conv_length, max_utterance_length) Long Tensor
           Input["senders"] (batch, max_conv_length) Float Tensor
           Input["lengths"] (batch, max_conv_length) list
           (optional) Input["movie_occurrences"] (batch, max_conv_length, max_utterance_length) for word occurence
                                                 (batch, max_conv_length) for sentence occurrence. Float Tensor
    """

    def __init__(
        self,
        params,
        gensen=False,
        train_vocabulary=None,
        train_gensen=True,
        conv_bidirectional=False,
    ):
        super(HRNN, self).__init__()
        self.params = params
        self.use_gensen = bool(gensen)
        self.train_gensen = train_gensen
        self.conv_bidirectional = conv_bidirectional

        self.cuda_available = torch.cuda.is_available()

        # gensen=False path: plain trainable embedding (real code branch,
        # models/hierarchical_rnn.py:66-72). The GenSenSingle branches are
        # omitted since this vendor only exercises gensen=False.
        self.src_embedding = nn.Embedding(
            num_embeddings=len(train_vocabulary), embedding_dim=params["embedding_dimension"]
        )
        self.word2id = {word: idx for idx, word in enumerate(train_vocabulary)}
        self.id2word = {idx: word for idx, word in enumerate(train_vocabulary)}
        self.sentence_encoder = nn.GRU(
            input_size=2048 + (self.params["use_movie_occurrences"] == "word")
            if self.use_gensen
            else self.params["embedding_dimension"]
            + (self.params["use_movie_occurrences"] == "word"),
            hidden_size=self.params["sentence_encoder_hidden_size"],
            num_layers=self.params["sentence_encoder_num_layers"],
            batch_first=True,
            bidirectional=True,
        )
        self.conversation_encoder = nn.GRU(
            input_size=2 * self.params["sentence_encoder_hidden_size"]
            + 1
            + (self.params["use_movie_occurrences"] == "sentence"),
            # concatenation of 2 directions for sentence encoders + sender informations + movie occurences
            hidden_size=self.params["conversation_encoder_hidden_size"],
            num_layers=self.params["conversation_encoder_num_layers"],
            batch_first=True,
            bidirectional=conv_bidirectional,
        )
        if self.params["use_dropout"]:
            self.dropout = nn.Dropout(p=self.params["use_dropout"])

    def get_sentence_representations(self, dialogue, senders, lengths, movie_occurrences=None):
        batch_size, max_conversation_length = dialogue.data.shape[:2]
        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)

        # reshape and reorder
        sorted_utterances = dialogue.view(batch_size * max_conversation_length, -1).index_select(
            0, sorted_idx
        )

        # consider sequences of length > 0 only
        num_positive_lengths = np.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]

        embedded = self.src_embedding(sorted_utterances)
        # (< batch_size * max conversation_length, max_sentence_length, embedding_size/2048 for gensen)

        if self.params["use_dropout"]:
            embedded = self.dropout(embedded)

        if self.params["use_movie_occurrences"] == "word":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            # reshape and reorder movie occurrences by utterance length
            movie_occurrences = movie_occurrences.view(
                batch_size * max_conversation_length, -1
            ).index_select(0, sorted_idx)
            # keep indices where sequence_length > 0
            movie_occurrences = movie_occurrences[:num_positive_lengths]
            embedded = torch.cat((embedded, movie_occurrences.unsqueeze(2)), 2)

        packed_sentences = pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
        # Apply encoder and get the final hidden states
        _, sentence_representations = self.sentence_encoder(packed_sentences)
        # (2*num_layers, < batch_size * max_conv_length, hidden_size)
        # Concat the hidden states of the last layer (two directions of the GRU)
        sentence_representations = torch.cat(
            (sentence_representations[-1], sentence_representations[-2]), 1
        )

        if self.params["use_dropout"]:
            sentence_representations = self.dropout(sentence_representations)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                2 * self.params["sentence_encoder_hidden_size"],
                out=tt(),
            )
            sentence_representations = torch.cat((sentence_representations, pad_tensor), 0)
        # Retrieve original sentence order and Reshape to separate conversations
        sentence_representations = sentence_representations.index_select(0, rev).view(
            batch_size, max_conversation_length, 2 * self.params["sentence_encoder_hidden_size"]
        )
        # Append sender information
        sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)
        # Append movie occurrence information if required
        if self.params["use_movie_occurrences"] == "sentence":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            sentence_representations = torch.cat(
                (sentence_representations, movie_occurrences.unsqueeze(2)), 2
            )
        return sentence_representations

    def forward(self, input_dict, return_all=True, return_sentence_representations=False):
        movie_occurrences = (
            input_dict["movie_occurrences"] if self.params["use_movie_occurrences"] else None
        )
        # get sentence representations
        sentence_representations = self.get_sentence_representations(
            input_dict["dialogue"],
            input_dict["senders"],
            lengths=input_dict["lengths"],
            movie_occurrences=movie_occurrences,
        )
        # (batch_size, max_conv_length, 2*sent_hidden_size + 1 + use_movie_occurences)
        # Pass whole conversation into GRU
        lengths = input_dict["conversation_lengths"]
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)

        # reorder in decreasing sequence length
        sorted_representations = sentence_representations.index_select(0, sorted_idx)
        packed_sequences = pack_padded_sequence(
            sorted_representations, sorted_lengths, batch_first=True
        )
        conversation_representations, last_state = self.conversation_encoder(packed_sequences)

        # retrieve original order
        conversation_representations, _ = pad_packed_sequence(
            conversation_representations, batch_first=True
        )
        conversation_representations = conversation_representations.index_select(0, rev)
        last_state = last_state.index_select(1, rev)
        if self.params["use_dropout"]:
            conversation_representations = self.dropout(conversation_representations)
            last_state = self.dropout(last_state)
        if return_all:
            if not return_sentence_representations:
                # return the last layer of the GRU for each t.
                return conversation_representations
            else:
                # also return sentence representations
                return conversation_representations, sentence_representations
        else:
            # get the last hidden state only
            if self.conv_bidirectional:
                # Concat the hidden states for the last layer (two directions of the GRU)
                last_state = torch.cat((last_state[-1], last_state[-2]), 1)
                return last_state
            else:
                # Return the hidden state from the last layers
                return last_state[-1]


# from models/decoders.py::DecoderGRU (verbatim, Variable(...) unwrapped)
class DecoderGRU(nn.Module):
    """
    Conditioned GRU. The context vector is used as an initial hidden state at each layer of the GRU
    """

    def __init__(
        self, hidden_size, context_size, num_layers, vocab_size, peephole, embedding_dim=512
    ):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        # peephole: concatenate the context to the input at every time step
        self.peephole = peephole
        if not peephole and context_size != hidden_size:
            raise ValueError(
                "peephole=False: the context size {} must match the hidden size {} in DecoderGRU".format(
                    context_size, hidden_size
                )
            )
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim + context_size * self.peephole,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.embedding.weight.data.set_(embedding_matrix)

    def forward(self, input_sequence, lengths, context=None, state=None):
        """
        If not peephole, use the context vector as initial hidden state at each layer.
        If peephole, concatenate context to embeddings at each time step instead.
        If context is not provided, assume that a state is given (for generation)
        :param state:
        :param input_sequence: (batch_size, seq_len)
        :param lengths: (batch_size)
        :param context: (batch, hidden_size) vector on which to condition
        :return: ouptut predictions (batch_size, seq_len, hidden_size) [, h_n (batch, num_layers, hidden_size)]
        """
        embedded = self.embedding(input_sequence)
        if context is not None:
            batch_size, context_size = context.data.shape
            seq_len = input_sequence.data.shape[1]
            if self.peephole:
                context_for_input = context.unsqueeze(1).expand(batch_size, seq_len, context_size)
                embedded = torch.cat((embedded, context_for_input), dim=2)
            packed = pack_padded_sequence(embedded, lengths, batch_first=True)

            if not self.peephole:
                # No peephole. Use context as initial hidden state
                # expand to the number of layers in the decoder
                context = (
                    context.unsqueeze(0)
                    .expand(self.num_layers, batch_size, self.hidden_size)
                    .contiguous()
                )

                output, _ = self.gru(packed, context)
            else:
                output, _ = self.gru(packed)
            return pad_packed_sequence(output, batch_first=True)[0]
        elif state is not None:
            output, h_n = self.gru(embedded, state)
            return output, h_n
        else:
            raise ValueError("Must provide at least state or context")


# from models/decoders.py::TextDecoder (verbatim)
class TextDecoder(nn.Module):
    """
    Regular decoder. Add a fc layer on top of the DecoderGRU to predict the output (used in HRED for example)
    """

    def __init__(
        self, hidden_size, context_size, num_layers, vocab_size, peephole, embedding_dim=512
    ):
        super(TextDecoder, self).__init__()
        self.num_layers = num_layers
        self.peephole = peephole
        self.decoder = DecoderGRU(
            hidden_size=hidden_size,
            context_size=context_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            peephole=peephole,
        )
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.cuda_available = torch.cuda.is_available()

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.decoder.set_pretrained_embeddings(embedding_matrix)

    def forward(self, input, lengths, context, log_probabilities):
        """
        :param log_probabilities:
        :param input: (batch, max_utterance_length)
        :param lengths:
        :param context: (batch, hidden_size)
        :return:
        """
        decoded = self.decoder(input, lengths, context=context)
        output = self.out(decoded)  # (batch, seq_len, vocab_size)
        # NOTE: dim=-1 added (mandatory in modern torch; upstream 2018 code
        # relied on the removed dim-less softmax default over the last dim).
        if log_probabilities:
            return F.log_softmax(output.transpose(0, 2), dim=-1).transpose(0, 2)
        else:
            return F.softmax(output.transpose(0, 2), dim=-1).transpose(0, 2)


# from models/hred.py::HRED, adapted forward() only for the gensen=False
# encoder path constructed above (encoder gensen kwarg set accordingly;
# the pretrained-embedding transfer line tied to GenSen is dropped since
# there is no GenSen instance in this vendor).
class HRED(nn.Module):
    def __init__(self, train_vocab, params=None):
        super(HRED, self).__init__()
        self.params = params
        self.train_vocab = train_vocab
        self.cuda_available = torch.cuda.is_available()

        # HRNN encoder (gensen=False branch -- see module docstring)
        self.encoder = HRNN(
            params=params["hrnn_params"],
            gensen=False,
            train_vocabulary=train_vocab,
            train_gensen=False,
            conv_bidirectional=False,
        )
        self.decoder = TextDecoder(
            context_size=params["hrnn_params"]["conversation_encoder_hidden_size"],
            vocab_size=len(train_vocab),
            **params["decoder_params"],
        )
        if self.cuda_available:
            self.cuda()

    def forward(self, input_dict):
        # encoder result: (batch_size, max_conv_length, hidden_size)
        conversation_representations = self.encoder(input_dict, return_all=True)
        batch_size, max_conversation_length, max_utterance_length = input_dict[
            "dialogue"
        ].data.shape

        # Decoder:
        utterances = input_dict["dialogue"].view(batch_size * max_conversation_length, -1)
        lengths = input_dict["lengths"]
        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(
            lengths, cuda=self.cuda_available
        )

        sorted_utterances = utterances.index_select(0, sorted_idx)

        # shift the context vectors one step in time
        tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
        pad_tensor = torch.zeros(
            batch_size, 1, self.params["hrnn_params"]["conversation_encoder_hidden_size"], out=tt()
        )

        conversation_representations = torch.cat(
            (pad_tensor, conversation_representations), 1
        ).narrow(1, 0, max_conversation_length)
        # and reshape+reorder the same way as utterances
        conversation_representations = (
            conversation_representations.contiguous()
            .view(
                batch_size * max_conversation_length,
                self.params["hrnn_params"]["conversation_encoder_hidden_size"],
            )
            .index_select(0, sorted_idx)
        )

        # consider only lengths > 0
        num_positive_lengths = np.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        conversation_representations = conversation_representations[:num_positive_lengths]

        # Run decoder
        # NOTE: dim=-1 added (see TextDecoder note above).
        outputs = F.log_softmax(
            self.decoder(
                sorted_utterances,
                sorted_lengths,
                conversation_representations,
                log_probabilities=True,
            ).transpose(0, 2),
            dim=-1,
        ).transpose(0, 2)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                max_utterance_length,
                len(self.train_vocab),
                out=tt(),
            )
            outputs = torch.cat((outputs, pad_tensor), 0)

        # retrieve original order
        outputs = outputs.index_select(0, rev).view(
            batch_size, max_conversation_length, max_utterance_length, -1
        )
        return outputs


def build_redial():
    train_vocab = [f"tok{i}" for i in range(200)]
    params = {
        "hrnn_params": {
            "use_movie_occurrences": False,
            "sentence_encoder_hidden_size": 16,
            "conversation_encoder_hidden_size": 16,
            "sentence_encoder_num_layers": 1,
            "conversation_encoder_num_layers": 1,
            "use_dropout": 0.0,
            "embedding_dimension": 12,
        },
        "decoder_params": {
            "hidden_size": 16,
            "num_layers": 1,
            "peephole": True,
            "embedding_dim": 12,
        },
    }
    return HRED(train_vocab=train_vocab, params=params)


def example_input_redial():
    batch_size = 2
    max_conv_length = 3
    max_utterance_length = 5

    dialogue = torch.randint(1, 200, (batch_size, max_conv_length, max_utterance_length))
    senders = torch.ones(batch_size, max_conv_length)
    lengths = np.full((batch_size, max_conv_length), max_utterance_length, dtype=np.int64)
    conversation_lengths = np.full((batch_size,), max_conv_length, dtype=np.int64)

    input_dict = {
        "dialogue": dialogue,
        "senders": senders,
        "lengths": lengths,
        "conversation_lengths": conversation_lengths,
        "movie_occurrences": None,
    }
    return (input_dict,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("REDIAL HRED", "build_redial", "example_input_redial", 2018, "vendored"),
]
