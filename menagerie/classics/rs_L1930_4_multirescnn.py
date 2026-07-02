# SOURCE: vendored from foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network @ master
# https://raw.githubusercontent.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/master/models.py
#
# "A Study of Multi-Filter and Residual Convolutional Networks for Automatic ICD Coding"
# (Li, Yu; NAACL 2020). Multi-window Conv1d feature extractors + residual conv blocks +
# per-code CAML-style label attention over MIMIC discharge-summary text. Classes are copied
# verbatim from the real repo's models.py with one minimal, non-architectural trim:
# `WordRep.__init__`/`forward` unconditionally imported `elmo.elmo.Elmo` at module level
# even when the run doesn't request ELMo features (the repo's own `-use_elmo` flag defaults
# to False, i.e. `MultiResCNN`/`CNN`/`MultiCNN`/`ResCNN` never touch ELMo by default); that
# import requires `allennlp==0.8.4` (torch-1.1-era, incompatible with the installed torch
# 2.x), so it is moved to a lazy import inside the already-existing `if self.use_elmo:`
# branch instead of vendoring allennlp's ELMo implementation. WordRep/OutputLayer/CNN/
# MultiCNN/ResidualBlock/ResCNN/MultiResCNN are otherwise unchanged; the bert_seq_cls model
# and `pick_model` dispatcher (which need pytorch_pretrained_bert, unused by the CNN family)
# are dropped.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from math import floor

MENAGERIE_ZOO = "vendored-pytorch"


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            # pretrained-embedding loading path (build_pretrain_embedding/load_embeddings
            # from the repo's utils.py); not exercised by this build (embed_file=None).
            from utils import build_pretrain_embedding, load_embeddings  # noqa: PLC0415

            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(
                    args.embed_file, dicts["w2ind"], True
                )
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts["w2ind"]) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.use_elmo = args.use_elmo
        if self.use_elmo:
            from elmo.elmo import Elmo  # noqa: PLC0415 (needs allennlp==0.8.4; not vendored, see header)

            self.elmo = Elmo(
                args.elmo_options_file,
                args.elmo_weight_file,
                1,
                requires_grad=args.elmo_tune,
                dropout=args.elmo_dropout,
                gamma=args.elmo_gamma,
            )
            with open(args.elmo_options_file, "r") as fin:
                import json

                _options = json.load(fin)
            self.feature_size += _options["lstm"]["projection_dim"] * 2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {
            1: [self.feature_size, args.num_filter_maps],
            2: [self.feature_size, 100, args.num_filter_maps],
            3: [self.feature_size, 150, 100, args.num_filter_maps],
            4: [self.feature_size, 200, 150, 100, args.num_filter_maps],
        }

    def forward(self, x, target, text_inputs):
        features = [self.embed(x)]

        if self.use_elmo:
            elmo_outputs = self.elmo(text_inputs)
            elmo_outputs = elmo_outputs["elmo_representations"][0]
            features.append(elmo_outputs)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, text_inputs):
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss


class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        filter_size = int(args.filter_size)

        self.conv = nn.Conv1d(
            self.word_rep.feature_size,
            args.num_filter_maps,
            kernel_size=filter_size,
            padding=int(floor(filter_size / 2)),
        )
        xavier_uniform(self.conv.weight)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(",") == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(
                self.word_rep.feature_size,
                args.num_filter_maps,
                kernel_size=filter_size,
                padding=int(floor(filter_size / 2)),
            )
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(",")
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(
                    self.word_rep.feature_size,
                    args.num_filter_maps,
                    kernel_size=filter_size,
                    padding=int(floor(filter_size / 2)),
                )
                xavier_uniform(tmp.weight)
                self.conv.add_module("conv-{}".format(filter_size), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(
                inchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(floor(kernel_size / 2)),
                bias=False,
            ),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(
                outchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=1,
                padding=int(floor(kernel_size / 2)),
                bias=False,
            ),
            nn.BatchNorm1d(outchannel),
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel),
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class ResCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(
                conv_dimension[idx],
                conv_dimension[idx + 1],
                int(args.filter_size),
                1,
                True,
                args.dropout,
            )
            self.conv.add_module("conv-{}".format(idx), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(",")

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(
                self.word_rep.feature_size,
                self.word_rep.feature_size,
                kernel_size=filter_size,
                padding=int(floor(filter_size / 2)),
            )
            xavier_uniform(tmp.weight)
            one_channel.add_module("baseconv", tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(
                    conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout
                )
                one_channel.add_module("resconv-{}".format(idx), tmp)

            self.conv.add_module("channel-{}".format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class _Args:
    """Stand-in for the repo's argparse.Namespace (options.py), tiny-sized for tracing."""

    def __init__(self):
        self.gpu = -1
        self.embed_file = None
        self.embed_size = 16
        self.use_ext_emb = False
        self.use_elmo = False
        self.elmo_options_file = None
        self.elmo_weight_file = None
        self.elmo_tune = False
        self.elmo_dropout = 0.0
        self.elmo_gamma = 0.1
        self.dropout = 0.0
        self.num_filter_maps = 8
        self.filter_size = "3,5,9"
        self.conv_layer = 1


def build_multirescnn():
    torch.manual_seed(0)
    args = _Args()
    vocab_size = 64
    Y = 12  # number of ICD codes (labels)
    dicts = {"w2ind": {str(i): i for i in range(vocab_size - 2)}}
    return MultiResCNN(args, Y, dicts)


def example_input_multirescnn():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 40
    vocab_size = 64
    Y = 12
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    target = torch.randint(0, 2, (batch_size, Y)).float()
    text_inputs = None  # only consumed by the ELMo branch, unused with use_elmo=False
    return (x, target, text_inputs)


MENAGERIE_ENTRIES = [
    ("MultiResCNN", "build_multirescnn", "example_input_multirescnn", 2020, "vendored"),
]
