# SOURCE: vendored from zhjohnchan/R2Gen @ main
# https://raw.githubusercontent.com/zhjohnchan/R2Gen/main/models/r2gen.py
# https://raw.githubusercontent.com/zhjohnchan/R2Gen/main/modules/visual_extractor.py
# https://raw.githubusercontent.com/zhjohnchan/R2Gen/main/modules/encoder_decoder.py
# https://raw.githubusercontent.com/zhjohnchan/R2Gen/main/modules/att_model.py
# https://raw.githubusercontent.com/zhjohnchan/R2Gen/main/modules/caption_model.py
#
# R2Gen ("Generating Radiology Reports via Memory-driven Transformer", Chen et al., EMNLP
# 2020): a CNN visual extractor (torchvision backbone) feeding a Transformer encoder-decoder
# whose decoder layers are conditioned on a recurrent "relational memory" module (a small
# self-attention-over-memory-slots RNN) via a memory-conditioned LayerNorm. Classes below
# (`VisualExtractor`, `CaptionModel`, `AttModel`, the Transformer/Encoder/Decoder/
# RelationalMemory stack, `EncoderDecoder`, `R2GenModel`) are copied verbatim from the four
# source files above and merged into one module. The only trimming applied: the beam-search /
# sampling code paths in `CaptionModel.beam_search` / `AttModel._sample_beam` / `AttModel.
# _sample` / `AttModel._diverse_sample` called `modules.utils.repeat_tensors` /
# `penalty_builder` / `split_tensors`, and `modules/utils.py` unconditionally imports `cv2`
# (not a base dependency here, and unrelated to the architecture -- it is only used by an
# unrelated heatmap-visualization helper in that file). Since none of `forward`/`_forward`/
# `mode='forward'` (the standard training forward pass traced below) touch the beam-search /
# sampling code paths, the `modules.utils` import and those sampling methods are dropped
# entirely rather than stubbed -- the architecture executed on the traced path is unchanged
# verbatim code. `R2GenModel.__init__`'s `args.dataset_name` dispatch (choosing
# `forward_iu_xray` vs `forward_mimic_cxr`) is preserved as-is; we trace the mimic_cxr
# (single-image) path.
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---- modules/visual_extractor.py ----
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(tv_models, self.visual_extractor)(weights=None)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


# ---- modules/caption_model.py (beam-search / diverse-sample methods dropped: unreachable
#      on the traced mode='forward' path and depend on modules.utils, see header) ----
class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get("mode", "forward")
        if "mode" in kwargs:
            del kwargs["mode"]
        return getattr(self, "_" + mode)(*args, **kwargs)


# ---- modules/att_model.py (pack_wrapper / sort_pack_padded_sequence kept verbatim;
#      _sample_beam / _sample / _diverse_sample dropped, see header) ----
def sort_pack_padded_sequence(input, lengths):
    from torch.nn.utils.rnn import pack_padded_sequence

    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    from torch.nn.utils.rnn import pad_packed_sequence

    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    from torch.nn.utils.rnn import PackedSequence

    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, args, tokenizer):
        super(AttModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        self.input_encoding_size = args.d_model
        self.rnn_size = args.d_ff
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.max_seq_length = args.max_seq_length
        self.att_feat_size = args.d_vf
        self.att_hid_size = args.d_model

        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.pad_idx = args.pad_idx

        self.use_bn = args.use_bn

        self.embed = lambda x: x
        self.fc_embed = lambda x: x
        self.att_embed = nn.Sequential(
            *(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())
                + (
                    nn.Linear(self.att_feat_size, self.input_encoding_size),
                    nn.ReLU(),
                    nn.Dropout(self.drop_prob_lm),
                )
                + ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())
            )
        )

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        p_att_feats = self.ctx2att(att_feats)
        return fc_feats, att_feats, p_att_feats, att_masks

    def get_logprobs_state(
        self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1
    ):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)
        return logprobs, state


# ---- modules/encoder_decoder.py ----
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask_ = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask_) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model
    ):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3
        )

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(
            nn.Linear(rm_num_slots * rm_d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(rm_d_model, rm_d_model),
        )

        self.mlp_beta = nn.Sequential(
            nn.Linear(rm_num_slots * rm_d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # noqa: E741 (verbatim var name from real repo)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):
    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, : self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoder(AttModel):
    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(
            num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads
        )
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(
                    self.d_model,
                    c(attn),
                    c(attn),
                    c(ff),
                    self.dropout,
                    self.rm_num_slots,
                    self.rm_d_model,
                ),
                self.num_layers,
            ),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm,
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = seq.data > 0
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
            att_feats, att_masks, seq
        )
        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs


# ---- models/r2gen.py ----
class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == "iu_xray":
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def forward_iu_xray(self, images, targets=None, mode="train"):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == "train":
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode="forward")
        elif mode == "sample":
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode="train"):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == "train":
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode="forward")
        elif mode == "sample":
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")
        else:
            raise ValueError
        return output


# ---- staging harness ----
class _Args:
    def __init__(self):
        self.visual_extractor = "resnet18"
        self.visual_extractor_pretrained = False
        self.dataset_name = "mimic_cxr"
        self.d_model = 32
        self.d_ff = 64
        self.d_vf = 512  # resnet18 final feature-map channels
        self.num_layers = 2
        self.num_heads = 4
        self.dropout = 0.1
        self.drop_prob_lm = 0.1
        self.max_seq_length = 12
        self.use_bn = 0
        self.bos_idx = 0
        self.eos_idx = 1
        self.pad_idx = 2
        self.rm_num_slots = 3
        self.rm_num_heads = 2
        self.rm_d_model = 32


class _TinyTokenizer:
    def __init__(self, vocab_size=40):
        self.idx2token = {i: str(i) for i in range(vocab_size)}


class R2GenTrainWrapper(nn.Module):
    """mode='train' forward-pass wrapper (ctx2att is required by AttModel._prepare_feature
    but never actually invoked by EncoderDecoder's overridden _prepare_feature / _forward
    train path; not added since it is unreachable here)."""

    def __init__(self):
        super().__init__()
        args = _Args()
        tokenizer = _TinyTokenizer(vocab_size=40)
        self.model = R2GenModel(args, tokenizer)

    def forward(self, images, targets):
        return self.model(images, targets, mode="train")


def build_r2gen():
    torch.manual_seed(0)
    return R2GenTrainWrapper()


def example_input_r2gen():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    targets = torch.randint(1, 40, (2, 10))
    targets[:, 0] = 0  # bos
    return (images, targets)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("R2Gen", "build_r2gen", "example_input_r2gen", 2020, "vendored"),
]
