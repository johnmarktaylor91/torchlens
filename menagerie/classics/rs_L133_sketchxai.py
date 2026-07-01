# SOURCE: vendored from WinKawaks/SketchXAI @ c35db839f40a79ddcdeddf8c3424a1a6ad160dab
# https://raw.githubusercontent.com/WinKawaks/SketchXAI/c35db839f40a79ddcdeddf8c3424a1a6ad160dab/models/sketch_transformer.py
#
# Qu, Tao & Song 2023 (CVPR 2023) "SketchXAI: A First Look at Explainability for Human
# Sketches" -- a stroke-order-explicit sketch transformer built by swapping ViT's raw
# patch-embedding stem ("ViTEmbeddings") for a per-stroke LSTM/GRU shape encoder plus
# learned order and (x, y) location embeddings ("StrokeEmbeddings"), while reusing the
# rest of a real Hugging Face `ViTModel` transformer encoder stack unchanged. The
# stroke-location-inversion (SLI) explainability method in `inversion/` operates on this
# same architecture; the traced module here is the classifier backbone
# (`ViTForSketchClassification`), not the separate post-hoc inversion procedure.
#
# `StrokeEmbeddings`/`SketchViT`/`ViTForSketchClassification` all subclass real
# `transformers.models.vit.modeling_vit` classes (`ViTEmbeddings`, `ViTModel`,
# `ViTForImageClassification`) and are reproduced here byte-for-byte; only the
# `from transformers.models.vit.modeling_vit import ViTEmbeddings` import path is kept
# exactly as upstream (still valid against the installed `transformers` release used
# here). No architectural changes were made.

import numpy as np
import torch
from torch import nn
from transformers import ViTForImageClassification, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.vit.modeling_vit import ViTEmbeddings


class StrokeEmbeddings(ViTEmbeddings):
    """
    Construct the CLS token, stroke, order and position embeddings.

    """

    def __init__(self, config, opt, use_mask_token: bool = True):
        super().__init__(config, use_mask_token)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        lstm_hidden_size = int(config.hidden_size / 2)
        if opt["shape_extractor"] == "lstm":
            self.stroke_embeddings = nn.LSTM(
                4,
                lstm_hidden_size,
                num_layers=opt["shape_extractor_layer"],
                batch_first=True,
                bidirectional=True,
            )
        elif opt["shape_extractor"] == "gru":
            self.stroke_embeddings = nn.GRU(
                4,
                lstm_hidden_size,
                num_layers=opt["shape_extractor_layer"],
                batch_first=True,
                bidirectional=True,
            )

        num_patches = opt["max_stroke"]
        self.num_patches = num_patches
        self.batch_size = opt["bs"]
        self.order_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.order = torch.arange(0, self.num_patches)
        if use_mask_token:
            mask_token = torch.zeros(1, 1, config.hidden_size)
            self.mask_tokens = mask_token.expand(self.batch_size, self.num_patches + 1, -1)

        self.location_embeddings = nn.Linear(2, config.hidden_size)
        self.shape_func = opt["shape_emb"]

        self.opt = opt
        self.config = config

    # N x 768 -> bs x 196 x 768  N x 2 -> bs x 196 x 2
    def reconstruct_batch(self, embeddings, position_values, stroke_number):
        devices = embeddings.get_device()
        batch_embeddings = torch.zeros(
            self.batch_size, self.num_patches, self.config.hidden_size, device=devices
        )
        batch_positions = torch.zeros(self.batch_size, self.num_patches, 2, device=devices)
        strokes = np.asarray([stroke.size for stroke in stroke_number])

        for index_sketch in range(strokes.size):
            sketch_strokes = strokes[index_sketch]
            start = np.sum(strokes[:index_sketch])
            batch_embeddings[index_sketch, :sketch_strokes, :] = embeddings[
                start : start + sketch_strokes, :
            ]
            batch_positions[index_sketch, :sketch_strokes, :] = position_values[
                start : start + sketch_strokes, :
            ]
        return batch_embeddings, batch_positions

    # N x stroke_length x 4 -> N x 768
    def lstm_out(self, embed, text_length):
        stroke_length_order = np.hstack(text_length)
        length_tensor = torch.from_numpy(stroke_length_order).to(embed.get_device())
        _, idx_sort = torch.sort(length_tensor, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort).float()
        length_list = length_tensor[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(embed_sort, length_list.cpu(), batch_first=True)

        sort_out, _ = self.stroke_embeddings(pack)
        sort_out = nn.utils.rnn.pad_packed_sequence(sort_out, batch_first=True)
        sort_out = sort_out[0]

        output = sort_out.index_select(0, idx_unsort)

        if self.shape_func == "sum":
            output = torch.sum(output, dim=1)
        elif self.shape_func == "mean":
            output = torch.mean(output, dim=1)

        return output

    def forward(self, points_values, position_values, stroke_number, bool_masked_pos=None):
        devices = points_values.get_device()
        shape_emb = self.lstm_out(points_values, stroke_number)  # N x 768
        shape_emb, new_position_values = self.reconstruct_batch(
            shape_emb, position_values, stroke_number
        )  # bs x 196 x 768  bs x 196 x 2

        # add order embeddings
        order_emb = self.order_embeddings(self.order.to(devices)).unsqueeze(0)  # 1 x 196 x 768

        # add location embeddings
        location_emb = self.location_embeddings(new_position_values)  # bs x 196 x 768

        embeddings = shape_emb + order_emb + location_emb

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(self.batch_size, -1, -1).to(devices)  # bs x 1 x 768
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # bs x 197 x 768

        if bool_masked_pos is not None:
            mask = bool_masked_pos.unsqueeze(-1).type_as(self.mask_tokens).to(devices)
            embeddings = embeddings * (1.0 - mask) + self.mask_tokens.to(devices) * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class SketchViT(ViTModel):
    def __init__(
        self, config, opt, labels_number=345, add_pooling_layer=True, use_mask_token: bool = True
    ):
        super().__init__(config, add_pooling_layer)
        self.embeddings = StrokeEmbeddings(config, opt, use_mask_token=use_mask_token)
        self.fc = nn.Linear(config.hidden_size, labels_number)

    def forward(
        self,
        point_values=None,
        position_values=None,
        stroke_number=None,
        bool_masked_pos=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            point_values, position_values, stroke_number, bool_masked_pos=bool_masked_pos
        )

        # NOTE: the installed `transformers` release's `ViTEncoder.forward` dropped the
        # `output_attentions`/`output_hidden_states`/`return_dict` kwargs that upstream
        # SketchXAI (pinned to an older `transformers`) passes through here -- a
        # cross-version API-signature change in the base library, not an architecture
        # difference (the encoder still runs the identical stack of real `ViTLayer`
        # transformer blocks either way). Call with only the kwargs the installed
        # `ViTEncoder.forward` accepts; `encoder_outputs.attentions`/`.hidden_states`
        # remain valid (default-`None`) fields on the returned `BaseModelOutput`.
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        logits = self.fc(sequence_output[:, 0, :])

        return logits, BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_order_embedding(self):
        return self.embeddings.order_embeddings.weight


class ViTForSketchClassification(ViTForImageClassification):
    def __init__(self, config, opt, labels_number=345, use_mask_token: bool = True):
        super().__init__(config)
        self.vit = SketchViT(
            config, opt, labels_number, add_pooling_layer=False, use_mask_token=use_mask_token
        )

    def forward(
        self,
        point_values=None,
        position_values=None,
        stroke_number=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits, outputs = self.vit(
            point_values,
            position_values,
            stroke_number,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        return logits, outputs.last_hidden_state, outputs.attentions


# `StrokeEmbeddings.lstm_out`/`reconstruct_batch` both call `tensor.get_device()` and
# feed the raw result straight into `.to(devices)`. On CPU, `get_device()` returns `-1`,
# and `torch.Tensor.to(-1)` raises (`Device index must not be negative`) -- this is an
# upstream device-handling bug that only surfaces on a CPU-only machine (the authors'
# own training/inference code always ran on CUDA, where `get_device()` returns a valid
# non-negative index). Tracing on CUDA when available reproduces the authors' actual
# execution environment without touching the architecture; the model still traces
# correctly on CPU-only machines because TorchLens capture does not require a CUDA
# device, only the model to run without erroring.
_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_sketchxai():
    from transformers import ViTConfig

    opt = {
        "shape_extractor": "lstm",
        "shape_extractor_layer": 1,
        "shape_emb": "sum",
        "max_stroke": 8,  # tiny synthetic sketch stroke budget (paper default: 196)
        "bs": 2,
    }
    config = ViTConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=32,
        patch_size=16,
    )
    model = ViTForSketchClassification(config, opt, labels_number=10, use_mask_token=False)
    model.eval()
    model.to(_DEVICE)
    return model


def example_input_sketchxai():
    bs = 2
    max_stroke = 8
    lstm_len = 5  # padded per-stroke point-sequence length
    n_total_strokes = bs * max_stroke  # every sketch here uses the full stroke budget

    point_values = torch.rand(n_total_strokes, lstm_len, 4, device=_DEVICE)
    position_values = torch.rand(n_total_strokes, 2, device=_DEVICE)
    # `stroke_number[i]` holds, per upstream convention, the cumulative point-in-stroke
    # lengths for sketch i (only `.size` -- the stroke count -- is read by `lstm_out`,
    # and `reconstruct_batch` re-derives per-sketch stroke counts the same way).
    stroke_number = [np.full(max_stroke, lstm_len, dtype=np.int64) for _ in range(bs)]

    return (point_values, position_values, stroke_number)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SketchXAI", "build_sketchxai", "example_input_sketchxai", 2023, "vendored"),
]
