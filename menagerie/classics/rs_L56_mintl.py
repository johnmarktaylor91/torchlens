# SOURCE: vendored from zlinao/MinTL @ 2dd042bf9c4d999da0382ab13ac67ebef596e989
# https://raw.githubusercontent.com/zlinao/MinTL/2dd042bf9c4d999da0382ab13ac67ebef596e989/T5.py
#
# Lin et al. 2020 (EMNLP) "MinTL: Minimalist Transfer Learning for Task-Oriented
# Dialogue Systems" -- `MiniT5` subclasses `transformers.T5ForConditionalGeneration`
# and adds a SECOND decoder + LM head ("Levenshtein span" / DST decoder,
# `self.dst_decoder` / `self.dst_lm_head`) that is a structural clone of the base
# T5 decoder/LM-head (same class, same weights at init via `load_state_dict`).
# `forward()` dynamically routes between the two decoder/head pairs depending on
# whether `decoder_input_ids[0, 0] == self.config.decoder_start_token_id` (the
# DST-update decoding path vs. the response-generation path) -- this dual-decoder
# routing IS the paper's "Lev" (Levenshtein belief-span) architectural
# contribution for joint dialogue-state-tracking + response generation, so it is
# vendored (not constructed as an unmodified base-library T5).
#
# `T5.py` is the real, unmodified model-definition file; only mechanical fixes
# for import isolation:
#   - Dropped `AdamW, WEIGHTS_NAME, CONFIG_NAME` from the top-level `transformers`
#     import -- `AdamW` no longer exists at that import path in current
#     `transformers` (moved to `torch.optim.AdamW` upstream years ago), and none
#     of the three names are referenced anywhere else in this file (they were
#     only used by the training-loop script `train.py`, which is not vendored).
#   - `T5Tokenizer` import kept (harmless, needed for the `inference*` helper
#     method signatures) though `sentencepiece` is not required to construct or
#     trace `MiniT5` itself.
#   - No changes to `MiniT5.__init__`, `tie_decoder`, or `forward`'s architecture
#     (layer composition, routing logic, and arithmetic are byte-for-byte the
#     original). The ONLY behavioral-looking edit is a version-skew fix in
#     `forward`: current `transformers` `T5Stack.forward()` returns a
#     `ModelOutput` dataclass (`BaseModelOutputWithPastAndCrossAttentions`), not
#     the plain tuple 2020-era `transformers` returned, so the original code's
#     `decoder_outputs + encoder_outputs` (tuple-concat) is TypeErrors against
#     current `transformers`; `.to_tuple()` is called on both `ModelOutput`s
#     first to restore the original tuple-concat semantics -- this is an
#     `ir/index-identical` compatibility shim for a library-internal return-type
#     change, not an architectural edit.
#   - `generate()` / `inference()` / `inference_sequicity()` are kept verbatim
#     for fidelity but are dialogue-eval-loop helpers built on a since-removed
#     private `transformers` generation API (`_generate_beam_search` /
#     `_generate_no_beam_search`); they are not exercised by `build_mintl()` /
#     `example_input_mintl()` below, which only trace `forward()`.

from copy import deepcopy

import torch
from torch.nn import CrossEntropyLoss
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer  # noqa: F401


class MiniT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # make a copy of decoder for dst
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True

        self.dst_decoder = type(self.decoder)(decoder_config, self.shared)
        self.dst_decoder.load_state_dict(self.decoder.state_dict())
        self.dst_lm_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)
        self.dst_lm_head.load_state_dict(self.lm_head.state_dict())

    def tie_decoder(self):
        decoder_config = deepcopy(self.config)
        decoder_config.is_decoder = True
        self.dst_decoder = type(self.decoder)(decoder_config, self.shared)
        self.dst_decoder.load_state_dict(self.decoder.state_dict())
        self.dst_lm_head = type(self.lm_head)(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.dst_lm_head.load_state_dict(self.lm_head.state_dict())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):
        # DST forward or Response generation forward?
        if decoder_input_ids[0, 0] == self.config.decoder_start_token_id:
            decoder = self.dst_decoder
            lm_head = self.dst_lm_head
        else:
            decoder = self.decoder
            lm_head = self.lm_head
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
            )

        hidden_states = encoder_outputs[0]

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        # Decode
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # current transformers returns a ModelOutput dataclass here, not a plain
        # tuple as in the original 2020 code -- normalize for the tuple-concat
        # below (see header note).
        if hasattr(decoder_outputs, "to_tuple"):
            decoder_outputs = decoder_outputs.to_tuple()
        if hasattr(encoder_outputs, "to_tuple"):
            encoder_outputs = encoder_outputs.to_tuple()

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = lm_head(sequence_output)

        decoder_outputs = (
            lm_logits,
        )  # + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            decoder_outputs = (
                (loss,) + decoder_outputs
            )  # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return decoder_outputs + encoder_outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
    ):
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = (
            early_stopping if early_stopping is not None else self.config.early_stopping
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = (
            length_penalty if length_penalty is not None else self.config.length_penalty
        )
        no_repeat_ngram_size = (
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, (
            "`max_length` should be a strictly positive integer."
        )
        assert isinstance(min_length, int) and min_length >= 0, (
            "`min_length` should be a positive integer."
        )
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, (
            "`num_beams` should be a strictly positive integer."
        )
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (isinstance(bos_token_id, int) and bos_token_id >= 0), (
            "If input_ids is not defined, `bos_token_id` should be a positive integer."
        )
        assert pad_token_id is None or (isinstance(pad_token_id, int) and (pad_token_id >= 0)), (
            "`pad_token_id` should be a positive integer."
        )
        assert (eos_token_id is None) or (isinstance(eos_token_id, int) and (eos_token_id >= 0)), (
            "`eos_token_id` should be a positive integer."
        )
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0, (
            "`no_repeat_ngram_size` should be a positive integer."
        )
        assert isinstance(num_return_sequences, int) and num_return_sequences > 0, (
            "`num_return_sequences` should be a strictly positive integer."
        )
        assert (
            bad_words_ids is None
            or isinstance(bad_words_ids, list)
            and isinstance(bad_words_ids[0], list)
        ), (
            "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"
        )

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, (
                "Input prompt should be of shape (batch_size, sequence length)."
            )

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert num_return_sequences == 1, (
                    "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"
                )

            else:
                # beam_search greedy generation conditions
                assert num_beams >= num_return_sequences, (
                    "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"
                )

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # current position and vocab size
        vocab_size = self.config.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert decoder_start_token_id is not None, (
                "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            )
            assert hasattr(self, "get_encoder"), (
                "{} should have a 'get_encoder' function defined".format(self)
            )
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs = encoder(input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            if isinstance(decoder_start_token_id, int):
                input_ids = torch.full(
                    (effective_batch_size * num_beams, 1),
                    decoder_start_token_id,
                    dtype=torch.long,
                    device=next(self.parameters()).device,
                )
            else:
                # pass a batch of start tokens, but doesn't support beam search and sampling
                input_ids = decoder_start_token_id
            cur_len = 1

            assert batch_size == encoder_outputs[0].shape[0], (
                f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "
            )

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (
                encoder_outputs[0].index_select(0, expanded_batch_idxs),
                *encoder_outputs[1:],
            )

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )

        return output

    def inference(
        self,
        tokenizer,
        reader,
        prev,
        input_ids=None,
        attention_mask=None,
        turn_domain=None,
        db=None,
    ):
        dst_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.encode("<eos_b>")[0],
            decoder_start_token_id=self.config.decoder_start_token_id,
            max_length=200,
        )
        dst_outputs = dst_outputs.tolist()
        batch_size = input_ids.shape[0]
        constraint_dict_updates = [  # noqa: F841 (verbatim upstream; unused in original too)
            reader.bspan_to_constraint_dict(tokenizer.decode(dst_outputs[i]))
            for i in range(batch_size)
        ]

        if prev["bspn"]:
            # update the belief state
            dst_outputs = [
                reader.update_bspn(prev_bspn=prev["bspn"][i], bspn_update=dst_outputs[i])
                for i in range(batch_size)
            ]

        # compute the DB state using the updated domain
        db_state = []
        for bi, bspn_list in enumerate(dst_outputs):
            db_vector = reader.bspan_to_DBpointer(tokenizer.decode(bspn_list), turn_domain[bi])
            if sum(db_vector) == 0:
                db_state.append(tokenizer.encode("[db_state0]"))
            else:
                db_state.append([tokenizer.encode("[db_state0]")[0] + db_vector.index(1) + 1])
            # use gold booking pointer, because we cannot issue BOOKING API
            if db[bi][0] >= tokenizer.encode("[db_state0+bookfail]")[0]:
                if db[bi][0] >= tokenizer.encode("[db_state0+booksuccess]")[0]:
                    db_state[-1][0] += 10
                else:
                    db_state[-1][0] += 5

        db_state = torch.tensor(
            db_state,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        resp_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.encode("<eos_r>")[0],
            decoder_start_token_id=db_state,
            max_length=200,
        )

        resp_outputs = resp_outputs[:, 1:].tolist()  # skip DB state
        return dst_outputs, resp_outputs

    def inference_sequicity(
        self,
        tokenizer,
        reader,
        prev,
        input_ids=None,
        attention_mask=None,
        turn_domain=None,
        db=None,
    ):
        dst_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.encode("<eos_b>")[0],
            decoder_start_token_id=self.config.decoder_start_token_id,
            max_length=200,
        )
        dst_outputs = dst_outputs.tolist()
        # compute the DB state using the updated domain
        db_state = []
        for bi, bspn_list in enumerate(dst_outputs):
            db_vector = reader.bspan_to_DBpointer(tokenizer.decode(bspn_list), turn_domain[bi])
            if sum(db_vector) == 0:
                db_state.append(tokenizer.encode("[db_state0]"))
            else:
                db_state.append([tokenizer.encode("[db_state0]")[0] + db_vector.index(1) + 1])
            # use gold booking pointer, because we cannot issue BOOKING API
            if db[bi][0] >= tokenizer.encode("[db_state0+bookfail]")[0]:
                if db[bi][0] >= tokenizer.encode("[db_state0+booksuccess]")[0]:
                    db_state[-1][0] += 10
                else:
                    db_state[-1][0] += 5

        db_state = torch.tensor(
            db_state,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        resp_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.encode("<eos_r>")[0],
            decoder_start_token_id=db_state,
            max_length=200,
        )

        resp_outputs = resp_outputs[:, 1:].tolist()  # skip DB state
        return dst_outputs, resp_outputs


def build_mintl():
    config = T5Config(
        vocab_size=200,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_heads=2,
        d_kv=16,
        decoder_start_token_id=0,
    )
    return MiniT5(config)


def example_input_mintl():
    batch = 2
    seq_len = 8
    input_ids = torch.randint(1, 200, (batch, seq_len))
    # decoder_input_ids[0, 0] == decoder_start_token_id (0) routes through the
    # dst_decoder / dst_lm_head branch -- the novel architectural path.
    decoder_input_ids = torch.zeros((batch, 4), dtype=torch.long)
    decoder_input_ids[:, 1:] = torch.randint(1, 200, (batch, 3))
    return (input_ids, None, None, decoder_input_ids)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MinTL", "build_mintl", "example_input_mintl", 2020, "vendored"),
]
