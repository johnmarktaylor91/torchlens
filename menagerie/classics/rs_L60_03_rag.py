# SOURCE: real HuggingFace transformers library classes (transformers==4.57.6),
# transformers/models/rag/modeling_rag.py -- RagSequenceForGeneration and
# RagTokenForGeneration are genuine, distinct model classes shipped in the installed
# base `transformers` package (no vendored/copied source, no architectural modification).
#
# RAG (Retrieval-Augmented Generation, Lewis et al. NeurIPS 2020,
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"). RagSequenceForGeneration
# marginalizes the generator's per-token likelihood at the SEQUENCE level (one document is
# responsible for the whole generated sequence, then likelihoods are combined across the
# top-k retrieved documents); RagTokenForGeneration marginalizes at the TOKEN level (each
# generated token can draw probability mass from a different retrieved document). Both
# combine a DPR-style question encoder + a generator (BART/T5-family) with a differentiable
# marginalization over retrieved documents, and are registered as first-class HF classes,
# not thin aliases of BART/T5.
#
# The staging wrapper below constructs each RAG model at a tiny random-init size and calls
# it directly with pre-supplied `context_input_ids` / `context_attention_mask` /
# `doc_scores` (bypassing `RagRetriever`, which needs a real wiki_dpr/FAISS index/dataset
# download that is out of scope for a tiny architecture trace) -- this is the documented
# `output_retrieved=False` retriever-bypass code path built into `RagSequenceForGeneration`
# / `RagTokenForGeneration` itself (see their `forward()` signatures), so it exercises the
# real generator + marginalization machinery end to end with no architectural changes.

import torch
import torch.nn as nn
from transformers import (
    BartConfig,
    DPRConfig,
    RagConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.dpr.modeling_dpr import DPRQuestionEncoder

MENAGERIE_ZOO = "vendored-pytorch"

_VOCAB_SIZE = 96
_N_DOCS = 2
_BATCH = 1

_Q_CONFIG = DPRConfig(
    vocab_size=_VOCAB_SIZE,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=32,
    projection_dim=0,
)

_GEN_CONFIG = BartConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=16,
    encoder_layers=1,
    decoder_layers=1,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=32,
    decoder_ffn_dim=32,
    max_position_embeddings=32,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    decoder_start_token_id=1,
)

_RAG_CONFIG = RagConfig.from_question_encoder_generator_configs(
    _Q_CONFIG,
    _GEN_CONFIG,
    n_docs=_N_DOCS,
    retrieval_vector_size=16,
    vocab_size=_VOCAB_SIZE,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    decoder_start_token_id=1,
)


class _RagWrapper(nn.Module):
    """Thin torchlens-friendly wrapper: real RAG class + positional forward surface.

    RAG's own forward() is kwargs-only and (by default) expects a live RagRetriever;
    this wrapper constructs the real HF class with a real DPR question encoder + real
    BART generator, then calls it with directly-supplied context tensors via the
    documented retriever-bypass path (`output_retrieved=False` + explicit
    `context_input_ids`/`context_attention_mask`/`doc_scores`). No RAG internals are
    modified.
    """

    def __init__(self, rag_cls):
        super().__init__()
        question_encoder = DPRQuestionEncoder(_Q_CONFIG)
        generator = BartForConditionalGeneration(_GEN_CONFIG)
        self.model = rag_cls(
            config=_RAG_CONFIG, question_encoder=question_encoder, generator=generator
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        context_input_ids,
        context_attention_mask,
        doc_scores,
        decoder_input_ids,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            decoder_input_ids=decoder_input_ids,
            output_retrieved=False,
        )
        return out.logits


def build_rag_sequence():
    m = _RagWrapper(RagSequenceForGeneration)
    m.eval()
    return m


def build_rag_token():
    m = _RagWrapper(RagTokenForGeneration)
    m.eval()
    return m


def _example_input():
    input_ids = torch.randint(3, _VOCAB_SIZE, (_BATCH, 5))
    attention_mask = torch.ones_like(input_ids)
    context_input_ids = torch.randint(3, _VOCAB_SIZE, (_BATCH * _N_DOCS, 8))
    context_attention_mask = torch.ones_like(context_input_ids)
    doc_scores = torch.randn(_BATCH, _N_DOCS)
    decoder_input_ids = torch.randint(3, _VOCAB_SIZE, (_BATCH, 4))
    return (
        input_ids,
        attention_mask,
        context_input_ids,
        context_attention_mask,
        doc_scores,
        decoder_input_ids,
    )


def example_input_rag_sequence():
    return _example_input()


def example_input_rag_token():
    return _example_input()


MENAGERIE_ENTRIES = [
    (
        "RAG-Sequence (Retrieval-Augmented Generation, sequence-level marginalization)",
        build_rag_sequence,
        example_input_rag_sequence,
        2020,
        "vendored-pytorch",
    ),
    (
        "RAG-Token (Retrieval-Augmented Generation, token-level marginalization)",
        build_rag_token,
        example_input_rag_token,
        2020,
        "vendored-pytorch",
    ),
]
