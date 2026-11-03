"""Public spellings of the read-time output decode, declared once.

Every user-visible name of the read-time decode door lives here so that a
rename is one edit in this file plus the matching keyword in the two ``Trace``
signatures (``decode_output`` and ``output_table`` in
``torchlens/_data_classes/trace.py``) and the glossary rows.
``tests/test_output_decode_read_time.py`` pins the signatures to these
constants so a half flip fails.
"""

from __future__ import annotations

#: Keyword naming the decode style (``"classification"``, ``"imagenet"``,
#: ``"imagenet1k"``, ``"hf_text"`` or a registered custom style).
STYLE_KW = "style"
#: Keyword naming the output head to decode (an output container path such as
#: ``"logits"``, ``"0"`` or ``"0.logits"``).
HEAD_KW = "head"
#: Keyword carrying the tokenizer an ``hf_text`` decode needs.
TOKENIZER_KW = "tokenizer"
#: Refusal code for a style no built-in or registered detector resolves.
STYLE_UNKNOWN_CODE = "decode_style_unknown"
#: Refusal code for a head that is not one of the trace's output heads.
HEAD_UNKNOWN_CODE = "decode_head_unknown"
#: Refusal code when several output heads qualify and none was named.
HEAD_AMBIGUOUS_CODE = "decode_head_ambiguous"
#: The remedy sentence that points at the read-time door.
UNDETECTED_REMEDY = (
    f"call trace.decode_output({STYLE_KW}=..., {HEAD_KW}=...) to decode from the retained outputs"
)
