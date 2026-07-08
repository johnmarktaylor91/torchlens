# SOURCE: real base-library class for UBC-NLP/araT5 via transformers T5ForConditionalGeneration
from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import T5Config, T5ForConditionalGeneration

MENAGERIE_ZOO = "vendored-pytorch"


class AraT5TraceAdapter(nn.Module):
    """Thin adapter that invokes the real T5ForConditionalGeneration with decoder ids."""

    def __init__(self) -> None:
        """Initialize a tiny random T5 model matching the AraT5 architecture class."""
        super().__init__()
        config = T5Config(
            d_model=8,
            d_ff=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=1,
            vocab_size=32,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
        self.model = T5ForConditionalGeneration(config)

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Run the real T5 model and return logits for TorchLens tracing."""
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits


def build_arat5() -> AraT5TraceAdapter:
    """Build a traceable AraT5 architecture adapter."""
    return AraT5TraceAdapter()


def example_input_arat5() -> tuple[Tensor, Tensor]:
    """Return encoder and decoder token ids for AraT5."""
    input_ids = torch.randint(0, 32, (1, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 32, (1, 6), dtype=torch.long)
    return input_ids, decoder_input_ids


MENAGERIE_ENTRIES = [
    ("AraT5", build_arat5, example_input_arat5, 2021, "CV2a-arat5"),
]
