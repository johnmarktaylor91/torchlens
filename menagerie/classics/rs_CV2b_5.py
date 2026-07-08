# SOURCE: real library class transformers.MBartForConditionalGeneration for VinAIResearch/BARTpho
import torch
import torch.nn as nn
import transformers


class BartphoWrapper(nn.Module):
    """TorchLens wrapper for the exact MBart class used by BARTpho."""

    def __init__(self) -> None:
        """Initialize a tiny random MBartForConditionalGeneration model."""
        super().__init__()
        config = transformers.MBartConfig(
            vocab_size=100,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=32,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        self.model = transformers.MBartForConditionalGeneration(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model from packed encoder and decoder token ids.

        Parameters
        ----------
        x:
            Integer tensor shaped ``(batch, 2, sequence)``.

        Returns
        -------
        torch.Tensor
            Language-model logits.
        """

        output = self.model(
            input_ids=x[:, 0, :].long().remainder(100),
            decoder_input_ids=x[:, 1, :].long().remainder(100),
        )
        return output.logits


def build_bartpho() -> BartphoWrapper:
    """Build a tiny BARTpho-compatible MBart model.

    Returns
    -------
    BartphoWrapper
        Traceable BARTpho wrapper.
    """

    return BartphoWrapper()


def example_input_bartpho() -> torch.Tensor:
    """Create a packed token input.

    Returns
    -------
    torch.Tensor
        Packed token ids.
    """

    return torch.zeros((1, 2, 16), dtype=torch.long)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BARTpho", build_bartpho, example_input_bartpho, 2021, "CV2b"),
]
