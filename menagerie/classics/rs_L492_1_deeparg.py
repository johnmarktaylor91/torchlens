# SOURCE: vendored from gaarangoa/deeparg @ main (deeparg/modern/modeling_deeparg.py,
# deeparg/modern/configuration_deeparg.py)
#
# DeepARG modern PyTorch head: a HuggingFace `PreTrainedModel` MLP classifier over
# precomputed DIAMOND-alignment feature vectors (antibiotic-resistance-gene family
# classification). The architecture is a bespoke feed-forward stack (Linear+ReLU(+Dropout)
# per hidden layer) wrapped as a `transformers` sequence-classification model -- real repo
# code, base-lib deps only (torch + transformers). Copied verbatim aside from import paths.
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

MENAGERIE_ZOO = "vendored-pytorch"


class DeepARGConfig(PretrainedConfig):
    model_type = "deeparg"

    def __init__(
        self,
        input_size=0,
        num_labels=0,
        features=None,
        id2label=None,
        label2id=None,
        hidden_sizes=None,
        dropout=0.5,
        dropout_after_layers=None,
        feature_normalization="batch_minmax",
        problem_type="single_label_classification",
        **kwargs,
    ):
        features = list(features or [])
        hidden_sizes = list(hidden_sizes or [2000, 1000, 500, 100])
        if dropout_after_layers is None:
            dropout_after_layers = max(0, len(hidden_sizes) - 1)

        if input_size == 0 and features:
            input_size = len(features)
        if num_labels == 0 and id2label:
            num_labels = len(id2label)

        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type=problem_type,
            **kwargs,
        )
        self.input_size = int(input_size)
        self.features = features
        self.hidden_sizes = hidden_sizes
        self.dropout = float(dropout)
        self.dropout_after_layers = int(dropout_after_layers)
        self.feature_normalization = feature_normalization


class DeepARGForSequenceClassification(PreTrainedModel):
    config_class = DeepARGConfig
    base_model_prefix = "deeparg"
    main_input_name = "input_features"

    def __init__(self, config):
        super().__init__(config)
        if config.input_size <= 0:
            raise ValueError("DeepARGConfig.input_size must be greater than zero.")
        if config.num_labels <= 0:
            raise ValueError("DeepARGConfig.num_labels must be greater than zero.")

        layers = []
        previous_size = config.input_size
        for layer_index, hidden_size in enumerate(config.hidden_sizes):
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            if layer_index < config.dropout_after_layers:
                layers.append(nn.Dropout(config.dropout))
            previous_size = hidden_size

        self.deeparg = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_features=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if input_features is None:
            input_features = kwargs.get("inputs_embeds")
        if input_features is None:
            raise ValueError("DeepARG requires input_features.")

        input_features = input_features.to(dtype=self.dtype)
        hidden_states = self.deeparg(input_features)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict is False:
            output = (logits,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        hidden_states_output = (hidden_states,) if output_hidden_states else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states_output,
            attentions=None,
        )

    @torch.no_grad()
    def predict_proba(self, input_features):
        was_training = self.training
        self.eval()
        outputs = self(input_features=input_features)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        if was_training:
            self.train()
        return probabilities


def build_deeparg():
    config = DeepARGConfig(
        input_size=64,
        num_labels=8,
        hidden_sizes=[32, 16],
        dropout=0.1,
    )
    return DeepARGForSequenceClassification(config)


def example_input_deeparg():
    return torch.randn(2, 64)


MENAGERIE_ENTRIES = [
    ("DeepARG", build_deeparg, example_input_deeparg, 2018, "SOURCE_AVAILABLE"),
]
