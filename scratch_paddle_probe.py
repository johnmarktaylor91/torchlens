import paddle

import torchlens as tl


class PaddleSingleLinear(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


trace = tl.trace(PaddleSingleLinear(), paddle.ones([1, 4]), backend="paddle")
print("num_layers_with_params", trace.num_layers_with_params)
for op in trace.layer_list:
    print(
        op.label,
        "| layer_label=",
        op.layer_label,
        "| module=",
        op.module,
        "| modules=",
        op.modules,
        "| has_params=",
        bool(getattr(op, "_param_logs", None) or getattr(op, "param_logs", None)),
    )
print("=== module fc layer_labels ===", trace.modules["fc"].layer_labels)
print("=== module fc num_layers ===", trace.modules["fc"].num_layers)
