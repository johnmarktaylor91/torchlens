from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from torchlens import log_forward_pass

device = "cuda" if torch.cuda.is_available() else "cpu"

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
#
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
# module = AutoModel.from_pretrained('facebook/dinov2-small')
#
# inputs_model = processor(images=image, return_tensors="pt")
# print(inputs_model['pixel_values'].shape)
# model_history = log_forward_pass(module, inputs_model['pixel_values'], layers_to_save='all', vis_opt='unrolled', vis_save_only=True, vis_graph_with_dynamo_explain=True)

from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

# prepare image and text prompt, using the appropriate prompt template

img_path = hf_hub_download(repo_id=model_path, filename='example.png')

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "What is the highest scoring model on ChartQA and what is its score?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)
print(inputs)
model_history = log_forward_pass(model, inputs['pixel_values'], layers_to_save='all', vis_opt='unrolled', vis_save_only=True, vis_graph_with_dynamo_explain=True)
