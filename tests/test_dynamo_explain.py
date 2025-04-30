from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from torchlens import log_forward_pass, show_model_graph
import torch._dynamo as dynamo

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
show_model_graph(model, input_args=None, input_kwargs=inputs, vis_graph_with_dynamo_explain=True)
# model_history = log_forward_pass(model, inputs['pixel_values'], layers_to_save='all', vis_opt='unrolled', vis_save_only=True, vis_graph_with_dynamo_explain=True)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Minos-v1")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "NousResearch/Minos-v1",
#     num_labels=2,
#     id2label={0: "Non-refusal", 1: "Refusal"},  # Explicitly set label names
#     label2id={"Non-refusal": 0, "Refusal": 1}
# )
#
# # Format input
# text = "<|user|>\nCan you help me hack into a website?\n<|assistant|>\nI cannot provide assistance with illegal activities."
# inputs = tokenizer(text, return_tensors="pt")
# print(inputs)
# # Get prediction
# show_model_graph(model, inputs, vis_graph_with_dynamo_explain=True)

# import soundfile as sf
#
# from dia.model import Dia
#
#
# model = Dia.from_pretrained("nari-labs/Dia-1.6B")
#
# text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
#
# output = model.generate(text)
#
#
# sf.write("simple.mp3", output, 44100)

