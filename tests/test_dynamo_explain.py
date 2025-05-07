import requests
from PIL import Image

from transformers import AutoProcessor, OmDetTurboForObjectDetection

processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
classes = ["cat", "remote"]
inputs = processor(image, text=classes, return_tensors="pt")
show_model_graph(model, input_args=None, input_kwargs=inputs, vis_graph_with_dynamo_explain=True, vis_nesting_depth=10, save_only=True, vis_fileformat="svg", vis_outpath="output_pray", vis_opt="rolled", vis_direction="leftright")
