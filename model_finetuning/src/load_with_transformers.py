# load the finetuned model with trasnformers library

import torch
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import datasets

config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPModel.from_pretrained("finetuned-clip.pth", config=config)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Now you can use the model and processor as you would normally
# test it

dataset = datasets.load_dataset("armaggheddon97/lego_minifigure_captions", split="train")

captions = [dataset[i]["caption"] for i in range(3)]
image = dataset[0]["image"]
inputs = processor(text=captions, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

# calculate probability of image and text pairs
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1).detach().cpu().numpy().tolist()

# print similarity for each label
print("Similarity for each label:")
print(f"Label 0: {probs[0][0]}")
print(f"Label 1: {probs[0][1]}")
print(f"Label 2: {probs[0][2]}")

