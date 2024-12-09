# This code uses the model from the OpenAI CLIP
# repository to train a model on the dataset

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import datasets
import tqdm

data = datasets.load_dataset("armaggheddon97/lego_minifigure_captions", split="train")

model, preprocess = clip.load("ViT-B/32", device="cuda", jit=False)


class image_title_dataset():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = preprocess(row["image"])
        text = clip.tokenize(row["caption"])
        return image, text.squeeze(0)
    
dataset = image_title_dataset(data)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def convert_model_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))

    for batch in pbar:
        optimizer.zero_grad()
        images, texts = batch
        images = images.cuda()
        texts = texts.cuda()
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_logits = model.logit_scale * image_features @ text_features.t()
        text_logits = model.logit_scale * text_features @ image_features.t()
        loss = loss_img(image_logits, torch.arange(len(images)).cuda()) + loss_txt(text_logits, torch.arange(len(texts)).cuda())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item()}")

torch.save(model.state_dict(), "finetuned-clip.pth")