# This code uses the model from the OpenAI CLIP
# repository to train a model on the dataset

from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import datasets
import tqdm

THIS_PATH = Path(__file__).resolve()

ROOT = THIS_PATH.parent
CHECKPOINTS = ROOT / "checkpoints"
if not CHECKPOINTS.exists():
    CHECKPOINTS.mkdir()

data = datasets.load_dataset("armaggheddon97/lego_minifigure_captions", split="train")

model, preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
model.train()

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

# ds_split = data.train_test_split(test_size=0.2)
dataset = image_title_dataset(data)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def convert_model_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()
    
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=5e-6, 
    betas=(0.9, 0.98), 
    eps=1e-6, 
    weight_decay=0.2
)

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
        img_logits, txt_logits = model(images, texts)
        ground_truth = torch.arange(len(images)).cuda()
        tot_loss = (
            loss_img(img_logits, ground_truth) 
            + loss_txt(txt_logits, ground_truth)
        ) / 2
        tot_loss.backward()

        optimizer.step()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {tot_loss.item():.4f}")
    
    torch.save(model, CHECKPOINTS / f"checkpoint-epoch-{epoch}.pt")
