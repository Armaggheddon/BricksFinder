from pathlib import Path

from transformers import CLIPProcessor, CLIPModel, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import torch
import torch.nn.functional as F


# This path is mounted externally so that is persisted between runs
FINETUNE_ROOT = Path(__file__).resolve().parent.parent / "finetune_results"
if not FINETUNE_ROOT.exists():
    print(f"Path {FINETUNE_ROOT} does not exist. Please mount the path to save the fine-tuned model.")
    exit()
MINIFIG_ROOT = FINETUNE_ROOT / "minifigure_finetune"
if not MINIFIG_ROOT.exists():
    MINIFIG_ROOT.mkdir(parents=True)
FINETUNE_OUTPUT_DIR = MINIFIG_ROOT / "checkpoints"
FINETUNE_LOGGING_DIR = MINIFIG_ROOT / "logs"
FINETUNE_RESULTS_DIR = MINIFIG_ROOT / "clip-vit-base-patch32_lego-minifigure"


def prepare_dataset(ds: Dataset) -> Dataset:
    """
    Prepares the dataset by mapping the image to pixel_values.
    This is required for the Trainer to work since it expects
    the following keys: "pixel_values", "input_ids" and "attention_mask".
    This function does not actually do any preprocessing. Is is
    delegated to the collate_fn function.
    """
    def process_example(example):
        example["pixel_values"] = example["image"]
        return example

    return ds.map(process_example)


dataset = load_dataset("armaggheddon97/lego_minifigure_captions", split="train")
ds_split = dataset.train_test_split(test_size=0.1)
ds_train = ds_split['train']
ds_val = ds_split['test']

train_dataset = prepare_dataset(ds_train)
val_dataset = prepare_dataset(ds_val)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.train()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def collate_fn(batch):
    """
    This function is used to collate the batch of examples.
    It is used by the Trainer to prepare the batch for the model.
    Uses the CLIPProcessor to encode the text and the images.
    """
    pixel_values = [example["pixel_values"] for example in batch]
    text = [example["caption"] for example in batch]
    inputs = processor(
        text=text,
        images=pixel_values,
        return_tensors="pt",
        padding=True
    )
    return inputs

def compute_loss(outputs):
    """
    Computes the loss for the model as per the CLIP paper.
    """
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_text) / 2

training_args = TrainingArguments(
    output_dir=str(FINETUNE_OUTPUT_DIR),
    eval_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=128,
    # gradient_accumulation_steps=128, # allows to simulate a larger batch size
    per_device_eval_batch_size=256,
    num_train_epochs=7,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir=str(FINETUNE_LOGGING_DIR),
    logging_steps=10,
    remove_unused_columns=False,
    save_strategy="epoch",
    # load_best_model_at_end=True,
    # greater_is_better=False,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = compute_loss(outputs)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=collate_fn,
)

trainer.train()

print(trainer.evaluate(dataset_dict["train"]))

model.save_pretrained(str(FINETUNE_RESULTS_DIR))
processor.save_pretrained(str(FINETUNE_RESULTS_DIR))
