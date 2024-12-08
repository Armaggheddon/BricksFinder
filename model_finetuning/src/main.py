import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Resize, CenterCrop, ConvertImageDtype, Normalize, InterpolationMode
import tqdm
from transformers import (
    VisionTextDualEncoderProcessor,
    VisionTextDualEncoderModel,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from loguru import logger
from datasets import load_dataset

# Code adapted from 
# https://huggingface.co/blog/herooooooooo/clip-finetune

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean=mean, std=std),
        )
    
    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.LongTensor([example["input_ids"] for example in examples])
    attention_mask = torch.LongTensor([example["attention_mask"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }   

def tokenize_captions(examples):
    captions = [example[0] for example in examples[data_args.caption_column]]
    text_inputs = tokenizer(
        captions, 
        max_length=data_args.max_seq_length, 
        padding="max_length", 
        truncation=True)
    examples["input_ids"] = text_inputs.input_ids
    examples["attention_mask"] = text_inputs.attention_mask
    return examples

def transform_images(examples):
    images = [torch.tensor(np.array(image)).permute(2, 0, 1) for image in examples[data_args.image_column]]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples

def filter_corrupt_images(examples):
    """remove problematic images"""
    valid_images = []
    for image_file in examples[data_args.image_column]:
        try:
            Image.open(image_file)
            valid_images.append(True)
        except Exception:
            valid_images.append(False)
    return valid_images


vision_id = "openai/clip-vit-base-patch32"
text_id = "roberta-base"

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    vision_id,
    text_id,
)

tokenizer = AutoTokenizer.from_pretrained(text_id)
image_processor = AutoImageProcessor.from_pretrained(vision_id)
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")

args_dict = {'output_dir': './clip-roberta-finetuned',
    'model_name_or_path': './clip-roberta',
    'data_dir': './data',
    'dataset_name': 'armaggheddon97/lego_minifigure_captions',
    'image_column': 'image',
    'caption_column': 'caption',
    'remove_unused_columns': False,
    'per_device_train_batch_size': 64,
    'per_device_eval_batch_size': 64,
    'learning_rate': 5e-05,
    'warmup_steps': 0,
    'weight_decay': 0.1,
    'overwrite_output_dir': True,
    'push_to_hub': False
}
parser = HfArgumentParser(
    (
        ModelArguments, 
        DataTrainingArguments, 
        TrainingArguments
    )
)
model_args, data_args, training_args = parser.parse_dict(args_dict)

dataset = load_dataset("armaggheddon97/lego_minifigure_captions", split="train")
dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
)
image_processor = AutoImageProcessor.from_pretrained(
    model_args.image_processor_name or model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
config = model.config

set_seed(training_args.seed)

image_transformations = Transform(
    config.vision_config.image_size,
    image_processor.image_mean,
    image_processor.image_std,
)
image_transformations = torch.jit.script(image_transformations)

train_dataset = train_dataset.map(
    function=tokenize_captions,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on train dataset",
)
train_dataset.set_transform(transform_images)

eval_dataset = eval_dataset.map(
    function=tokenize_captions,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on eval dataset",
)
eval_dataset.set_transform(transform_images)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

train_result = trainer.train()

trainer.log_metrics("train", train_result.metrics)
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)

trainer.save_model("./clip-roberta-finetuned")
