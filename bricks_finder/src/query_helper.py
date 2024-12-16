from pathlib import Path
from enum import Enum

import faiss
import torch
from loguru import logger
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from datasets import load_dataset
from transformers import CLIPModel, CLIPTokenizerFast


HF_MINIFIGURES_DATASET_ID = "armaggheddon97/lego_minifigure_captions"
HF_MINIFIGURE_MODEL_ID = "armaggheddon97/clip-vit-base-patch32_lego-minifigure"
HF_BRICKS_REPO_ID = "armaggheddon97/lego_bricks_captions"



class IndexType(Enum):
    MINIFIGURE = "minifigure"
    BRICK = "brick"


class QueryHelper:
    def __init__(self, vector_index_root: Path, default_index: IndexType = IndexType.MINIFIGURE):
        
        self.minifigure_root = vector_index_root / "minifigures"
        self.minifigure_dataset = self.minifigure_root / "dataset_cache"
        self.minifigure_model = self.minifigure_root / "model_cache"
        self.minifigure_index = self.minifigure_root / "index.faiss"
        self.brick_root = vector_index_root / "bricks"
        self.brick_dataset = self.brick_root / "dataset_cache"
        self.brick_model = self.brick_root / "model_cache"
        self.brick_index = self.brick_root / "index.faiss"

        self.minifigure_root.mkdir(parents=True, exist_ok=True)
        self.brick_root.mkdir(parents=True, exist_ok=True)

        self.image_transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.image_transform = torch.jit.script(self.image_transform)

        load_dataset(
            HF_MINIFIGURES_DATASET_ID,
            cache_dir=self.minifigure_root,
            split="train",
            download_mode="reuse_cache_if_exists",
        )
        if not self.minifigure_index.exists():
            pass
        
        # load_dataset(
        #     HF_BRICKS_REPO_ID,
        #     cache_dir=self.brick_root,
        #     split="train",
        #     download_mode="reuse_cache_if_exists",
        # )
        # if not self.brick_index.exists():
        #     self.build_index(IndexType.BRICK)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.load_model(default_index)

        
    def load_model(self, index_type: IndexType):
        # free up resources
        del self.dataset
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        del self.vector_index

        dataset_id = HF_MINIFIGURES_DATASET_ID if index_type == IndexType.MINIFIGURE else HF_BRICKS_REPO_ID
        model_id = HF_MINIFIGURE_MODEL_ID if index_type == IndexType.MINIFIGURE else HF_BRICKS_REPO_ID
        model_cache = self.minifigure_model if index_type == IndexType.MINIFIGURE else self.brick_model
        index_path = self.minifigure_index if index_type == IndexType.MINIFIGURE else self.brick_index
        dataset_cache = self.minifigure_dataset if index_type == IndexType.MINIFIGURE else self.brick_dataset
        
        self.dataset = load_dataset(
            dataset_id,
            cache_dir=dataset_cache,
            split="train",
            download_mode="reuse_cache_if_exists",
        )

        self.model = CLIPModel.from_pretrained(
            model_id, 
            cache_dir=model_cache, 
            device=self.device, 
            dtype=self.model_dtype
        )

        self.vector_index = faiss.read_index(str(index_path))
        self.loaded_index = index_type

    def build_index(
        self, 
        dataset_path: Path, 
        model_id: str, 
        model_cache: Path,
        index_path: Path
    ):
        dataset = load_dataset(
            dataset_path,
            split="train",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.float16 if device == "cuda" else torch.float32
        model = CLIPModel.from_pretrained(
            model_id, 
            cache_dir=model_cache, 
            device=device, 
            dtype=model_dtype
        )

        # generate embedding for each image in the dataset
        vector_index = faiss.IndexFlatIP(768)
        for i, example in enumerate(dataset):
            image = example["image"]
            patches = self.image_transform(image).unsqueeze(0)
            image_features = model.get_image_features(**patches).cpu().numpy()
            vector_index.add(image_features)
        
        # save the index
        faiss.write_index(vector_index, str(index_path))

        # this is a one time operation, clear resources at end
        del dataset
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        del vector_index
