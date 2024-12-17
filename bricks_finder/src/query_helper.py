from pathlib import Path
from enum import Enum
from io import StringIO
from dataclasses import dataclass

import faiss
import tqdm
from PIL import Image
from PIL.Image import Image as PILImage
import torch
from torchvision.transforms.functional import to_tensor, resize
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

    @staticmethod
    def from_str(value: str) -> 'IndexType':
        if value == "minifigure":
            return IndexType.MINIFIGURE
        elif value == "brick":
            return IndexType.BRICK
        else:
            raise ValueError(f"Invalid IndexType: {value}")

@dataclass
class QueryResult:
    idx: int
    image: PILImage
    additional_info: str
    distance: float


class QueryHelper:
    def __init__(self, vector_index_root: Path, default_index: IndexType = IndexType.MINIFIGURE):
        
        self.minifigure_root = vector_index_root / "minifigures"
        self.minifigure_dataset = self.minifigure_root / "dataset_cache"
        self.minifigure_model = self.minifigure_root / "model_cache"
        self.minifigure_tokenizer = self.minifigure_root / "tokenizer_cache"
        self.minifigure_index = self.minifigure_root / "index.faiss"
        self.brick_root = vector_index_root / "bricks"
        self.brick_dataset = self.brick_root / "dataset_cache"
        self.brick_model = self.brick_root / "model_cache"
        self.brick_tokenizer = self.brick_root / "tokenizer_cache"
        self.brick_index = self.brick_root / "index.faiss"

        self.minifigure_root.mkdir(parents=True, exist_ok=True)
        self.brick_root.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        class ImageTransform(torch.nn.Module):
            def __init__(self):
                super(ImageTransform, self).__init__()
                self.transform = Compose([
                    Resize(224, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(224),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ])
            def forward(self, x) -> torch.Tensor:
                return self.transform(x)

        image_transform = ImageTransform()
        self.image_transform = torch.compile(image_transform).to(self.device)

        self.current_index_type = default_index
        self.model, self.tokenizer = self._load_model()
        self.dataset = self._load_dataset()

        if not self.minifigure_index.exists():
            self._build_index(IndexType.MINIFIGURE, self.minifigure_index)
        self.vector_index = self._load_vector_index()
        
        # load_dataset(
        #     HF_BRICKS_REPO_ID,
        #     cache_dir=self.brick_root,
        #     split="train",
        #     download_mode="reuse_cache_if_exists",
        # )
        # if not self.brick_index.exists():
        #     self.build_index(IndexType.BRICK)

    def _free_resources(self):
        logger.info(f"Releasing resources for {self.current_index_type} dataset and model")
        if self.dataset is not None:
            del self.dataset
        if self.model is not None:
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
        if self.tokenizer is not None:
            del self.tokenizer
        if self.vector_index is not None:
            del self.vector_index
        logger.info(f"Resources released for {self.current_index_type} dataset and model")
    
    def _load_model(self):
        model_id = (
            HF_MINIFIGURE_MODEL_ID 
            if self.current_index_type == IndexType.MINIFIGURE 
            else HF_BRICKS_REPO_ID
        )
        model_cache = (
            self.minifigure_model 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_model
        )
        tokenizer_cache = (
            self.minifigure_tokenizer 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_tokenizer
        )
        model = CLIPModel.from_pretrained(
            model_id, 
            cache_dir=model_cache, 
            torch_dtype=self.model_dtype
        ).to(self.device)
        logger.info(f"Loaded {model_id} model")

        tokenizer = CLIPTokenizerFast.from_pretrained(
            model_id,
            cache_dir=tokenizer_cache
        )
        logger.info(f"Loaded {model_id} tokenizer")
        return model, tokenizer

    
    def _load_dataset(self):
        dataset_id = (
            HF_MINIFIGURES_DATASET_ID 
            if self.current_index_type == IndexType.MINIFIGURE 
            else HF_BRICKS_REPO_ID
        )
        dataset_cache = (
            self.minifigure_dataset 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_dataset
        )
        dataset = load_dataset(
            dataset_id,
            cache_dir=dataset_cache,
            split="train",
            download_mode="reuse_cache_if_exists",
        )
        logger.info(f"Loaded {self.current_index_type.value} dataset")        
        return dataset

    def _load_vector_index(self):
        index_path = (
            self.minifigure_index 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_index
        )
        vector_index = faiss.read_index(str(index_path))
        logger.info(f"Loaded {self.current_index_type.value} vector index")
        return vector_index
        
    def _switch_dataset(self, index_type: IndexType = IndexType.MINIFIGURE):
        if self.current_index_type == index_type:
            logger.info(f"Already using {index_type} dataset and model")
            return

        self._free_resources()

        self.current_index_type = index_type

        self.model, self.tokenizer = self._load_model()
        self.dataset = self._load_dataset()
        self.vector_index = self._load_vector_index()


    def _build_index(
        self, 
        index_type: IndexType,
        index_path: Path
    ):
        self._switch_dataset(index_type)

        # generate embedding for each image in the dataset
        vector_index = faiss.IndexFlatIP(512)
        for i, example in tqdm.tqdm(enumerate(self.dataset)):
            image = example["image"]
            image_tensor = to_tensor(image).to(self.device)
            patches = self.image_transform(image_tensor).unsqueeze(0)
            image_features = self.model.get_image_features(patches).detach().cpu().numpy()
            vector_index.add(image_features)
        
        # save the index
        faiss.write_index(vector_index, str(index_path))

    def query(self, query: str | PILImage, top_k: int, index_type: IndexType = None) -> list[QueryResult]:
        embeddings = None
        if isinstance(query, str):
            # query is a text to be embedded
            input_tokens = self.tokenizer(
                query,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True
            ).to(self.device)
            embeddings = self.model.get_text_features(**input_tokens).detach().cpu().numpy()
        else:
            input_tokens = self.image_transform(query).unsqueeze(0).to(self.device)
            embeddings = self.model.get_image_features(**input_tokens).detach().cpu().numpy()

        D, I = self.vector_index.search(embeddings, top_k)
        results = []

        for i, idx in enumerate(I[0]):
            image = self.dataset[int(idx)]["image"]
            
            additional_info = StringIO()
            for key, value in self.dataset[int(idx)].items():
                if key != "image":
                    additional_info.write(f"{key}: {value}\n")
            
            results.append(
                QueryResult(
                    idx=str(idx),
                    image=image,
                    additional_info=additional_info.getvalue(),
                    distance=D[0][i]
                )
            )

        return results
    
    def get_image_info(self, idx: int, index_type: IndexType = None) -> tuple[str, PILImage]:
        additional_info = StringIO()
        for key, value in self.dataset[idx].items():
            if key != "image":
                additional_info.write(f"{key}: {value}\n")
        return additional_info.getvalue(), self.dataset[idx]["image"]