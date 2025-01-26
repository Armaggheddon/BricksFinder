from pathlib import Path
from enum import Enum
from io import StringIO
from dataclasses import dataclass
import shutil

import faiss
import tqdm
from PIL import Image
from PIL.Image import Image as PILImage
import torch
from torchvision.transforms.functional import to_tensor, resize
from loguru import logger
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from datasets import load_dataset, Dataset
from transformers import CLIPModel, CLIPTokenizerFast


HF_MINIFIGURES_DATASET_ID = "armaggheddon97/lego_minifigure_captions"
HF_MINIFIGURE_MODEL_ID = "armaggheddon97/clip-vit-base-patch32_lego-minifigure"
HF_BRICKS_DATASET_ID = "armaggheddon97/lego_brick_captions"
HF_BRICKS_MODEL_ID = "armaggheddon97/clip-vit-base-patch32_lego-brick"


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
    distance: float

@dataclass
class IndexPaths:
    root: Path
    dataset: Path
    model: Path
    tokenizer: Path
    index: Path


class QueryHelper:
    def __init__(
            self, 
            vector_index_root: Path,
            startup_index: IndexType = IndexType.MINIFIGURE,
            rebuild_indexes: bool = False,
            invalidate_cache: bool = False
        ) -> None:
        minifigure_root = vector_index_root / "minifigures"
        self.minifigure_paths = IndexPaths(
            root = minifigure_root,
            dataset = minifigure_root / "dataset_cache",
            model = minifigure_root / "model_cache",
            tokenizer = minifigure_root / "tokenizer_cache",
            index = minifigure_root / "index.faiss"
        )
        brick_root = vector_index_root / "bricks"
        self.brick_paths = IndexPaths(
            root = brick_root,
            dataset = brick_root / "dataset_cache",
            model = brick_root / "model_cache",
            tokenizer = brick_root / "tokenizer_cache",
            index = brick_root / "index.faiss"
        )

        if invalidate_cache:
            shutil.rmtree(minifigure_root, ignore_errors=True)
            shutil.rmtree(brick_root, ignore_errors=True)

        minifigure_root.mkdir(parents=True, exist_ok=True)
        self.minifigure_paths.model.mkdir(parents=True, exist_ok=True)
        self.minifigure_paths.tokenizer.mkdir(parents=True, exist_ok=True)
        brick_root.mkdir(parents=True, exist_ok=True)
        self.brick_paths.model.mkdir(parents=True, exist_ok=True)
        self.brick_paths.tokenizer.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.image_transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        self.current_index_type: IndexType | None = None
        self.model: CLIPModel | None = None
        self.tokenizer: CLIPTokenizerFast | None = None
        self.dataset: Dataset | None = None
        self.vector_index: faiss.IndexFlatL2 | None = None

        self._load_default_index(startup_index, rebuild_indexes)

    def _load_default_index(self, index_type: IndexType = IndexType.MINIFIGURE, rebuild_indexes: bool = False) -> None:
        self.current_index_type = index_type
        if rebuild_indexes:
            logger.info("Deleting existing indexes...")
            if self.minifigure_paths.index.exists():
                self.minifigure_paths.index.unlink()
                logger.success("Deleted minifigure index")
            else:
                logger.info("Minifigure index does not exist")
            if self.brick_paths.index.exists():
                self.brick_paths.index.unlink()
                logger.success("Deleted brick index")
            else:
                logger.info("Brick index does not exist")

        logger.info(f"Loading model and tokenizer for {index_type}")
        self.model, self.tokenizer = self._load_model()
        logger.success(f"Loaded model and tokenizer for {index_type}")
        
        logger.info(f"Loading dataset for {index_type}")
        self.dataset = self._load_dataset()
        logger.success(f"Loaded dataset for {index_type}")

        logger.info(f"Loading vector index for {index_type}")
        self.vector_index = self._load_vector_index()
        logger.success(f"Loaded vector index for {index_type}")

    def _free_resources(self):
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
    
    def _load_model(self) -> tuple[CLIPModel, CLIPTokenizerFast]:
        hf_repo_id = (
            HF_MINIFIGURE_MODEL_ID 
            if self.current_index_type == IndexType.MINIFIGURE 
            else HF_BRICKS_MODEL_ID
        )
        model_cache = (
            self.minifigure_paths.model
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_paths.model
        )
        tokenizer_cache = (
            self.minifigure_paths.tokenizer 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_paths.tokenizer
        )
        model = CLIPModel.from_pretrained(
            hf_repo_id, 
            cache_dir=model_cache, 
            torch_dtype=self.model_dtype
        ).to(self.device)

        tokenizer = CLIPTokenizerFast.from_pretrained(
            hf_repo_id,
            cache_dir=tokenizer_cache
        )
        return model, tokenizer

    def _load_dataset(self) -> Dataset:
        hf_dataset_id = (
            HF_MINIFIGURES_DATASET_ID 
            if self.current_index_type == IndexType.MINIFIGURE 
            else HF_BRICKS_DATASET_ID
        )
        dataset_cache = (
            self.minifigure_paths.dataset 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_paths.dataset
        )
        dataset = load_dataset(
            hf_dataset_id,
            cache_dir=dataset_cache,
            split="train",
            download_mode="reuse_cache_if_exists",
        )       
        return dataset

    def _load_vector_index(self) -> faiss.IndexFlatL2:
        index_path = (
            self.minifigure_paths.index 
            if self.current_index_type == IndexType.MINIFIGURE 
            else self.brick_paths.index
        )
        if not index_path.exists():
            self._build_index(index_path)
        vector_index = faiss.read_index(str(index_path))
        return vector_index
        
    def _switch_index_type(self, index_type: IndexType) -> None:
        if self.current_index_type == index_type:
            return

        logger.info(f"Switching {self.current_index_type} to {index_type}")
        self._free_resources()
        logger.info(f"Resources for {self.current_index_type} freed")

        self.current_index_type = index_type

        self.model, self.tokenizer = self._load_model()
        logger.info(f"Loaded model and tokenizer for {index_type} dataset")
        self.dataset = self._load_dataset()
        logger.info(f"Loaded dataset for {index_type} dataset")
        self.vector_index = self._load_vector_index()
        logger.info(f"Loaded vector index for {index_type} dataset")

        logger.success(f"Switched to {index_type} dataset and model")

    def _build_index(
        self,
        index_path: Path
    ) -> None:
        vector_index = faiss.IndexFlatIP(512)
        for row in tqdm.tqdm(self.dataset, desc=f"Building index for {self.current_index_type.value}"):
            image = row["image"]
            if (im_mode := image.mode) != "RGB":
                # Some images in brick dataset are grayscale
                # this should handle that and more cases
                if im_mode in ["1", "L", "P", "RGBA"]: 
                    image = image.convert("RGB")
                else:
                    continue
            
            # Only improvement would be to use a batch and then using 
            # the gpu would be beneficial, increasing however code complexity.
            # Since is not a performance critical operation, we can keep it simple
            patches = self.image_transform(image).unsqueeze(0).to(self.device)
            image_features = self.model.get_image_features(patches).detach().cpu().numpy()
            vector_index.add(image_features)
        
        # save the index
        faiss.write_index(vector_index, str(index_path))

    def query(self, query: str | PILImage, top_k: int, index_type: IndexType = None) -> list[QueryResult]:
        self._switch_index_type(index_type)

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
            embeddings = self.model.get_image_features(input_tokens).detach().cpu().numpy()

        D, I = self.vector_index.search(embeddings, top_k)
        results = []

        for i, idx in enumerate(I[0]):
            image = self.dataset[int(idx)]["image"]
            
            # additional_info = StringIO()
            # for key, value in self.dataset[int(idx)].items():
            #     if key != "image":
            #         additional_info.write(f"{key}: {value}\n")
            
            results.append(
                QueryResult(
                    idx=str(idx),
                    image=image,
                    # additional_info=additional_info.getvalue(),
                    distance=D[0][i]
                )
            )

        return results
    
    def get_image_info(self, idx: int) -> dict:
        additional_info = dict()
        for key, value in self.dataset[idx].items():
            if key != "image":
                additional_info[key] = value
        return additional_info
            