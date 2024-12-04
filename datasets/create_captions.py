from pathlib import Path
import json
import time
from io import BytesIO

from PIL import Image
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
import google.generativeai as genai

GEMINI_SYSTEM_PROMPT = (
    "You are a precise and detailed image captioning assistant. Your task is "
    "to describe LEGO minifigures in a single sentence, focusing on their "
    "unique features, such as attire, accessories, facial expression, "
    "and theme. Avoid generic terms and aim for specificity, "
    "while remaining concise."
)
GEMINI_CAPTION_PROMPT = "describe the image"


class CaptionGenerator:
    def __init__(
        self,
        dataset_name: Path | str,
        dataset_path: Path,
        api_keys_path: Path,
        split: str = "",
    ) -> None:
        self.is_local = isinstance(dataset_name, Path)
        if isinstance(dataset_name, str):
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True)
            logger.info("Loaded dataset from Hugging Face datasets.")
        else:
            # dataset is from local parquet files
            # retrieve all parquet files in the directory
            parquet_files = [
                str(parquet_path)
                for parquet_path in list(dataset_name.glob("*.parquet"))
            ]
            parquet_files.sort() # sort alphabetically
            
            logger.info(f"Found {len(parquet_files)} parquet files.")
            self.dataset = load_dataset(
                path=str(dataset_name), 
                data_files=parquet_files,
                split="train",
                streaming=True
            )
            logger.info("Loaded dataset from local parquet files.")

        self.dataset_path = dataset_path
        with open(api_keys_path) as f:
            api_keys = json.load(f)
        self.gemini_api_keys = api_keys["gemini_api_keys"]

        self.captions_path = dataset_path / "captions"
        if not self.captions_path.exists():
            self.captions_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loaded {len(self.gemini_api_keys)} Gemini API keys.")

    def _api_keys_generator(self):
        current_api_key = 0
        while True:
            logger.success(f"Rotated to API key {current_api_key}")
            yield self.gemini_api_keys[current_api_key]
            current_api_key = (current_api_key + 1) % len(self.gemini_api_keys)
    
    def generate_captions(self):

        api_key_gen = self._api_keys_generator()
        genai.configure(api_key=next(api_key_gen))
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=GEMINI_CAPTION_PROMPT)

        caption_file_template = "{idx}_caption.json"
        pbar = tqdm(
            iter(self.dataset), 
            desc=f"Captioned {caption_file_template.format(idx=0)}"
        )

        for idx, row in enumerate(pbar):
            is_success = False
            
            img = row["image"]
            if isinstance(img, dict):
                # if dataset is local,
                # the image is a dictionary with "bytes" key
                img = Image.open(BytesIO(img["bytes"]))
            while not is_success:
                try:
                    response = model.generate_content(
                        [GEMINI_CAPTION_PROMPT, img]
                    )
                    if not response.text:
                        raise Exception("Empty response")

                    caption_file_name = caption_file_template.format(idx=idx)
                    caption_file_path = self.captions_path / caption_file_name
                    caption_file_content = {
                        "image_idx": idx,
                        "caption": response.text,
                    }
                    with open(caption_file_path, "w") as f:
                        f.write(json.dumps(caption_file_content))
                    is_success = True
                except Exception as e:
                    logger.error(f"Error generating caption for {idx}: {e}")
                    time.sleep(0.1)
                    genai.configure(api_key=next(api_key_gen))
            
            pbar.set_description(
                f"Captioned {caption_file_template.format(idx=idx)}")
            






        
