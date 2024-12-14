from pathlib import Path
import os
import urllib.request
import json
from io import BytesIO
import time

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_exceptions
import pandas as pd
from loguru import logger

GEMINI_SYSTEM_PROMPT = (
    "You are a precise and detailed image captioning assistant. Your task is "
    "to describe LEGO minifigures in a single sentence, focusing on their "
    "unique features, such as attire, accessories, facial expression, "
    "and theme. Avoid generic terms and aim for specificity, "
    "while remaining concise. Caption the image"
)

GEMINI_API_KEYS_PATH = Path(__file__).parent / "gemini_api_keys.json"

class CaptionGenerator:
    def __init__(
        self,
        dataset_name: Path | str = None,
        split: str = None,
        images_path: Path = None,
        captions_path: Path = None,

    ) -> None:
        
        if dataset_name is not None:
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
        elif images_path is not None:
            images = list(images_path.glob("*.jpg"))
            images.sort()
            self.dataset = [{"image": img} for img in images]
            logger.info(f"Loaded {len(images)} images.")
        else:
            logger.error("Either dataset_name or images_path must be provided.")
            raise ValueError("Either dataset_name or images_path must be provided, not both.")

        self.dataset_iterator = iter(self.dataset)

        self.captions_path = captions_path
        with open(GEMINI_API_KEYS_PATH) as f:
            api_keys = json.load(f)
        self.gemini_api_keys = api_keys["gemini_api_keys"]

        logger.info(f"Loaded {len(self.gemini_api_keys)} Gemini API keys.")

        self.gemini_api_errors = [
            genai_types.BrokenResponseError,
            genai_types.IncompleteIterationError,
            genai_types.StopCandidateException,
            google_exceptions.ResourceExhausted,
            google_exceptions.RetryError
        ]
        self.api_key_idx = 0
    
    def caption(self):
        offset_idx = 0
        file_names = self.captions_path.glob("*.json")
        for file_name in file_names:
            idx = int(file_name.stem)
            if idx >= offset_idx:
                offset_idx = idx + 1
        
        for _ in range(offset_idx):
            next(self.dataset_iterator)
        
        logger.info(f"Starting from index {offset_idx}")
        
        pbar = tqdm(
            self.dataset_iterator, 
            desc=f"Captioned {offset_idx} images"
        )

        for idx, row in enumerate(pbar, start=offset_idx):
            img = row["image"]
            if isinstance(img, dict):
                img = Image.open(BytesIO(img["bytes"]))
            elif isinstance(img, Path):
                img = Image.open(img)
            
            self._caption_img(idx, img)

            pbar.set_description(f"Captioned {idx} images")
        
        logger.success("Finished generating captions.")

    def _caption_img(self, idx, img):
        genai.configure(api_key=self.gemini_api_keys[self.api_key_idx])
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=75,
            temperature=1.0,
            top_p=0.95
        )
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-002",
            generation_config=generation_config
        )

        retry_count = 0
        max_retries = 3
        is_success = False
        while not is_success and retry_count < max_retries:
            try:
                response = model.generate_content(
                    contents=[GEMINI_SYSTEM_PROMPT, img],
                )
                if not response.text:
                    raise Exception("Empty response")
                
                caption_file_name = f"{idx}.json"
                caption_file_path = self.captions_path / caption_file_name
                caption_file_content = {
                    "image_idx": idx,
                    "caption": response.text.strip(),
                }
                with open(caption_file_path, "w") as f:
                    f.write(json.dumps(caption_file_content))
                is_success = True
            except Exception as e:
                if type(e) in self.gemini_api_errors:
                    time.sleep(10)
                    # rotate api key
                    self.api_key_idx = (self.api_key_idx + 1) % len(self.gemini_api_keys)
                    genai.configure(api_key=self.gemini_api_keys[self.api_key_idx])
                    logger.error(f"Failed generating caption. Rotating to API[{self.api_key_idx}]")
                else:
                    retry_count += 1
        
        if retry_count == max_retries and not is_success:
            logger.error(f"Failed to generate caption for {idx}")
            caption_file_name = f"{idx}.json"
            caption_file_path = self.captions_path / caption_file_name
            with open(caption_file_path, "w") as f:
                f.write(json.dumps({
                    "image_idx": idx,
                    "caption": ""
                }))
            
    


def download_csv_files(
    destination_path: Path,
    urls: list[str],
    filenames: list[str]
): 
    for url, filename in zip(urls, filenames):
        file_path = destination_path / filename
        urllib.request.urlretrieve(
            url,
            file_path
        )
        logger.success(f"Downloaded: {filename}")

        os.system(f"gzip -d -f {file_path}")
        logger.success(f"Unzipped: {filename}")

def csv_to_parquet(
    csv_files: list[Path],
    parquet_files: list[Path]
):
    for csv_file, parquet_file in zip(csv_files, parquet_files):
        dataframe = pd.read_csv(csv_file)
        dataframe.to_parquet(parquet_file)
        logger.success(f"Converted: {csv_file} to {parquet_file}")


def download_images(
    dataframe: pd.DataFrame,
    image_column: str,
    download_path: Path,
    image_name_template: str
): 
    max_retries = 5
    for idx, row in dataframe.iterrows():
        curr_retries = 0
        is_downloaded = False

        image_path = download_path / f"{image_name_template.format(idx)}.jpg"

        while not is_downloaded and curr_retries < max_retries:
            try:
                image = Image.open(urllib.request.urlopen(row[image_column]))
                image.save(image_path, format="JPEG")
                is_downloaded = True
            except Exception as e:
                curr_retries += 1

def touch_folder(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)
    logger.success(f"Created folder: {folder_path}")   

def zip_dataset(data_path: Path):
    os.system(f"zip -r {data_path / 'data'}.zip {data_path}")
    logger.success(f"Zipped data to: {data_path / 'data'}.zip")