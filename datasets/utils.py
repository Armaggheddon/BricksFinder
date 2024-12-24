from pathlib import Path
import os
import urllib.request
import json
import time

from PIL import Image
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_exceptions
import pandas as pd
from loguru import logger

GEMINI_MINIFIGURE_SYSTEM_PROMPT = (
    "You are a precise and detailed image captioning assistant. Your task is "
    "to describe LEGO minifigures in a single sentence, focusing on their "
    "unique features, such as attire, accessories, facial expression, "
    "and theme. Avoid generic terms and aim for specificity, "
    "while remaining concise. Caption the image"
)
GEMINI_BRICK_SYSTEM_PROMPT = ("")

GEMINI_API_KEYS_PATH = Path(__file__).parent / "gemini_api_keys.json"

class CaptionGenerator:
    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        images_path: Path = None,
        captions_path: Path = None,
        type: str = "minifigure"
    ) -> None:
        if not images_path.exists():
            logger.error(f"Images path does not exist: {images_path}")
            raise FileNotFoundError(f"Images path does not exist: {images_path}")
        if not captions_path.exists():
            logger.error(f"Captions path does not exist: {captions_path}")
            raise FileNotFoundError(f"Captions path does not exist: {captions_path}")
        
        self.dataframe = dataframe
        self.images_path = images_path
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

        self.prompt = (
            GEMINI_MINIFIGURE_SYSTEM_PROMPT
            if type == "minifigure"
            else GEMINI_BRICK_SYSTEM_PROMPT
        )
    
    def caption(self, resume: bool = True):
        start_offset = 0
        
        if resume:
            for _, row in self.dataframe.iterrows():
                caption_file_path = self.captions_path / f"{row['idx']}.json"
                if not caption_file_path.exists():
                    break
                start_offset += 1
        
        logger.info(f"Setting offset to {start_offset}")

        dataframe_iter = self.dataframe.iterrows()
        for _ in range(start_offset):
            next(dataframe_iter)
        
        pbar = tqdm(
            dataframe_iter, 
            desc=f"Captioned {start_offset} images",
            total=len(self.dataframe),
            initial=start_offset
        )

        for count, row in enumerate(pbar):
            img_path = self.images_path / f"{row['idx']}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path)
            self._caption_img(row["idx"], img)

            pbar.set_description(f"Captioned {count} images")
        
        logger.success("Finished generating captions.")

    def _caption_img(self, idx: int, img: Image.Image):
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
                    contents=[self.prompt, img],
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
            

def download_images(
    dataframe: pd.DataFrame,
    image_column: str,
    download_path: Path,
    resume: bool = True
): 
    max_retries = 5
    start = 0

    if resume: 
        for _, row in dataframe.iterrows():
            image_path = download_path / f"{row['idx']}.jpg"
            if not image_path.exists():
                break
            start += 1

    logger.info(f"Setting start to {start}")

    # consume start rows from the dataframe iterator
    dataframe_iter = dataframe.iterrows()
    for _ in range(start):
        next(dataframe_iter)

    pbar = tqdm(
        dataframe_iter, 
        desc="Downloaded xxx, failed yyy",
        total=len(dataframe),
        initial=start
    )
    downloaded_count = start
    failed_count = 0
    for _, row in pbar:
        curr_retries = 0
        is_downloaded = False

        image_path = download_path / f"{row['idx']}.jpg"
        while not is_downloaded and curr_retries < max_retries:
            try:
                image = Image.open(urllib.request.urlopen(row[image_column]))
                image.save(image_path, format="JPEG")
                is_downloaded = True
            except Exception as e:
                curr_retries += 1
        
        if is_downloaded:
            downloaded_count += 1
        else:
            failed_count += 1
        
        pbar.set_description(f"Downloaded {downloaded_count}, failed {failed_count}")

def touch_folder(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)
    logger.success(f"Created folder: {folder_path}")   
