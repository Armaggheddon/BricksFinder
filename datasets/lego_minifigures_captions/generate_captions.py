# This code assumes that the data folder contains the parquet files
from pathlib import Path
import os
import json

from loguru import logger
import google.generativeai as genai
import pandas as pd

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

PARQUET_PATH = THIS_PATH / "data"
CAPTIONS_PATH = THIS_PATH / "captions"
GEMINI_API_KEYS_PATH = THIS_PATH.parent / "gemini_api_keys.json"



if __name__ == "__main__":
    
    if not PARQUET_PATH.exists():
        logger.error(f"Path {PARQUET_PATH} does not exist.")
        exit(1)

    if not CAPTIONS_PATH.exists():
        CAPTIONS_PATH.mkdir(exist_ok=True)

    # load the api_keys
    with open(GEMINI_API_KEYS_PATH) as f:
        api_keys = json.load(f)
    gemini_api_keys = api_keys["gemini_api_keys"]

    print(gemini_api_keys)
    
