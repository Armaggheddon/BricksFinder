import os
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd

import utils

# Create a temporary parquet file with all the columns. This will
# be used to assign the id to the downloaded images.
CREATE_ROOT_PARQUET = False

# Note that the images for this dataset are A LOT!
# Roughly 1.3 million images, not counting for
# not working urls, etc. So setting this to True
# will take a lot of time and space.
DOWNLOAD_IMAGES = False

# Generate captions for the images using the Gemini model.
CREATE_GEMINI_CAPTIONS = False

# Create the final dataset parquet files.
CREATE_AND_UPLOAD_DATASET = False


THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_ROOT = THIS_PATH / "raw_data"
DATASET_ROOT = THIS_PATH / "lego_brick_captions"
DATASET_IMAGES_PATH = DATASET_ROOT / "images"
DATASET_PARQUET_PATH = DATASET_ROOT / "data"
DATASET_CAPTIONS_PATH = DATASET_ROOT / "captions"

INVENTORY_PARTS_PARQUET = "inventory_parts.parquet"
PARTS_PARQUET = "parts.parquet"
COLORS_PARQUET = "colors.parquet"
PART_CATEGORIES_PARQUET = "part_categories.parquet"

def create_root_parquet():
    """
    Create a parquet file with all the data from the CSVs
    this will parquet will be used to create the actual final dataset. 
    It acts as a middle step to avoid reading multiple CSVs when creating
    the final dataset.
    """
    inventory_parts = pd.read_parquet(RAW_DATA_ROOT / INVENTORY_PARTS_PARQUET)
    logger.info(f"Loaded: {INVENTORY_PARTS_PARQUET}")
    parts = pd.read_parquet(RAW_DATA_ROOT / PARTS_PARQUET)
    logger.info(f"Loaded: {PARTS_PARQUET}")
    colors = pd.read_parquet(RAW_DATA_ROOT / COLORS_PARQUET)
    logger.info(f"Loaded: {COLORS_PARQUET}")
    part_categories = pd.read_parquet(RAW_DATA_ROOT / PART_CATEGORIES_PARQUET)
    logger.info(f"Loaded: {PART_CATEGORIES_PARQUET}")

def remove_missing_rows(dataframe: pd.DataFrame):
    """
    Remove rows from the dataframe that have missing images.
    """
    pass

def upload_dataset_to_hf(dataframe: pd.DataFrame):
    """
    Upload the dataset to the Hugging Face Datasets Hub.
    """
    pass


if __name__ == "__main__":
    utils.touch_folder(RAW_DATA_ROOT)
    utils.touch_folder(DATASET_ROOT)
    utils.touch_folder(DATASET_IMAGES_PATH)
    utils.touch_folder(DATASET_PARQUET_PATH)
    utils.touch_folder(DATASET_CAPTIONS_PATH)

    logger.info("Starting dataset creation with options:")
    logger.info(f"CREATE_ROOT_PARQUET: {CREATE_ROOT_PARQUET}")
    logger.info(f"DOWNLOAD_IMAGES: {DOWNLOAD_IMAGES}")
    logger.info(f"CREATE_GEMINI_CAPTIONS: {CREATE_GEMINI_CAPTIONS}")
    logger.info(f"CREATE_PARQUET: {CREATE_AND_UPLOAD_DATASET}")

    try:
        with open(THIS_PATH  / ".env", "r") as f:
            env_vars = f.read().splitlines()
            for env_var in env_vars:
                key, value = env_var.split("=")
                os.environ[key] = value.strip("\"")
        _ = os.environ["HF_TOKEN"]
        logger.info("Loaded HF_TOKEN")
        _ = os.environ["GEMINI_API_KEY"]
        logger.info("Loaded GEMINI_API_KEY")
    except FileNotFoundError:
        logger.error("No .env file found, did you forget to rename example.env?")
        logger.info("Setting CREATE_AND_UPLOAD_DATASET to False")
        CREATE_AND_UPLOAD_DATASET = False
    except KeyError:
        logger.error("Check if both HF_TOKEN and GEMINI_API_KEY have been set in the .env file")
        logger.warning("Setting CREATE_AND_UPLOAD_DATASET and CREATE_GEMINI_CAPTIONS to False")
        CREATE_AND_UPLOAD_DATASET = False
        CREATE_GEMINI_CAPTIONS = False

    exit()

    if CREATE_ROOT_PARQUET:
        create_root_parquet()

    if DOWNLOAD_IMAGES:
        utils.download_images(
            dataframe=pd.read_parquet(
                DATASET_ROOT / "lego_bricks_no_img.parquet"),
            image_column="img_url",
            download_path=DATASET_IMAGES_PATH
        )

        remove_missing_rows(
            pd.read_parquet(
                DATASET_ROOT / "lego_bricks_no_img.parquet"
            )
        )

    if CREATE_GEMINI_CAPTIONS:
        utils.CaptionGenerator(
            dataframe=pd.read_parquet(
                DATASET_ROOT / "lego_bricks_no_img.parquet"
            ),
            images_path=DATASET_IMAGES_PATH,
            captions_path=DATASET_CAPTIONS_PATH,
            type="brick"
        ).caption()

    if CREATE_AND_UPLOAD_DATASET:
        upload_dataset_to_hf(
            pd.read_parquet(
                DATASET_ROOT / "lego_bricks_no_img.parquet"
            )
        )
