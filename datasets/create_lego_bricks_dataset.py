import os
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd

import utils

# Download the original CSVs from rebrickable.
# Note that the same data, is already available 
# as parquet files in the raw_data folder
# and was downloaded on 27 November 2024.
DOWNLOAD_CSVS = False
CREATE_ROOT_PARQUET = True
# Note that the images for this dataset are A LOT!
# Roughly 1.3 million images, not counting for
# not working urls, etc. So setting this to True
# will take a lot of time and space.
DOWNLOAD_IMAGES = False
CREATE_PARQUET = False
# Create a zip file with the final dataset
# parquet files ready to be uploaded to the hub. 
# The zip file can be omitted if not using a remote
# headless machine to upload the dataset.
CREATE_ZIP = False

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_ROOT = THIS_PATH / "raw_data"
LEGO_BRICKS_DATASET_ROOT = THIS_PATH / "lego_brick_captions"
DATASET_IMAGES_PATH = LEGO_BRICKS_DATASET_ROOT / "images"
DATASET_PARQUET_PATH = LEGO_BRICKS_DATASET_ROOT / "data"
DATASET_CAPTIONS_PATH = LEGO_BRICKS_DATASET_ROOT / "captions"

INVENTORY_PARTS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/inventory_parts.csv.gz?1732432105.4099913",
    "zip_filename": "inventory_parts.csv.gz",
    "filename": "inventory_parts.csv",
    "parquet_filename": "inventory_parts.parquet"
}
PARTS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/parts.csv.gz?1732432081.1133678",
    "zip_filename": "parts.csv.gz",
    "filename": "parts.csv",
    "parquet_filename": "parts.parquet"
}
COLORS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/colors.csv.gz?1732432080.0813415",
    "zip_filename": "colors.csv.gz",
    "filename": "colors.csv",
    "parquet_filename": "colors.parquet"
}
PART_CATEGORIES_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/part_categories.csv.gz?1732432080.0853415",
    "zip_filename": "part_categories.csv.gz",
    "filename": "part_categories.csv",
    "parquet_filename": "part_categories.parquet"
}


def download_dataset(download_path: Path):
    urllib.request.urlretrieve(
        INVENTORY_PARTS_CSV["url"],
        download_path / INVENTORY_PARTS_CSV["zip_filename"]
    )
    urllib.request.urlretrieve(
        PARTS_CSV["url"],
        download_path / PARTS_CSV["zip_filename"]
    )
    urllib.request.urlretrieve(
        COLORS_CSV["url"],
        download_path / COLORS_CSV["zip_filename"]
    )
    urllib.request.urlretrieve(
        PART_CATEGORIES_CSV["url"],
        download_path / PART_CATEGORIES_CSV["zip_filename"]
    )

def unzip_csv_files(files: list[Path]):
    for file_path in files:
        os.system(f"gzip -d -f {file_path}")
        logger.success(f"Unzipped: {file_path}")

def convert_csvs_to_parquet(raw_data_root: Path):
    inventory_parts = pd.read_csv(raw_data_root / INVENTORY_PARTS_CSV["filename"])
    parts = pd.read_csv(raw_data_root / PARTS_CSV["filename"])
    colors = pd.read_csv(raw_data_root / COLORS_CSV["filename"])
    part_categories = pd.read_csv(raw_data_root / PART_CATEGORIES_CSV["filename"])

    logger.success("Loaded all CSVs")

    inventory_parts.to_parquet(
        raw_data_root / INVENTORY_PARTS_CSV["parquet_filename"])
    parts.to_parquet(
        raw_data_root / PARTS_CSV["parquet_filename"])
    colors.to_parquet(
        raw_data_root / COLORS_CSV["parquet_filename"])
    part_categories.to_parquet(
        raw_data_root / PART_CATEGORIES_CSV["parquet_filename"])

    logger.success("Saved all parquet files")
    

def create_root_parquet(raw_data_root: Path, dataset_root: Path):
    """
    Create a parquet file with all the data from the CSVs
    this will parquet will be used to create the actual final dataset. 
    It acts as a middle step to avoid reading multiple CSVs when creating
    the final dataset.
    """
    inventory_parts = pd.read_parquet(
        raw_data_root / INVENTORY_PARTS_CSV["parquet_filename"])
    parts = pd.read_parquet(
        raw_data_root / PARTS_CSV["parquet_filename"])
    colors = pd.read_parquet(
        raw_data_root / COLORS_CSV["parquet_filename"])
    part_categories = pd.read_parquet(
        raw_data_root / PART_CATEGORIES_CSV["parquet_filename"])

    logger.success("Loaded all Parquet files")

    lego_bricks = inventory_parts.copy()

    # add columns from parts to lego_bricks, rename new columns as 
    # part_name, part_cat_id, and part_material
    lego_bricks = lego_bricks.merge(parts, on="part_num")
    lego_bricks = lego_bricks.rename(
        columns={
            "name": "part_name", 
            "part_cat_id": "part_cat_id", 
            "material": "part_material"
        }
    )

    # add columns from colors to lego_bricks, rename new columns as
    # color_rgb and is_transparent
    lego_bricks = lego_bricks.merge(colors, left_on="color_id", right_on="id")
    lego_bricks = lego_bricks.rename(
        columns={
            "id": "color_id",
            "name": "color_name",
            "rgb": "color_rgb",
            "is_trans": "is_transparent"
        }
    )
    # remove duplicate duplicate column "color_id"
    lego_bricks = lego_bricks.drop(columns=["color_id"])

    # add columns from part_categories to lego_bricks, rename new columns as
    # part_cat_name
    lego_bricks = lego_bricks.merge(
        part_categories, left_on="part_cat_id", right_on="id")
    lego_bricks = lego_bricks.rename(columns={"name": "part_cat_name"})
    lego_bricks = lego_bricks.drop(columns=["id"])

    logger.success("Merged all CSVs")

    lego_bricks.to_parquet(dataset_root / "lego_bricks_no_img.parquet")
    logger.success("Saved lego_bricks_no_img.parquet")

def create_folder_if_not_exists(folder_path: Path):
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        logger.success(f"Created folder: {folder_path}")

def check_parquets_exist(csv_root: Path):
    if not all(
        [
            (csv_root / INVENTORY_PARTS_CSV["parquet_filename"]).exists(),
            (csv_root / PARTS_CSV["parquet_filename"]).exists(),
            (csv_root / COLORS_CSV["parquet_filename"]).exists(),
            (csv_root / PART_CATEGORIES_CSV["parquet_filename"]).exists()
        ]
    ):
        logger.error("Some CSV files are missing.")
        return False
    return True

if __name__ == "__main__":
    create_folder_if_not_exists(RAW_DATA_ROOT)
    create_folder_if_not_exists(LEGO_BRICKS_DATASET_ROOT)
    create_folder_if_not_exists(DATASET_IMAGES_PATH)
    create_folder_if_not_exists(DATASET_PARQUET_PATH)

    logger.info("Starting dataset creation with options:")
    logger.info(f"DOWNLOAD_CSVS: {DOWNLOAD_CSVS}")
    logger.info(f"CREATE_ROOT_PARQUET: {CREATE_ROOT_PARQUET}")
    logger.info(f"DOWNLOAD_IMAGES: {DOWNLOAD_IMAGES}")
    logger.info(f"CREATE_PARQUET: {CREATE_PARQUET}")
    logger.info(f"CREATE_ZIP: {CREATE_ZIP}")

    if DOWNLOAD_CSVS:
        download_dataset(RAW_DATA_ROOT)
        unzip_csv_files([
            RAW_DATA_ROOT / INVENTORY_PARTS_CSV["zip_filename"],
            RAW_DATA_ROOT / PARTS_CSV["zip_filename"],
            RAW_DATA_ROOT / COLORS_CSV["zip_filename"],
            RAW_DATA_ROOT / PART_CATEGORIES_CSV["zip_filename"]
        ])
        convert_csvs_to_parquet(RAW_DATA_ROOT)

    if CREATE_ROOT_PARQUET:
        if not check_parquets_exist(RAW_DATA_ROOT):
            logger.error(
                "Use DOWNLOAD_CSVS=True to download the CSVs and "
                "convert them to parquet."
            )
            exit(1)
        create_root_parquet(RAW_DATA_ROOT, LEGO_BRICKS_DATASET_ROOT)

    if DOWNLOAD_IMAGES:
        pass

    if CREATE_PARQUET:
        pass

    if CREATE_ZIP:
        pass
