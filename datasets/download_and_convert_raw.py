import urllib.request
from pathlib import Path
import os

import pandas as pd
from loguru import logger


THIS_PATH = Path(__file__).parent
RAW_DATA_ROOT = THIS_PATH / "raw_data"
if not RAW_DATA_ROOT.exists():
    RAW_DATA_ROOT.mkdir(parents=True, exist_ok=True)

rebrickable_csvs = [
    {
        "url": "https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz?1732432083.3254244",
        "compressed_filename": "minifigs.csv.gz",
        "csv_filename": "minifigs.csv",
        "parquet_filename": "minifigs.parquet"
    },
    {
        "url": "https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz?1732432105.8860035",
        "compressed_filename": "inventory_minifigs.csv.gz",
        "csv_filename": "inventory_minifigs.csv",
        "parquet_filename": "inventory_minifigs.parquet"
    },
    {
        "url": "https://cdn.rebrickable.com/media/downloads/inventory_parts.csv.gz?1732432105.4099913",
        "compressed_filename": "inventory_parts.csv.gz",
        "csv_filename": "inventory_parts.csv",
        "parquet_filename": "inventory_parts.parquet"
    },
    {
        "url": "https://cdn.rebrickable.com/media/downloads/parts.csv.gz?1732432081.1133678",
        "compressed_filename": "parts.csv.gz",
        "csv_filename": "parts.csv",
        "parquet_filename": "parts.parquet"
    },
    {
        "url": "https://cdn.rebrickable.com/media/downloads/colors.csv.gz?1732432080.0813415",
        "compressed_filename": "colors.csv.gz",
        "csv_filename": "colors.csv",
        "parquet_filename": "colors.parquet"
    },
    {
        "url": "https://cdn.rebrickable.com/media/downloads/part_categories.csv.gz?1732432080.0853415",
        "compressed_filename": "part_categories.csv.gz",
        "csv_filename": "part_categories.csv",
        "parquet_filename": "part_categories.parquet"
    },
    {
        "url": "https://cdn.rebrickable.com/media/downloads/inventories.csv.gz?1734937682.781456",
        "compressed_filename": "inventories.csv.gz",
        "csv_filename": "inventories.csv",
        "parquet_filename": "inventories.parquet"
    }
]

def download_and_unzip_csv_files():
    for rebrickable_csv in rebrickable_csvs:
        compressed_file_path = RAW_DATA_ROOT / rebrickable_csv["compressed_filename"]
        urllib.request.urlretrieve(
            rebrickable_csv["url"],
            compressed_file_path
        )
        logger.success(f"Downloaded: {rebrickable_csv['compressed_filename']}")

        os.system(f"gzip -d -f {compressed_file_path}")
        logger.success(f"Unzipped: {rebrickable_csv['compressed_filename']}")

def csv_to_parquet():
    for rebrickable_csv in rebrickable_csvs:
        csv_file_path = RAW_DATA_ROOT / rebrickable_csv["csv_filename"]
        parquet_file_path = RAW_DATA_ROOT / rebrickable_csv["parquet_filename"]

        dataframe = pd.read_csv(csv_file_path)
        dataframe.to_parquet(parquet_file_path)
        logger.success(f"Converted: {csv_file_path} to {parquet_file_path}")


if __name__ == "__main__":
    download_and_unzip_csv_files()
    csv_to_parquet()

    