import os
import urllib.request
from pathlib import Path

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import tqdm


DOWNLOAD_IMAGES = False
DOWNLOAD_CSVS = False
CREATE_PARQUET = True
CREATE_ZIP = True

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_ROOT = THIS_PATH / "raw_data"
MINIFIGURES_DATASET_ROOT = THIS_PATH / "lego_minifigures_captions"
DATASET_IMAGES_PATH = MINIFIGURES_DATASET_ROOT / "images"
DATASET_PARQUET_PATH = MINIFIGURES_DATASET_ROOT / "data"

MINIFIGS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz?1732432083.3254244",
    "zip_filename": "minifigs.csv.gz",
    "filename": "minifigs.csv"
}
INVENTORY_MINIFIGS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz?1732432105.8860035",
    "zip_filename": "inventory_minifigs.csv.gz",
    "filename": "inventory_minifigs.csv"
}


def download_dataset(download_path: Path):
    """
    Downloads the dataset from the Rebrickable website
    """
    urllib.request.urlretrieve(
        MINIFIGS_CSV["url"], 
        download_path / MINIFIGS_CSV["zip_filename"]
    )
    logger.success(f"Downloaded: {MINIFIGS_CSV['zip_filename']}")
    urllib.request.urlretrieve(
        INVENTORY_MINIFIGS_CSV["url"], 
        download_path / INVENTORY_MINIFIGS_CSV["zip_filename"]
    )
    logger.success(f"Downloaded: {INVENTORY_MINIFIGS_CSV['zip_filename']}")

def unzip_csv_files(files: list[Path]):
    # unzipping transforms the filename
    # from filename.csv.gz to filename.csv
    for file_path in files:
        os.system(f"gzip -d -f {file_path}")
        logger.success(f"Unzipped: {file_path}")

def download_images(dataframe: pd.DataFrame, download_path: Path):
    """
    Downloads the images from the URLs in the dataframe 
    and saves them in the download_path folder
    as {row_id}.jpg
    """
    max_retries = 5
    for idx, row in tqdm.tqdm(dataframe.iterrows()):
        curr_retries = 0
        is_downloaded = False
        while not is_downloaded and curr_retries < max_retries:
            try:
                image = Image.open(urllib.request.urlopen(row["img_url"]))
                image.save(download_path / f"{idx}.jpg", format="JPEG")
                is_downloaded = True
            except Exception as e:
                logger.error(f"Error downloading image: {row['img_url']}")
                logger.error(f"Error: {e}")
                curr_retries += 1

def create_dataframe():  
    minifigs = pd.read_csv( RAW_DATA_ROOT / MINIFIGS_CSV["filename"],)
    logger.info(f"Loaded: {MINIFIGS_CSV['filename']}")
    inventory_minifigs = pd.read_csv(RAW_DATA_ROOT / INVENTORY_MINIFIGS_CSV["filename"])
    logger.info(f"Loaded: {INVENTORY_MINIFIGS_CSV['filename']}")
    result_df = minifigs.copy()
    result_df["inventory_id"] = minifigs["fig_num"].apply(
        lambda x: inventory_minifigs[inventory_minifigs["fig_num"] == x]["inventory_id"].tolist()
    )
    logger.info("Added inventory_id column to minifigs dataframe")
    result_df["file_name"] = result_df.index.astype(str) + ".jpg"
    result_df["name"] = result_df["name"].apply(
        lambda x: x.strip() )
    logger.info("Merged minifigs and inventory_minifigs")
    # reorder columns
    result_df = result_df[
        [
            "file_name", 
            "img_url", 
            "fig_num", 
            "name", 
            "num_parts", 
            "inventory_id"
        ]
    ]
    
    if DOWNLOAD_IMAGES:
        download_images(result_df, DATASET_IMAGES_PATH)
    
    missing_images = len(result_df) - len(os.listdir(DATASET_IMAGES_PATH))
    logger.info(f"Missing images: {missing_images}")

    # remove rows with missing images
    result_df = result_df[
        result_df["file_name"].apply(lambda x: x in os.listdir(DATASET_IMAGES_PATH))]
    logger.info(
        f"Removed rows with missing images, new shape: {result_df.shape}")

    # save dataframe as parquet
    result_df.to_parquet(DATASET_PARQUET_PATH / "minifigures_no_img.parquet")


def create_parquet(dataframe: pd.DataFrame):
    """
    Creates a parquet file from the dataframe. 
    The final parquet file will only contain the following columns:
    - fig_num
    - image
    - short_caption
    - long_caption [for now omitted, when generated it will be added]
    """
    table_rows = []

    for idx, row in tqdm.tqdm(dataframe.iterrows()):
        row = row.to_dict()
        image_name = f"{idx}.jpg"
        image_path = DATASET_IMAGES_PATH / image_name
        image_bytes = b""
        if image_path.exists():
            image_bytes = image_path.read_bytes()
        row.pop("file_name")
        row.pop("img_url")
        row.pop("inventory_id")
        row.pop("num_parts")
        row["short_caption"] = row.pop("name")
        row["image"] = {"bytes": image_bytes, "path": f"{idx}.jpg"}
        table_rows.append(row)
    
    parquet_table = pa.Table.from_pylist(table_rows)
    # save parquet table in multiple files with max size of 200MB
    total_rows = parquet_table.num_rows
    row_size_bytes = parquet_table.nbytes / total_rows
    max_rows_per_file = int((200 * 1024 * 1024) // row_size_bytes)

    # calculate number of output files
    num_files = (total_rows // max_rows_per_file) + 1

    start_idx = 0
    file_idx = 0
    while start_idx < total_rows:
        # Calculate the end index for this chunk
        end_idx = min(start_idx + max_rows_per_file, total_rows)

        # Slice the table to create a chunk
        chunk = parquet_table.slice(start_idx, end_idx - start_idx)

        # Write the chunk to a Parquet file
        output_file = DATASET_PARQUET_PATH / f"minifigures-{file_idx:05d}-of-{num_files:05d}.parquet"
        pq.write_table(chunk, output_file)

        print(f"Written {output_file} with rows {start_idx} to {end_idx - 1}")
        start_idx = end_idx
        file_idx += 1

def create_data_zip(data_path: Path):
    os.system(f"zip -r {data_path / 'data'}.zip {data_path}")
    logger.success(f"Zipped data to: {data_path / 'data'}.zip")

def create_folder_if_not_exists(folder_path: Path):
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        logger.success(f"Created folder: {folder_path}")

if __name__ == "__main__":
    create_folder_if_not_exists(RAW_DATA_ROOT)
    create_folder_if_not_exists(MINIFIGURES_DATASET_ROOT)
    create_folder_if_not_exists(DATASET_IMAGES_PATH)
    create_folder_if_not_exists(DATASET_PARQUET_PATH)

    logger.info("Starting dataset creation with options:")
    logger.info(f"DOWNLOAD_IMAGES: {DOWNLOAD_IMAGES}")
    logger.info(f"DOWNLOAD_CSVS: {DOWNLOAD_CSVS}")
    logger.info(f"CREATE_PARQUET: {CREATE_PARQUET}")
    logger.info(f"CREATE_ZIP: {CREATE_ZIP}")

    if DOWNLOAD_CSVS:
        download_dataset(RAW_DATA_ROOT)
        unzip_csv_files([
            RAW_DATA_ROOT / MINIFIGS_CSV["zip_filename"],
            RAW_DATA_ROOT / INVENTORY_MINIFIGS_CSV["zip_filename"]
        ])
        create_dataframe()
    
    if CREATE_PARQUET:
        dataframe = pd.read_parquet(
            MINIFIGURES_DATASET_ROOT / "minifigures_no_img.parquet")
        create_parquet(dataframe)
    
    if CREATE_ZIP:
        create_data_zip(DATASET_PARQUET_PATH)