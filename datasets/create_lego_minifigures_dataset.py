import os
from pathlib import Path
import json

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

import utils


# Download the original CSVs from rebrickable.
DOWNLOAD_CSVS = False

# Convert the CSVs to parquet files.
CONVERT_CSVS_TO_PARQUET = False

# Create a temporary parquet file with all the columns. This will
# be used to assign the id to the downloaded images.
CREATE_ROOT_PARQUET = False

# Using the temporary parquet file, will download all the images
# and overwrite the parquet file with only the rows that have images.
DOWNLOAD_IMAGES = False

# Generate captions for the images using the Gemini model.
CREATE_GEMINI_CAPTIONS = False

# Create the final dataset parquet files.
CREATE_DATASET_PARQUET = False

# Create a zip file with the final dataset
# parquet files ready to be uploaded to the hub. 
CREATE_ZIP = False


THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_ROOT = THIS_PATH / "raw_data"
MINIFIGURES_DATASET_ROOT = THIS_PATH / "lego_minifigures_captions"
DATASET_IMAGES_PATH = MINIFIGURES_DATASET_ROOT / "images"
DATASET_PARQUET_PATH = MINIFIGURES_DATASET_ROOT / "data"
DATASET_CAPTIONS_PATH = MINIFIGURES_DATASET_ROOT / "captions"

MINIFIGS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz?1732432083.3254244",
    "zip_filename": "minifigs.csv.gz",
    "filename": "minifigs.csv",
    "parquet_filename": "minifigs.parquet"
}
INVENTORY_MINIFIGS_CSV = {
    "url": "https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz?1732432105.8860035",
    "zip_filename": "inventory_minifigs.csv.gz",
    "filename": "inventory_minifigs.csv",
    "parquet_filename": "inventory_minifigs.parquet"
}

def create_root_parquet():
    """
    Creates a temporary parquet file with all the columns. This will
    be used to assign the id to the downloaded images.
    """
    minifigs = pd.read_parquet( RAW_DATA_ROOT / MINIFIGS_CSV["parquet_filename"],)
    logger.info(f"Loaded: {MINIFIGS_CSV['filename']}")
    inventory_minifigs = pd.read_csv(RAW_DATA_ROOT / INVENTORY_MINIFIGS_CSV["parquet_filename"])
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
    result_df.to_parquet(MINIFIGURES_DATASET_ROOT / "minifigures_no_img.parquet")

def remove_missing_rows(dataframe: pd.DataFrame):
    """
    Uses the dataframe created from create_tmp_parquet and removes 
    the rows with missing images. The final parquet file will only
    contain rows with images available. Relies on the fact that
    the images are downloaded in the DATASET_IMAGES_PATH folder.
    """
    missing_images = len(dataframe) - len(os.listdir(DATASET_IMAGES_PATH))
    logger.info(f"Missing images: {missing_images}")

    # remove rows with missing images
    dataframe = dataframe[
        dataframe["file_name"].apply(lambda x: x in os.listdir(DATASET_IMAGES_PATH))]
    logger.info(
        f"Removed rows with missing images, new shape: {dataframe.shape}")

    # save dataframe as parquet
    dataframe.to_parquet(DATASET_PARQUET_PATH / "minifigures_no_img.parquet")


def create_parquet(dataframe: pd.DataFrame):
    """
    Creates a parquet file from the dataframe. 
    The final parquet file will only contain the following columns:
    - fig_num
    - image
    - short_caption
    - caption

    Requires that the images are downloaded in the DATASET_IMAGES_PATH folder
    and the captions are generated in the DATASET_CAPTIONS_PATH folder.
    """
    table_rows = []
    json_idx = 0

    for img_idx, row in tqdm.tqdm(dataframe.iterrows()):
        image_name = f"{img_idx}.jpg"
        image_path = DATASET_IMAGES_PATH / image_name
        caption_name = f"{json_idx}.json"
        caption_path = DATASET_CAPTIONS_PATH / caption_name
        
        image_bytes = b""
        # if image does not exist, someone has deleted it,
        # previous steps should have removed the row from the dataframe
        image_bytes = image_path.read_bytes() 
        with open(caption_path) as f:
            caption_data = json.load(f)
        
        row_data = {
            "fig_num": row["fig_num"],
            "image": {"bytes": image_bytes, "path": f"{img_idx}.jpg"},
            "short_caption": row["name"],
            "caption": caption_data["caption"]
        }
        table_rows.append(row_data)

        json_idx += 1
    
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
        # calculate the end index for this chunk
        end_idx = min(start_idx + max_rows_per_file, total_rows)

        # slice the table to create a chunk
        chunk = parquet_table.slice(start_idx, end_idx - start_idx)

        # write the chunk to a Parquet file
        output_file = DATASET_PARQUET_PATH / f"train-{file_idx:05d}-of-{num_files:05d}.parquet"
        pq.write_table(chunk, output_file)

        print(f"Written {output_file} with rows {start_idx} to {end_idx - 1}")
        start_idx = end_idx
        file_idx += 1


if __name__ == "__main__":
    utils.touch_folder(RAW_DATA_ROOT)
    utils.touch_folder(MINIFIGURES_DATASET_ROOT)
    utils.touch_folder(DATASET_CAPTIONS_PATH)
    utils.touch_folder(DATASET_IMAGES_PATH)
    utils.touch_folder(DATASET_PARQUET_PATH)

    logger.info("Starting dataset creation with options:")
    logger.info(f"DOWNLOAD_CSVS: {DOWNLOAD_CSVS}")
    logger.info(f"CONVERT_CSVS_TO_PARQUET: {CONVERT_CSVS_TO_PARQUET}")
    logger.info(f"CREATE_ROOT_PARQUET: {CREATE_ROOT_PARQUET}")
    logger.info(f"DOWNLOAD_IMAGES: {DOWNLOAD_IMAGES}")
    logger.info(f"CREATE_GEMINI_CAPTIONS: {CREATE_GEMINI_CAPTIONS}")
    logger.info(f"CREATE_PARQUET: {CREATE_DATASET_PARQUET}")
    logger.info(f"CREATE_ZIP: {CREATE_ZIP}")

    if DOWNLOAD_CSVS:
        utils.download_csv_files(
            destination_path=RAW_DATA_ROOT,
            urls=[
                MINIFIGS_CSV["url"], 
                INVENTORY_MINIFIGS_CSV["url"]
            ],
            filenames=[
                MINIFIGS_CSV["zip_filename"], 
                INVENTORY_MINIFIGS_CSV["zip_filename"]
            ]
        )
    
    if CONVERT_CSVS_TO_PARQUET:
        utils.csv_to_parquet(
            csv_files=[
                RAW_DATA_ROOT / MINIFIGS_CSV["zip_filename"], 
                RAW_DATA_ROOT / INVENTORY_MINIFIGS_CSV["zip_filename"]
            ],
            parquet_files=[
                RAW_DATA_ROOT / MINIFIGS_CSV["parquet_filename"], 
                RAW_DATA_ROOT / INVENTORY_MINIFIGS_CSV["parquet_filename"]
            ]
        )

    if CREATE_ROOT_PARQUET:
        create_root_parquet()

    if DOWNLOAD_IMAGES:
        utils.download_images(
            dataframe=pd.read_parquet(
                MINIFIGURES_DATASET_ROOT / "minifigures_no_img.parquet"
            ),
            image_column="img_url",
            download_path=DATASET_IMAGES_PATH,
            image_name_template="{id}"
        )

        remove_missing_rows(
            pd.read_parquet(
                MINIFIGURES_DATASET_ROOT / "minifigures_no_img.parquet"
            )
        )

    if CREATE_GEMINI_CAPTIONS:
        utils.CaptionGenerator(
            dataset_name=DATASET_PARQUET_PATH,
            split="train",
            captions_path=DATASET_CAPTIONS_PATH,
        ).caption()
    
    if CREATE_DATASET_PARQUET:
        create_parquet(
            pd.read_parquet(
                MINIFIGURES_DATASET_ROOT / "minifigures_no_img.parquet"
            )
        )
    
    if CREATE_ZIP:
        utils.zip_dataset(DATASET_PARQUET_PATH)