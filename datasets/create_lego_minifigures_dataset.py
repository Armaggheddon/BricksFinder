import os
from pathlib import Path
import json

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from datasets import Dataset

import utils


# Create a temporary parquet file with all the columns. This will
# be used to assign the id to the downloaded images.
CREATE_ROOT_PARQUET = False

# Using the temporary parquet file, will download all the images
# and overwrite the parquet file with only the rows that have images.
DOWNLOAD_IMAGES = False

# Generate captions for the images using the Gemini model.
CREATE_GEMINI_CAPTIONS = False

# Create the final dataset parquet files.
CREATE_AND_UPLOAD_DATASET = False


THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_ROOT = THIS_PATH / "raw_data"
DATASET_ROOT = THIS_PATH / "lego_minifigures_captions"
DATASET_IMAGES_PATH = DATASET_ROOT / "images"
DATASET_PARQUET_PATH = DATASET_ROOT / "data"
DATASET_CAPTIONS_PATH = DATASET_ROOT / "captions"

ROOT_PARQUET = "minifigures_no_img.parquet"
MINIFIGS_PARQUET = "minifigs.parquet"
INVENTORY_MINIFIGS_PARQUET = "inventory_minifigs.parquet"
INVENTORIES_PARQUET = "inventories.parquet"
INVENTORY_PARTS_PARQUET = "inventory_parts.parquet"

def create_root_parquet():
    """
    Creates a temporary parquet file with all the columns. This will
    be used to assign the id to the downloaded images.
    """
    minifigs = pd.read_parquet( RAW_DATA_ROOT / MINIFIGS_PARQUET)
    logger.info(f"Loaded: {MINIFIGS_PARQUET}")
    inventory_minifigs = pd.read_csv(RAW_DATA_ROOT / INVENTORY_MINIFIGS_PARQUET)
    logger.info(f"Loaded: {INVENTORY_MINIFIGS_PARQUET}")
    inventories = pd.read_csv(RAW_DATA_ROOT / INVENTORIES_PARQUET)
    logger.info(f"Loaded: {INVENTORIES_PARQUET}")
    inventory_parts = pd.read_csv(RAW_DATA_ROOT / INVENTORY_PARTS_PARQUET)
    logger.info(f"Loaded: {INVENTORY_PARTS_PARQUET}")

    # add inventory_id to minifigs matching fig_num
    step_1 = minifigs.copy()
    step_1["minifig_inventory_id"] = minifigs["fig_num"].apply(
        lambda x: inventory_minifigs[inventory_minifigs.fig_num == x].inventory_id.tolist()
    )

    # add inventory_id to minifigs matching fig_num
    step_2 = step_1.merge(
        inventories["set_num", "id"],
        left_on="fig_num",
        right_on="set_num",
    )
    step_2.rename(columns={"id": "part_inventory_id"}, inplace=True)
    step_2.drop(columns="set_num", inplace=True) # is same as fig_num

    # add part_num (list of ids) to minifigs matching part_inventory_id
    step_3 = step_2.copy()
    step_3["part_num"] = step_3.part_inventory_id.apply(
        lambda x: inventory_parts[inventory_parts.inventory_id == x].part_num.tolist()
    )
    # remove rows with null img_url
    step_3 = step_3[step_3.img_url.notnull()]

    # rename name column to short_caption
    step_3.rename(columns={"name": "short_caption"}, inplace=True)

    # add an idx column from 0 to len(df)
    step_3["idx"] = range(len(step_3))

    
    # reorder columns
    result_df = step_3[
        [
            "idx",
            "fig_num", 
            "short_caption", 
            "num_parts", 
            "img_url", 
            "minifig_inventory_id", 
            "part_inventory_id",
            "part_num"
        ]
    ]
    result_df.to_parquet(DATASET_ROOT / ROOT_PARQUET)

def remove_missing_rows(dataframe: pd.DataFrame):
    """
    Uses the dataframe created from create_tmp_parquet and removes 
    the rows with missing images. The final parquet file will only
    contain rows with images available. Relies on the fact that
    the images are downloaded in the DATASET_IMAGES_PATH folder.
    """
    missing_images = len(dataframe) - len(os.listdir(DATASET_IMAGES_PATH))
    logger.info(f"Missing images: {missing_images}")

    image_files = os.listdir(DATASET_IMAGES_PATH)
    # get the idx from the image file name
    image_idxs = [int( x.rsplit(".", maxsplit=1)) for x in image_files]

    # remove rows with missing images
    dataframe = dataframe[
        dataframe["idx"].isin(image_idxs)]
    logger.info(
        f"Removed rows with missing images, new shape: {dataframe.shape}")

    # save dataframe as parquet
    dataframe.to_parquet(DATASET_PARQUET_PATH / "minifigures_no_img.parquet")

def upload_dataset_to_hf(dataframe: pd.DataFrame):
    """
    Creates a parquet file from the dataframe. 
    The final parquet file will only contain the following columns:
    - fig_num 
    - short_caption 
    - num_parts 
    - img_url 
    - minifig_inventory_id 
    - part_inventory_id
    - part_num

    Requires that the images are downloaded in the DATASET_IMAGES_PATH folder
    and the captions are generated in the DATASET_CAPTIONS_PATH folder.
    """
    table_rows = []

    for _, row in tqdm.tqdm(dataframe.iterrows()):
        image_file_name = f"{row["idx"]}.jpg"
        image_file_path = DATASET_IMAGES_PATH / image_file_name
        if not image_file_path.exists():
            continue
        caption_file_name = f"{row["idx"]}.json"
        caption_file_path = DATASET_CAPTIONS_PATH / caption_file_name
        if not caption_file_path.exists():
            continue
        image_bytes = b""
        # if image does not exist, someone has deleted it,
        # previous steps should have removed the row from the dataframe
        image_bytes = image_file_path.read_bytes() 
        with open(caption_file_path) as f:
            caption_data = json.load(f)
        if not caption_data["caption"] or caption_data["caption"] == "":
            # if the caption is empty, skip the row
            continue
        
        row_data = {
            "image": {"bytes": image_bytes, "path": image_file_name}, 
            "short_caption": row["short_caption"], 
            "caption": caption_data["caption"], 
            "fig_num": row["fig_num"], 
            "num_parts": row["num_parts"],
            "minifig_inventory_id": row["minifig_inventory_id"], 
            "part_inventory_id": row["part_inventory_id"], 
            "part_num": row["part_num"]
        }
        table_rows.append(row_data)
    
    parquet_table = pa.Table.from_pylist(table_rows)
    
    hf_dataset = Dataset.from_parquet(parquet_table)
    hf_dataset.save_to_disk(DATASET_PARQUET_PATH)
    hf_dataset.push_to_hub(
        repo_id="armaggheddon97/lego_minifigure_captions",
        split="train",
        max_shard_size="200MB",
        commit_message="Initial commit",
        token=os.environ["HF_TOKEN"]
    )


if __name__ == "__main__":
    utils.touch_folder(RAW_DATA_ROOT)
    utils.touch_folder(DATASET_ROOT)
    utils.touch_folder(DATASET_CAPTIONS_PATH)
    utils.touch_folder(DATASET_IMAGES_PATH)
    utils.touch_folder(DATASET_PARQUET_PATH)

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
        _ = os.environ["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        logger.error("No .env or variables not found, did you forget to rename example.env?")
        logger.warning("Setting CREATE_AND_UPLOAD_DATASET and CREATE_GEMINI_CAPTIONS to False")
        CREATE_AND_UPLOAD_DATASET = False
        CREATE_GEMINI_CAPTIONS = False
    
    if CREATE_ROOT_PARQUET:
        create_root_parquet()

    if DOWNLOAD_IMAGES:
        utils.download_images(
            dataframe=pd.read_parquet(
                DATASET_ROOT / "minifigures_no_img.parquet"),
            image_column="img_url",
            download_path=DATASET_IMAGES_PATH
        )

        remove_missing_rows(
            pd.read_parquet(
                DATASET_ROOT / "minifigures_no_img.parquet"
            )
        )

    if CREATE_GEMINI_CAPTIONS:
        utils.CaptionGenerator(
            api_key=os.environ["GEMINI_API_KEY"],
            dataframe=pd.read_parquet(
                DATASET_ROOT / "minifigures_no_img.parquet"
            ),
            images_path=DATASET_IMAGES_PATH,
            captions_path=DATASET_CAPTIONS_PATH,
            type="minifigure"
        ).caption()
    
    if CREATE_AND_UPLOAD_DATASET:
        upload_dataset_to_hf(
            pd.read_parquet(
                DATASET_ROOT / "minifigures_no_img.parquet"
            )
        )