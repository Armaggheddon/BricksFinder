import os
from pathlib import Path
import urllib.request
import json

import tqdm
from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

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
CREATE_AND_UPLOAD_DATASET = True


THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_ROOT = THIS_PATH / "raw_data"
DATASET_ROOT = THIS_PATH / "lego_brick_captions"
DATASET_IMAGES_PATH = DATASET_ROOT / "images"
DATASET_PARQUET_PATH = DATASET_ROOT / "data"
DATASET_CAPTIONS_PATH = DATASET_ROOT / "captions"

INVENTORY_PARTS_PARQUET = "inventory_parts.parquet"
PARTS_PARQUET = "parts.parquet"
COLORS_PARQUET = "colors.parquet"

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

    # delete the rows with missing images
    inventory_parts = inventory_parts[inventory_parts["img_url"].notna()]

    step_1 = inventory_parts.merge(parts, on="part_num")
    step_1.rename(columns={"name": "short_caption"}, inplace=True)
    step_2 = step_1.merge(colors, left_on="color_id", right_on="id")
    step_3 = step_2.rename(columns={"name": "color_name"})
    step_3 = step_3.drop(columns=["id"]) # drop the color "id" duplicate
    logger.info("Merged all the dataframes")
    
    # group by img_url and merge the rows as list of items
    step_4 = step_3.groupby("img_url").agg(
        {
            "inventory_id": list,
            "part_num": list,
            "short_caption": list,
            "part_material": list,
            "color_id": list,
            "color_name": list,
            "rgb": list,
            "is_trans": list
        }
    ).reset_index()
    logger.info("Grouped by img_url")

    # for each row, take out the first element of the lists of all columns
    # in a separate column, and aggregate the rest of the elements in a list
    # of dictionaries
    step_5 = step_4.apply(
        lambda row: {
            "img_url": row["img_url"],
            "inventory_id": row["inventory_id"][0],
            "part_num": row["part_num"][0],
            "short_caption": row["short_caption"][0],
            "part_material": row["part_material"][0],
            "color_id": row["color_id"][0],
            "color_name": row["color_name"][0],
            "color_rgb": row["rgb"][0],
            "is_trans": row["is_trans"][0],
            "extra": [
                {
                    "inventory_id": row["inventory_id"][i],
                    "part_num": row["part_num"][i],
                    "short_caption": row["short_caption"][i],
                    "part_material": row["part_material"][i],
                    "color_id": row["color_id"][i],
                    "color_name": row["color_name"][i],
                    "color_rgb": row["rgb"][i],
                    "is_trans": row["is_trans"][i]
                }
                for i in range(1, len(row["inventory_id"]))
            ]
        },
        axis=1    
    )
    step_5 = pd.DataFrame(step_5.tolist())
    logger.info("Aggregated the data")

    # add idx column
    step_5["idx"] = range(len(step_5))

    step_5.to_parquet(DATASET_ROOT / "lego_bricks_no_img.parquet")
    logger.info("Saved the parquet file")

def remove_missing_rows(dataframe: pd.DataFrame):
    """
    Remove rows from the dataframe that have missing images.
    """
    missing_images = len(dataframe) - len(os.listdir(DATASET_IMAGES_PATH))
    logger.info(f"Missing images: {missing_images}")

    image_files = os.listdir(DATASET_IMAGES_PATH)
    # get the idx from the image file name as idx.jpg
    image_idxs = [int(x.rsplit(".", maxsplit=1)[0]) for x in image_files]

    # remove rows with missing images
    dataframe = dataframe[dataframe["idx"].isin(image_idxs)]
    logger.info(f"Removed rows with missing images, new shape: {dataframe.shape}")

    dataframe.to_parquet(DATASET_PARQUET_PATH / "lego_bricks_no_img.parquet")

def upload_dataset_to_hf(dataframe: pd.DataFrame):
    """
    Upload the dataset to the Hugging Face Datasets Hub.
    """
    table_rows = []
    for _, row in tqdm.tqdm(dataframe.iterrows()):
        image_file_name = f"{row['idx']}.jpg"
        image_file_path = DATASET_IMAGES_PATH / image_file_name
        if not image_file_path.exists():
            continue
        caption_file_name = f"{row['idx']}.json"
        caption_file_path = DATASET_CAPTIONS_PATH / caption_file_name
        if not caption_file_path.exists():
            continue

        image_bytes = b""
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
            "part_num": row["part_num"],
            "inventory_id": row["inventory_id"],
            "part_material": row["part_material"],
            "color_id": row["color_id"],
            "color_name": row["color_name"],
            "color_rgb": row["color_rgb"],
            "is_trans": row["is_trans"],
            "extra": row["extra"]
        }
        # row_data = {
        #     "image": {"bytes": image_bytes, "path": image_file_name},
        #     "short_caption": row["short_caption"],
        #     "caption": caption_data["caption"],
        #     "inventory_id": row["inventory_id"],
        #     "part_num": row["part_num"],
        #     "part_material": row["part_material"],
        #     "color_id": row["color_id"],
        #     "color_name": row["color_name"],
        #     "color_rgb": row["color_rgb"],
        #     "is_trans": row["is_trans"],
        #     "extra": row["extra"]
        # }
        table_rows.append(row_data)

    parquet_table = pa.Table.from_pylist(table_rows)
    pq.write_table(parquet_table, DATASET_PARQUET_PATH / "lego_bricks.parquet")

    hf_dataset = Dataset.from_parquet(str(DATASET_PARQUET_PATH / "lego_bricks.parquet"))
    hf_dataset.save_to_disk(str(DATASET_PARQUET_PATH))
    hf_dataset.push_to_hub(
        repo_id="armaggheddon97/lego_brick_captions",
        split="train",
        max_shard_size="200MB",
        commit_message="Initial commit",
        token=os.environ["HF_TOKEN"]
    )
    logger.success("Uploaded the dataset to the Hugging Face Hub")


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
