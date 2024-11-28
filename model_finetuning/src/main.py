from pathlib import Path
import pandas as pd


DATASET_PATH = Path("/lego_bricks_dataset/processed_data")
IMAGES_PATH = DATASET_PATH / "images"

lego_bricks = pd.read_pickle(DATASET_PATH / "lego_bricks.pkl")

img_text_pairs = lego_bricks[["img_url", "part_name"]].dropna()

