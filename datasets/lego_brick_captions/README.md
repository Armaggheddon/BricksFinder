# LEGO Brick Captions

The **LEGO Brick Captions** dataset contains XXX images of LEGO bricks with captions. The dataset contains the following columns:
- `image`: The jpeg image of the brick in the format `{"bytes": bytes, "path": str}` so that can be interpreted as `PIL.Image` objects in the huggingface `datasets` library.
- `short_caption`: The short caption describing the minifigure in the image.
- `caption`: The caption describing the brick which is generated using Gemini-1.5-flash-002 with the following prompt:
    ```python3
    GEMINI_PROMPT = (
      "TODO"
    )
    ```
- `inventory_id`: The inventory id of the brick in the Rebrickable database.
- `part_num`: The part number of the brick as in the original csv file from Rebrickable.
- `part_material`: The material of the brick.
- `color_id`: The color id of the brick in the Rebrickable database.
- `color_name`: The name of the color of the brick.
- `color_rgb`: The RGB value of the color of the brick.
- `is_trans`: Whether the brick is transparent or not.
- `extra`: Since multiple bricks correspond to the same image, this column contains all the other bricks that use the same image. The data is in a list of dictionaries with the following keys:
    - `short_caption`: The short caption describing the minifigure in the image.
    - `inventory_id`: The inventory id of the brick in the Rebrickable database.
    - `part_num`: The part number of the brick as in the original csv file from Rebrickable.
    - `part_material`: The material of the brick.
    - `color_id`: The color id of the brick in the Rebrickable database.
    - `color_name`: The name of the color of the brick.
    - `color_rgb`: The RGB value of the color of the

The data has been collected from the [Rebrickable](https://rebrickable.com/downloads/) website and the images have been downloaded from the column `img_url` in original *inventory_parts.csv* file from the website. 

> [!NOTE]
> The total number of minifigures in `inventory_parts.csv` are 1304782, after aggregating the data by `img_url`, the number of unique images are 80913. The dataset has the first item for each image in a separate column and the rest of items in the `extra` column.

*The data was downloaded from the Rebrickable website on 27 November 2024.*


Again a massive shoutout and thanks goes to the [Rebrickable](https://rebrickable.com/) team for providing all the data and images for the LEGO minifigures, and more!

> [!TIP]
> This dataset is also available on the Hugging Face Datasets Hub TODO: Add link.

## Usage with pandas
Using this dataset with pandas requires the `pyarrow` library to be installed. Also the parquet files have to be downloaded.
```python
from pathlib import Path
import pandas as pd

PATH_TO_DATASET = Path("path_to_dataset")

# Load the dataset
df = pd.read_parquet(PATH_TO_DATASET / "train-00000-of-XXXXX.parquet")
print(df.head())
```

## Usage with huggingface/datasets
```python
from datasets import load_dataset

# Load the dataset in streaming mode
ds = load_dataset("armaggheddon97/XXX", split="train", streaming=True)

# Print the dataset info
print(next(iter(ds)))
```
> [!TIP]
> The `image` column using the `datasets` library is already in PIL format.

