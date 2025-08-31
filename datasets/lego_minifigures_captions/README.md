# LEGO Minifigure Captions

The **LEGO Minifigure Captions** dataset contains 12966 images of LEGO minifigures with captions. The dataset contains the following columns:
- `image`: The jpeg image of the minifigure in the format `{"bytes": bytes, "path": str}` so that can be interpreted as `PIL.Image` objects in the huggingface `datasets` library.
- `short_caption`: The short caption describing the minifigure in the image.
- `caption`: The caption describing the minifigure which is generated using Gemini-1.5-flash with the following prompt:
    ```python3
    GEMINI_PROMPT = (
      "You are a precise and detailed image captioning assistant. Your task is "
      "to describe LEGO minifigures in a single sentence, focusing on their "
      "unique features, such as attire, accessories, facial expression, "
      "and theme. Avoid generic terms and aim for specificity, "
      "while remaining concise. Caption the image"
    )
    ```
- `fig_num`: The figure number of the minifigure as in the original csv file from Rebrickable.
- `num_parts`: The number of parts that the minifigure is made of.
- `minifig_inventory_id`: The inventory id of the minifigure in the Rebrickable database.
- `part_inventory_id`: The inventory id of the part in the Rebrickable database.
- `part_num`: The part numbers that the minifigure is made of, which can be directly searched on the Rebrickable website.

The data has been collected from the [Rebrickable](https://rebrickable.com/downloads/) website and the images have been downloaded from the column `img_url` in original *minifigs.csv* file from the website. 

> [!NOTE]
> The total number of minifigures in `minifigs.csv` are 14985, but only 12966 images were downloaded due to some images not being available.

*The data was downloaded from the Rebrickable website on 27 November 2024.*


Again a massive shoutout and thanks goes to the [Rebrickable](https://rebrickable.com/) team for providing all the data and images for the LEGO minifigures, and more!

> [!TIP]
> This dataset is also available on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/Armaggheddon/lego_minifigure_captions).

## Usage with pandas
Using this dataset with pandas requires the `pyarrow` library to be installed. Also the parquet files have to be downloaded.
```python
from pathlib import Path
import pandas as pd

PATH_TO_DATASET = Path("path_to_dataset")

# Load the dataset
df = pd.read_parquet(PATH_TO_DATASET / "minifigures-00000-of-00003.parquet")
print(df.head())
```

## Usage with huggingface/datasets
```python
from datasets import load_dataset

# Load the dataset in streaming mode
ds = load_dataset("Armaggheddon/lego_minifigure_captions", split="train", streaming=True)

# Print the dataset info
print(next(iter(ds)))
```
> [!TIP]
> The `image` column using the `datasets` library is already in PIL format.

