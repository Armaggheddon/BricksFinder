# Dataset
This folder contains the code used to create the dataset uploaded to Hugging Face. The dataset is a collection of Lego bricks and minifigures. Some additional data is also included, such as parquet files with all the useful information aggregated in a single file for both bricks and minifigures in their respective folders.

This code generates 2 datasets:
- `lego_brick_captions`: A dataset with the captions of the images of the Lego bricks.
- `lego_minifigure_captions`: A dataset with the captions of the images of the Lego minifigures.

The code is divided into 3 main files:
- `create_lego_brick_dataset.py`: This file creates the `lego_brick_captions` dataset.
- `create_lego_minifigure_dataset.py`: This file creates the `lego_minifigure_captions` dataset.
- `download_and_convert_raw.py`: This file downloads the raw data from the Rebrickable website and converts it to parquet files.

Before running the `create_..._dataset.py` files, the `download_and_convert_raw.py` file must be run to download the raw data from the Rebrickable website and convert it to parquet files since the `create_..._dataset.py` files rely on the parquet files created by `download_and_convert_raw.py`.

For both `create_..._dataset.py` files, the following steps are performed:
1. Create a parquet file with all the informations aggregated in a single file.
1. Download the images for each item, and remove the items without images.
1. Create the captions for the images using the Gemini APIs.
1. Merge the captions with the parquet dataset, converts the required columns to a pyarrow table and then loads them as a Hugging Face Dataset. Finally the dataset is uploaded to the Hugging Face Hub (requires an API key with the right permissions).


As an important note, I want to mention that all the data for the starting point of this dataset is from the Rebrickable website. They have done an amazing job collecting all the information about Lego bricks and minifigures. I have only used their data to create this dataset, and have added the captions using Gemini 1.5 Flash (002).

> [!NOTE]
> All the data in this folder has been collected from the [Rebrickable](https://rebrickable.com/downloads/) on 27 November 2024. Therefore all credits for the data go to the Rebrickable team.

## Setup
The code in this folder relies on a python environment. The only exception is for the [`image_captioning`](./image_captioning/) folder, which uses a docker container due to the fact that uses the GPU for execution. Refer to the [`README.md`](./image_captioning/README.md) in that folder for additional information.

To setup the python environment, you can use the following commands:
```bash
python3 -m venv .dataset
source .dataset/bin/activate
pip install -r requirements.txt
```

## Usage
Each of the `create_..._dataset.py` files can be run independently. On top of each of the files there are a couple of options:
- `CREATE_ROOT_PARQUET`: If set to `True`, a parquet file with all the informations aggregated in a single file will be created, this file is required for the next steps.
- `DOWNLOAD_IMAGES`: If set to `True`, the images for each item will be downloaded in a folder named `images` in `.jpg` format.
- `CREATE_GEMINI_CAPTIONS`: If set to `True`, the captions for the images will be created using the Gemini APIs and saved in the `captions` folder as `.json`.
- `CREATE_AND_UPLOAD_DATASET`: If set to `True`, the captions will be merged with the parquet dataset, converts the required columns to a pyarrow table and then loads them as a Hugging Face Dataset. Finally the dataset is uploaded to the Hugging Face Hub (requires an API key with the right permissions).

The code is already set to run all the steps in the `create_..._dataset.py` files. If you want to run only a subset of the steps, you can change the variables at the top of the file.

> [!WARNINIG]
> The code for the dataset generation will take a significant amount of time to both download and caption the images. Additionally, the captions are generated using the Gemini APIs, which require an API key. You can get a free API key from the [Gemini website](https://www.gemini.com/). Note that the free API key has a limit of 1500 requests per day, so it will take multiple days in case of a free API key.

> [!NOTE]
> Avoid running the code to generate the dataset from scratch since it will put unnecessary load on the Rebrikable servers. All the required data and more, is already available. Refer to the [Available data](#available-data) section for more information.

## Available data

This folder already contains more than enought data to be used for both the dataset creation and more. The data available is:

- `lego_brick_captions`: **COMING SOON**
- `lego_minifigure_captions`: this is the working directory used for creating the dataset. The final dataset is available in the [Hugging Face datasets](https://huggingface.co/datasets/armaggheddon97/lego_minifigure_captions).
- `raw_data`: 
    - `colors.parquet`: parquet containing `id`, `name`, `rgb`, `is_trans`. The first row contains the following information:
        | id | name | rgb | is_trans |
        |----|------|-----|----------|
        | 0 | Black | 05131D | false |
    - `inventories.parquet`: parquet containing `id`, `version`, `set_num`. The first row contains the following information:
        | id | version | set_num |
        |----|---------|---------|
        | 1 | 1 | 7922-1 |
    - `inventory_minifigs.parquet`: parquet containing `inventory_id`, `fig_num`, `quantity`. The first row contains the following information:
        | inventory_id | fig_num | quantity |
        |--------------|---------|----------|
        | 3 | fig-001549 | 1 |
    - `inventory_parts.parquet`: parquet containing `inventory_id`, `part_num`, `color_id`, `quantity`, `is_spare`, `img_url`. The first row contains the following information:
        | inventory_id | part_num | color_id | quantity | is_spare | img_url |
        |--------------|----------|----------|----------|----------|---------|
        | 1 | 48379c04 | 72 | 1 | false | https://cdn.rebrickable.com/media/parts/photos/1/48379c01-1-839cbcec-62de-4733-ba23-20f35f4dd5d5.jpg |
    - `minifigs.parquet`: parquet containing `fig_num`, `name`, `num_parts`, `img_url`. The first row contains the following information:
        | fig_num | name | num_parts | img_url |
        |---------|------|-----------|---------|
        | fig-000001 | Toy Store Employee | 4 | https://cdn.rebrickable.com/media/sets/fig-000001.jpg |
    - `part_categories.parquet` parquet containing `id`, `name`. The first row contains the following information:
        | id | name |
        |----|------|
        | 1 | Baseplates |
    - `parts.parquet`: parquet containing `part_num`, `name`, `part_cat_id`, `part_material`. The first row contains the following information:
        | part_num | name | part_cat_id | part_material |
        |----------|------|-------------|--------------|
        | 003381 | Sticker Sheet for Set 663-1 | 58 | Plastic |

The final datasets with the downloaded images and captions are not included in this folder due to their size, but they are available in the Hugging Face datasets:

- Lego Bricks Captions (**COMING SOON**)
- [Lego Minifigures Captions](https://huggingface.co/datasets/armaggheddon97/lego_minifigure_captions)

