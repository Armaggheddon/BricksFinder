# Dataset
This folder contains the code used to create the dataset uploaded to Hugging Face. The dataset is a collection of Lego bricks and minifigures. Some additional data is also included, such as parquet files with all the useful information aggregated in a single file for both bricks and minifigures in their respective folders.

This code generates 2 datasets:
- `lego_brick_captions`: A dataset with the captions of the images of the Lego bricks.
- `lego_minifigure_captions`: A dataset with the captions of the images of the Lego minifigures.

The code is divided into 3 main files:
- `create_lego_brick_dataset.py`: This file creates the `lego_brick_captions` dataset.
- `create_lego_minifigure_dataset.py`: This file creates the `lego_minifigure_captions` dataset.
- `create_captions.py`: This file contains the functions to create the captions for the images using the Gemini APIs.

For both `create_..._dataset.py` files, the following steps are performed:
1. Download the original csv files from the Rebrickable website.
1. Convert the csv files to parquet files. This is mostly for convinience and to reduce the size of the dataset.
1. Create a parquet file with all the informations aggregated in a single file.
1. Download the images for each item.
1. Create the final parquet file with the images, the captions are still missing.
1. Create the captions for the images using the Gemini APIs.
1. Merge the captions with the parquet dataset.
1. Create a zip file including the final parquet files. This step is optional and was only useful due to the fact that I used a remote machine to run this code


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
- `DOWNLOAD_CSVS`: If set to `True`, the original csv files from the Rebrickable website will be downloaded.
- `CONVERT_CSVS_TO_PARQUET`: If set to `True`, the csv files will be converted to parquet files.
- `CREATE_ROOT_PARQUET`: If set to `True`, a parquet file with all the informations aggregated in a single file will be created.
- `DOWNLOAD_IMAGES`: If set to `True`, the images for each item will be downloaded in a folder named `images` in the same folder as the dataset name.
- `CREATE_GEMINI_CAPTIONS`: If set to `True`, the captions for the images will be created using the Gemini APIs.
- `CREATE_PARQUET`: If set to `True`, the final parquet file with the images and captions will be created.
- `CREATE_ZIP`: If set to `True`, a zip file with the final parquet file will be created.

The code is already set to run all the steps in the `create_..._dataset.py` files. If you want to run only a subset of the steps, you can change the variables at the top of the file.

> [!WARNINIG]
> The code for the dataset generation will take a significant amount of time to both download and caption the images. Additionally, the captions are generated using the Gemini APIs, which require an API key. You can get a free API key from the [Gemini website](https://www.gemini.com/). Note that the free API key has a limit of 1500 requests per day, so it will take multiple days in case of a free API key.

> [!NOTE]
> Avoid running the code to generate the dataset from scratch since it will put unnecessary load on the Rebrikable servers. All the required data and more, is already available. Refer to the [Available data](#available-data) section for more information.

## Available data

This folder already contains more than enought data to be used for both the dataset creation and more. The data available is:

- `lego_brick_captions`: **COMING SOON**
- `lego_minifigure_captions`: 
    - `minifigures_no_img.parquet`: parquet containing `file_name`, `img_url`, `fig_num`, `name`, `num_parts`, `inventory_id`. The first row contains the following information:
        | file_name | img_url | fig_num | name | num_parts | inventory_id |
        |-----------|---------|---------|------|-----------|--------------|
        | 0.jpg |  https://cdn.rebrickable.com/media/sets/fig-000001.jpg | fig-000001 | Toy Store Employee | 4 | [42484] |
- `raw_data`: 
    - `colors.parquet`: parquet containing `id`, `name`, `rgb`, `is_trans`. The first row contains the following information:
        | id | name | rgb | is_trans |
        |----|------|-----|----------|
        | 0 | Black | 05131D | false |
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

