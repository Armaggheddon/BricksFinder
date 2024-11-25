import urllib.request
import os
from pathlib import Path
import tqdm
from PIL import Image
import shutil

import pandas as pd
from concurrent.futures import ThreadPoolExecutor


this_path = Path(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = this_path / "processed_data"
IMAGES_PATH = PROCESSED_PATH / "images"


def download_image(row, i):
    url = row["img_url"]
    if pd.notna(url):
        try:
            img = Image.open(urllib.request.urlopen(url))
            img.save(IMAGES_PATH / f"{i}.jpg")
        except:
            print(f"Error downloading image for {row['part_num']}")
            return i
    return None

def download_images():
    lego_bricks = pd.read_pickle(PROCESSED_PATH / "lego_bricks.pkl")

    failed_downloads = []

    # count the number of images to download where img_url is not null
    print(f"Downloading {lego_bricks['img_url'].count()} images")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_image, row, i) for i, row in lego_bricks.iterrows()]
        for future in tqdm.tqdm(futures, total=len(futures)):
            result = future.result()
            if result is not None:
                failed_downloads.append(result)
    

if __name__ == "__main__":
    
    if os.path.exists(IMAGES_PATH):
        if len(os.listdir(IMAGES_PATH)) > 0:
            print("It appears that some images have already been downloaded")
            print("Do you want to delete them and download again?")
            answer = input("y/n: ")
            if answer.lower() == "n":
                exit()
            if answer.lower() != "y":
                print("Invalid answer")
                exit()
            print("Deleting folder content")
            shutil.rmtree(IMAGES_PATH)
    
    os.makedirs(IMAGES_PATH, exist_ok=True)
    download_images()
