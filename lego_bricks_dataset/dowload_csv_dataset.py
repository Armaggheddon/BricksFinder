import os
import urllib.request
import shutil


THEMES_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/themes.csv.gz?1732432080.0733411"
COLORS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/colors.csv.gz?1732432080.0813415"
PART_CATEGORIES_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/part_categories.csv.gz?1732432080.0853415"
PARTS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/parts.csv.gz?1732432081.1133678"
PART_RELATIONSHIPS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/part_relationships.csv.gz?1732432106.366016"
ELEMENTS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/parts.csv.gz?1732432081.1133678"
SETS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/sets.csv.gz?1732432083.029417"
MINIFIGS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz?1732432083.3254244"
INVENTORIES_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/inventories.csv.gz?1732432082.3934007"
INVENTORY_PARTS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/inventory_parts.csv.gz?1732432105.4099913"
INVENTORY_SETS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/inventory_sets.csv.gz?1732432105.549995"
INVENTORY_MINIFIGS_CSV_LINK = "https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz?1732432105.8860035"


def download_dataset(download_path: str):
    urllib.request.urlretrieve(THEMES_CSV_LINK, f"{download_path}/themes.csv.gz")
    urllib.request.urlretrieve(COLORS_CSV_LINK, f"{download_path}/colors.csv.gz")
    urllib.request.urlretrieve(PART_CATEGORIES_CSV_LINK, f"{download_path}/part_categories.csv.gz")
    urllib.request.urlretrieve(PARTS_CSV_LINK, f"{download_path}/parts.csv.gz")
    urllib.request.urlretrieve(PART_RELATIONSHIPS_CSV_LINK, f"{download_path}/part_relationships.csv.gz")
    urllib.request.urlretrieve(ELEMENTS_CSV_LINK, f"{download_path}/elements.csv.gz")
    urllib.request.urlretrieve(SETS_CSV_LINK, f"{download_path}/sets.csv.gz")
    urllib.request.urlretrieve(MINIFIGS_CSV_LINK, f"{download_path}/minifigs.csv.gz")
    urllib.request.urlretrieve(INVENTORIES_CSV_LINK, f"{download_path}/inventories.csv.gz")
    urllib.request.urlretrieve(INVENTORY_PARTS_CSV_LINK, f"{download_path}/inventory_parts.csv.gz")
    urllib.request.urlretrieve(INVENTORY_SETS_CSV_LINK, f"{download_path}/inventory_sets.csv.gz")
    urllib.request.urlretrieve(INVENTORY_MINIFIGS_CSV_LINK, f"{download_path}/inventory_minifigs.csv.gz")

def unzip_csv_files(files_path: str):
    os.system(f"gzip -d {files_path}/*.gz")


if __name__ == "__main__":
    
    this_path = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(this_path, "raw_data")
    
    if os.path.exists(raw_data_path):
        if len(os.listdir(raw_data_path)) > 0:
            print("Some files have already been downloaded")
            print("Do you want to delete them and download again?")
            answer = input("y/n: ")
            if answer.lower() == "n":
                exit()
            if answer.lower() != "y":
                print("Invalid answer")
                exit()
            print("Deleting folder content")
            shutil.rmtree(raw_data_path)
    
    os.makedirs(raw_data_path, exist_ok=True)
            
    download_dataset(raw_data_path)
    unzip_csv_files(raw_data_path)