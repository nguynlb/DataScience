import os
from pathlib import Path

import requests as req
from zipfile import ZipFile

url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
data_dir = Path("/data")

if data_dir.is_dir():
    print(f"Data has already been downloaded. Skip downloading...")
else:
    print(f"Data is not found. Start downloading...")
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile(data_dir / "pizza_steak_sushi.zip", "wb") as file:
        try:
            res = req.get(url)
            file.write(res.content)
            print(f"Zipfile has been downloaded")
        except:
            print(f"Something has wrong.")
    print(f"Start unziping...")
    
    with ZipFile(data_dir / "pizza_steak_sushi.zip", "r") as file:
        file.extractall(data_dir)
        print(f"Download successful...")
