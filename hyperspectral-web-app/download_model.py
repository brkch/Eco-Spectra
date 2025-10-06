import os
import requests

MODEL_DIR = "models"
MODEL_FILE = "unet_plastic_segmentation_LAST.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://www.dropbox.com/scl/fi/i2ysd92g7k22zws3fnnnc/unet_plastic_segmentation_LAST.h5?dl=1"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_FILE} …")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
else:
    print(f"{MODEL_FILE} already present — no download needed.")
