import os
import gdown
import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "dual_head_best.h5"
ZIP_PATH = "dual_head_model.zip"
IMG_SIZE = 256
MIN_SPILL_AREA_PIXELS = 500

def download_and_extract_model():
    """Download zip from Google Drive and extract the .h5 model."""
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg"  # <-- update with zip file ID
        print("Downloading model zip from:", url)
        gdown.download(url, ZIP_PATH, quiet=False)

        # Extract
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Model extracted")

def load_tf_model():
    """Load Keras model after ensuring it's downloaded & extracted."""
    download_and_extract_model()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
    return model

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict_and_analyze(image, threshold=0.5):
    model = load_tf_model()
    arr = preprocess_image(image)

    preds = model.predict(arr)
    mask = (preds[0, :, :, 0] > threshold).astype(np.uint8)

    mask_img = Image.fromarray(mask * 255)

    spill_pixels = np.sum(mask)
    total_pixels = mask.size
    perc_area = (spill_pixels / total_pixels) * 100

    status = "✅ Spill Detected" if spill_pixels > MIN_SPILL_AREA_PIXELS else "❌ No Spill"

    return status, image, mask_img, int(spill_pixels), perc_area
