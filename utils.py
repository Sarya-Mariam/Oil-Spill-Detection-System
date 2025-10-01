import os
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "dual_head_best.h5"
IMG_SIZE = 256
MIN_SPILL_AREA_PIXELS = 500

def download_model_if_needed():
    """Download the model from Google Drive if it's not already present."""
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1qgq7nawEts3s-aSJ7mF81zIMwepoQSPi"
        print("Downloading model from:", url)
        gdown.download(url, MODEL_PATH, quiet=False)

    # Debug: check file size to ensure it’s large enough
    size = os.path.getsize(MODEL_PATH)
    print(f"✅ Downloaded model size: {size/1024/1024:.2f} MB")

def load_tf_model():
    """Load the Keras model (without compile)."""
    download_model_if_needed()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
    return model

def preprocess_image(image):
    """Resize and normalize the image."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    return arr

def predict_and_analyze(image, threshold=0.5):
    """Run prediction, analyze, and return status + images/stats."""
    model = load_tf_model()
    arr = preprocess_image(image)

    preds = model.predict(arr)
    # assuming the output is (1, H, W, 1) or something like that:
    mask = (preds[0, :, :, 0] > threshold).astype(np.uint8)

    mask_img = Image.fromarray(mask * 255)

    spill_pixels = np.sum(mask)
    total_pixels = mask.size
    perc_area = (spill_pixels / total_pixels) * 100

    status = "✅ Spill Detected" if spill_pixels > MIN_SPILL_AREA_PIXELS else "❌ No Spill"

    return status, image, mask_img, int(spill_pixels), perc_area
