import os
import gdown
import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "dual_head_best.h5"
IMG_SIZE = 256
MIN_SPILL_AREA_PIXELS = 500

def download_model_if_needed():
    """Download and extract model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/file/d/1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg/view?usp=sharing"
        output = "dual_head_model.zip"
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Model downloaded and extracted")

def load_tf_model():
    """Load TensorFlow/Keras model."""
    download_model_if_needed()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image):
    """Resize and normalize the input image."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict_and_analyze(image, threshold=0.5):
    """Run prediction and return results for UI."""
    model = load_tf_model()
    arr = preprocess_image(image)

    # Prediction
    preds = model.predict(arr)
    mask = (preds[0, :, :, 0] > threshold).astype(np.uint8)

    # Convert to PIL
    mask_img = Image.fromarray(mask * 255)

    # Spill area stats
    spill_pixels = np.sum(mask)
    total_pixels = mask.size
    perc_area = (spill_pixels / total_pixels) * 100

    status = "✅ Spill Detected" if spill_pixels > MIN_SPILL_AREA_PIXELS else "❌ No Spill"

    return status, image, mask_img, int(spill_pixels), perc_area
