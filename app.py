import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import gdown
import os
import zipfile

MODEL_PATH = "unet_oilspill_final.h5"
GOOGLE_DRIVE_ID = "1NN6K6mZNWLpP_BQpSszj3y1ieEBjIgrO"# <-- replace with your file ID
url = f"https://drive.google.com/file/d/1K2dd9_P2zIgHrSrq5sNc2kyJtnj1o_gf/view?usp=sharing"

if not os.path.exists("unet_oilspill_final.h5"):
    gdown.download(url, "unet_model.zip", quiet=False, fuzzy=True)
    import zipfile
    with zipfile.ZipFile("unet_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")


model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# Load trained model (ensure the file exists in same folder or provide full path)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
IMG_SIZE = 256

def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return arr

st.title("🌊 AI SpillGuard – Oil Spill Detection")
st.write("Upload a satellite image to detect oil spill regions using a trained U-Net model.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg","jpeg","png","tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    arr = preprocess_image(image)

    # Predict segmentation mask
    pred = model.predict(np.expand_dims(arr, 0))[0]

    # Step 1: Apply stricter confidence threshold
    pred_bin = (pred[:,:,0] > 0.7).astype("uint8")

    # Step 2: Morphological filtering to remove noise
    kernel = np.ones((3,3), np.uint8)
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

    # Step 3: Decide oil spill vs no spill
    spill_ratio = np.sum(pred_bin) / pred_bin.size
    THRESHOLD = 0.05  # Require at least 5% of pixels to be spill

    if spill_ratio > THRESHOLD:
        st.success(f"🌊 Oil Spill Detected! (covering ~{spill_ratio*100:.2f}% of image)")
    else:
        st.info("✅ No Oil Spill Detected")

    # Overlay visualization
    overlay = cv2.addWeighted(
        np.array(image.resize((IMG_SIZE, IMG_SIZE))), 0.7,
        cv2.applyColorMap((pred_bin*255).astype("uint8"), cv2.COLORMAP_JET), 0.3, 0
    )
    st.image(overlay, caption="Predicted Oil Spill Regions", use_container_width=True)
















