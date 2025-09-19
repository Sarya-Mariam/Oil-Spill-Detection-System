import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import gdown
import os

MODEL_PATH = "unet_oilspill_final.h5"
GOOGLE_DRIVE_ID = "1NOJ7tL3pL6BJi8xIz8EumPR0BvW8Trd"  # <-- replace with your file ID

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
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    arr = preprocess_image(image)
    pred = model.predict(np.expand_dims(arr, 0))[0]
    pred_bin = (pred[:,:,0] > 0.5).astype("float32")

    # Overlay result
    overlay = cv2.addWeighted(
        np.array(image.resize((IMG_SIZE, IMG_SIZE))), 0.7,
        cv2.applyColorMap((pred_bin*255).astype("uint8"), cv2.COLORMAP_JET), 0.3, 0
    )
    st.image(overlay, caption="Predicted Oil Spill Regions", use_column_width=True)


