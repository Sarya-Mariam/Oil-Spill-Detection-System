# Streamlit app to classify and segment oil spills using a dual-head UNet model
# The model file is downloaded automatically from Google Drive if not present locally.

import os
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import gdown
from tensorflow import keras

st.set_page_config(layout="wide", page_title="Oil-spill Detector & Segmenter")

st.title("Oil-spill Detector — Dual-head UNet")
st.write("Upload an image; the app will predict whether it contains an oil spill and show the segmented region.")

# Fixed model input size (remove adjuster to avoid confusion)
IMG_SIZE = 256  # set this to the size your model expects

MODEL_PATH = "models/dual_head_best.h5"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg"

# Ensure model exists locally
if not os.path.exists(MODEL_PATH):
    st.info("Model not found locally. Downloading from Google Drive...")
    os.makedirs("models", exist_ok=True)
    zip_path = "models/model.zip"
    gdown.download(GOOGLE_DRIVE_URL, zip_path, quiet=False, fuzzy=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("models")

# Load model
@st.cache_resource
def load_model(path: str):
    try:
        model = keras.models.load_model(path, compile=False)
        st.success("Loaded model successfully (Keras)")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model_obj = load_model(MODEL_PATH)

# Preprocessing & helpers
def preprocess_image_pil(img: Image.Image, size: int):
    img = img.convert('RGB')
    img = ImageOps.fit(img, (size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    nhwc = np.expand_dims(arr, 0)   # add batch dimension
    return arr, nhwc

def postprocess_mask(mask: np.ndarray, orig_size):
    mask = np.array(mask)

    # Remove all size-1 dims
    mask = np.squeeze(mask)

    # If mask is 3D, reduce it
    if mask.ndim == 3:
        # Case: (H, W, C) → take first channel
        mask = mask[..., 0]
    elif mask.ndim == 1:
        # Case: flat vector → reshape square if possible
        side = int(np.sqrt(mask.size))
        if side * side == mask.size:
            mask = mask.reshape((side, side))
        else:
            # fallback: just return a blank mask
            return np.zeros(orig_size[::-1], dtype=np.uint8)

    # Now force 2D
    if mask.ndim != 2:
        # fallback: blank mask
        mask = np.zeros(orig_size[::-1], dtype=np.uint8)

    # Convert to PIL, resize back to input size
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize(orig_size, resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0
    bin_mask = (arr >= 0.5).astype(np.uint8)
    return bin_mask


def predict(model, pil_image: Image.Image, size: int):
    _, tensor = preprocess_image_pil(pil_image, size)
    out = model.predict(tensor)

    # Try to split classification and segmentation
    class_out, mask_out = None, None
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        class_out, mask_out = out[0], out[1]
    else:
        if out.ndim == 4 and out.shape[-1] == 1:
            mask_out = np.squeeze(out, axis=0)[..., 0]
        elif out.ndim == 2:
            class_out = out

    # Classification probability
    is_oil_prob = None
    if class_out is not None:
        v = np.squeeze(class_out)
        if np.ndim(v) == 0:
            v = float(v)
            is_oil_prob = 1.0 / (1.0 + np.exp(-v))  # sigmoid
        elif np.ndim(v) == 1:
            p = np.exp(v) / np.sum(np.exp(v))       # softmax
            is_oil_prob = float(p[-1])

    # Segmentation mask
    mask_prob = None
    if mask_out is not None:
        mask_prob = np.squeeze(mask_out)
        if mask_prob.max() > 1.01:
            mask_prob = mask_prob / 255.0

    return is_oil_prob, mask_prob

# Image uploader
st.header("Upload image to analyze")
uploaded_image = st.file_uploader("Upload image (jpg/png/tif...)", type=['jpg','jpeg','png','tif','tiff'])

col1, col2 = st.columns([1,1])

if uploaded_image:
    pil = Image.open(uploaded_image)
    with col1:
        st.subheader("Input image")
        st.image(pil, use_container_width=True)

    if not model_obj:
        st.warning("Model not loaded yet.")
    else:
        with st.spinner("Running prediction..."):
            try:
                prob, mask_prob = predict(model_obj, pil, IMG_SIZE)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                prob, mask_prob = None, None

        with col2:
            st.subheader("Results")
            if prob is not None:
                st.metric("Oil spill probability", f"{prob*100:.1f}%")
                is_oil = prob >= 0.5
                st.write("Prediction:", "**OIL SPILL**" if is_oil else "**NO OIL SPILL**")
            else:
                st.write("Classification result not available.")

            if mask_prob is not None:
                bin_mask = postprocess_mask(mask_prob, pil.size)
                overlay = pil.convert('RGBA')
                mask_img = Image.fromarray((bin_mask*200).astype('uint8')).convert('L')
                color_mask = Image.new('RGBA', pil.size, (255,0,0,120))
                overlay.paste(color_mask, (0,0), mask_img)

                st.image(overlay, caption='Overlay: red = predicted oil region', use_container_width=True)
                st.image(bin_mask*255, caption='Binary mask (white = predicted oil)', use_container_width=True)

                buf = io.BytesIO()
                overlay.save(buf, format='PNG')
                buf.seek(0)
                st.download_button('Download overlay PNG', data=buf, file_name='overlay.png', mime='image/png')

                buf2 = io.BytesIO()
                Image.fromarray((bin_mask*255).astype('uint8')).save(buf2, format='PNG')
                buf2.seek(0)
                st.download_button('Download mask PNG', data=buf2, file_name='mask.png', mime='image/png')
            else:
                st.write("Segmentation mask not available.")

st.markdown("---")
st.subheader("Troubleshooting & tips")
st.markdown(
"""
- The model file is automatically downloaded from Google Drive if not found locally.
- Ensure your Google Drive link/ID is correct.
- If your model expects a different input size or normalization, change `IMG_SIZE` in the code.
"""
)










