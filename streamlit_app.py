import os
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from tensorflow import keras

st.set_page_config(layout="wide", page_title="Oil-spill Detector & Segmenter")

st.title("Oil-spill Detector — Dual-head UNet")
st.write("Upload an image; the app will predict whether it contains an oil spill and show the segmented region.")

# Sidebar config
st.sidebar.header("Model configuration")
IMG_SIZE = st.sidebar.number_input("Model input size (square)", min_value=32, max_value=2048, value=256, step=32)

MODEL_PATH = Path("models/dual_head_best.h5")

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}")
        return None
    try:
        model = keras.models.load_model(path, compile=False)
        st.success("Loaded model successfully (Keras)")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image_pil(img: Image.Image, size: int):
    img = img.convert('RGB')
    img = ImageOps.fit(img, (size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    nhwc = np.expand_dims(arr, 0)   # add batch dimension
    return arr, nhwc

def postprocess_mask(mask: np.ndarray, orig_size):
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize(orig_size, resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    bin_mask = (arr >= 0.5).astype(np.uint8)
    return bin_mask

def predict(model, pil_image: Image.Image, size: int):
    orig_size = pil_image.size  # (W,H)
    _, tensor = preprocess_image_pil(pil_image, size)
    out = model.predict(tensor)

    class_out, mask_out = None, None
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        class_out, mask_out = out[0], out[1]
    else:
        if out.ndim == 4 and out.shape[-1] == 1:
            mask_out = np.squeeze(out, axis=0)[..., 0]
        elif out.ndim == 2:
            class_out = out

    is_oil_prob = None
    if class_out is not None:
        v = np.squeeze(class_out)
        if v.size > 1:  # softmax
            p = np.exp(v) / np.sum(np.exp(v))
            is_oil_prob = float(p[-1])
        else:           # sigmoid
            is_oil_prob = float(1.0 / (1.0 + np.exp(-v)))

    mask_prob = None
    if mask_out is not None:
        mask_prob = np.squeeze(mask_out)
        if mask_prob.max() > 1.01:
            mask_prob = mask_prob / 255.0

    return is_oil_prob, mask_prob

# Load model once
model_obj = load_model(str(MODEL_PATH))

# Image uploader
st.header("Upload image to analyze")
uploaded_image = st.file_uploader("Upload image (jpg/png/tif...)", type=['jpg','jpeg','png','tif','tiff'])

col1, col2 = st.columns([1,1])

if uploaded_image:
    pil = Image.open(uploaded_image)
    with col1:
        st.subheader("Input image")
        st.image(pil, use_column_width=True)

    if not model_obj:
        st.warning("Model not loaded yet. Ensure dual_head_best.h5 is in models/ directory.")
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

                st.image(overlay, caption='Overlay: red = predicted oil region', use_column_width=True)
                st.image(bin_mask*255, caption='Binary mask (white = predicted oil)', use_column_width=True)

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
- Ensure your model file `dual_head_best.h5` is downloaded into `models/` (via setup.sh).
- If your model expects a different input size or normalization, change `IMG_SIZE` or update `preprocess_image_pil`.
- For large deployments, confirm Google Drive download works during build (see setup.sh).
"""
)









