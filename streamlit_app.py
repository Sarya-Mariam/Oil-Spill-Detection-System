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

# Fixed model input size
IMG_SIZE = 256

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

def postprocess_mask(mask: np.ndarray, target_size, threshold=0.5):
    """
    Process model output mask to binary mask matching original image size
    """
    # Remove batch dimension if present
    if mask.ndim == 4:
        mask = mask[0]  # (1, H, W, C) -> (H, W, C)
    
    # Handle channel dimension
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]  # (H, W, 1) -> (H, W)
        else:
            # Take first channel if multiple
            mask = mask[..., 0]
    
    # Now should be 2D
    if mask.ndim != 2:
        st.error(f"Unexpected mask shape after processing: {mask.shape}")
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    # Normalize if needed
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    # Resize to original image size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_img.resize(target_size, resample=Image.BILINEAR)
    
    # Convert back to numpy and threshold
    mask_arr = np.array(mask_resized).astype(np.float32) / 255.0
    binary_mask = (mask_arr >= threshold).astype(np.uint8)
    
    return binary_mask


def predict(model, pil_image: Image.Image, size: int):
    """
    Run prediction on image
    Returns: (classification_probability, segmentation_mask_array)
    """
    _, tensor = preprocess_image_pil(pil_image, size)
    
    # Get model prediction
    out = model.predict(tensor, verbose=0)
    
    # Debug: print output structure
    if isinstance(out, (list, tuple)):
        st.sidebar.write(f"Model outputs {len(out)} tensors")
        for i, o in enumerate(out):
            st.sidebar.write(f"Output {i} shape: {o.shape}")
    else:
        st.sidebar.write(f"Model output shape: {out.shape}")
    
    # Parse outputs - adjust based on your model architecture
    classification_prob = None
    segmentation_mask = None
    
    if isinstance(out, (list, tuple)):
        # Dual-head model with separate outputs
        if len(out) >= 2:
            # Typically: [classification_output, segmentation_output]
            class_out = out[0]
            seg_out = out[1]
            
            # Process classification output
            class_out = np.squeeze(class_out)
            if class_out.ndim == 0:
                # Single sigmoid output
                classification_prob = float(class_out)
                # Apply sigmoid if needed (raw logit)
                if classification_prob < 0 or classification_prob > 1:
                    classification_prob = 1.0 / (1.0 + np.exp(-classification_prob))
            elif class_out.ndim == 1:
                # Softmax output (2 classes: no-oil, oil)
                if class_out.shape[0] == 2:
                    probs = np.exp(class_out) / np.sum(np.exp(class_out))
                    classification_prob = float(probs[1])  # probability of oil class
                else:
                    classification_prob = float(class_out[0])
            
            # Segmentation mask
            segmentation_mask = seg_out
    else:
        # Single output - could be either classification or segmentation
        if out.shape[-1] == 1 and len(out.shape) == 4:
            # Likely segmentation mask (B, H, W, 1)
            segmentation_mask = out
        elif out.shape[-1] <= 2 and len(out.shape) == 2:
            # Likely classification (B, num_classes)
            class_out = np.squeeze(out)
            if class_out.ndim == 1 and len(class_out) == 2:
                probs = np.exp(class_out) / np.sum(np.exp(class_out))
                classification_prob = float(probs[1])
            else:
                classification_prob = float(class_out)
    
    return classification_prob, segmentation_mask


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
                prob, mask_array = predict(model_obj, pil, IMG_SIZE)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                prob, mask_array = None, None

        with col2:
            st.subheader("Results")
            
            # Classification results
            if prob is not None:
                st.metric("Oil spill probability", f"{prob*100:.1f}%")
                is_oil = prob >= 0.5
                if is_oil:
                    st.success("Prediction: **OIL SPILL DETECTED**")
                else:
                    st.info("Prediction: **NO OIL SPILL**")
            else:
                st.warning("⚠️ Classification result not available")
                st.write("Check sidebar for debug info about model outputs")

            # Segmentation results
            if mask_array is not None:
                try:
                    bin_mask = postprocess_mask(mask_array, pil.size, threshold=0.5)
                    
                    # Calculate coverage
                    coverage = (bin_mask.sum() / bin_mask.size) * 100
                    st.metric("Predicted oil coverage", f"{coverage:.2f}%")
                    
                    # Create overlay
                    overlay = pil.convert('RGBA')
                    mask_img = Image.fromarray((bin_mask * 255).astype('uint8')).convert('L')
                    color_mask = Image.new('RGBA', pil.size, (255, 0, 0, 120))
                    overlay.paste(color_mask, (0, 0), mask_img)

                    st.image(overlay, caption='Overlay: red = predicted oil region', use_container_width=True)
                    st.image(bin_mask * 255, caption='Binary mask (white = predicted oil)', use_container_width=True)

                    # Download buttons
                    buf = io.BytesIO()
                    overlay.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button('📥 Download overlay PNG', data=buf, file_name='overlay.png', mime='image/png')

                    buf2 = io.BytesIO()
                    Image.fromarray((bin_mask * 255).astype('uint8')).save(buf2, format='PNG')
                    buf2.seek(0)
                    st.download_button('📥 Download mask PNG', data=buf2, file_name='mask.png', mime='image/png')
                    
                except Exception as e:
                    st.error(f"Failed to process segmentation mask: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Segmentation mask not available")
                st.write("Check sidebar for debug info about model outputs")

st.markdown("---")
st.subheader("Troubleshooting & tips")
st.markdown(
"""
- The model file is automatically downloaded from Google Drive if not found locally.
- **Debug info** is shown in the sidebar when you run a prediction
- If results look wrong, check the output shapes in the sidebar
- Default threshold for segmentation is 0.5 (50%)
- Ensure your model expects 256×256 input images
"""
)







