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
    
    # Apply threshold BEFORE normalization (mask might be logits)
    # Apply sigmoid if values look like logits
    if mask.min() < 0 or mask.max() > 1:
        mask = 1.0 / (1.0 + np.exp(-mask))  # sigmoid
    
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
    
    # Parse outputs based on YOUR model structure:
    # Output 0: (1, 256, 256, 1) - Segmentation
    # Output 1: (1, 1) - Classification
    classification_prob = None
    segmentation_mask = None
    
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        # Your model: [segmentation, classification]
        seg_out = out[0]  # (1, 256, 256, 1)
        class_out = out[1]  # (1, 1)
        
        # Process classification output (1, 1) -> scalar
        class_val = float(np.squeeze(class_out))
        # Apply sigmoid if it's a logit
        if class_val < 0 or class_val > 1:
            classification_prob = 1.0 / (1.0 + np.exp(-class_val))
        else:
            classification_prob = class_val
        
        # Segmentation mask
        segmentation_mask = seg_out
        
    elif isinstance(out, (list, tuple)):
        # Fallback for other dual-output formats
        st.warning("Unexpected number of outputs, using first two")
        segmentation_mask = out[0]
        if len(out) > 1:
            classification_prob = float(np.squeeze(out[1]))
    else:
        # Single output
        if out.shape[-1] == 1 and len(out.shape) == 4:
            segmentation_mask = out
        else:
            classification_prob = float(np.squeeze(out))
    
    return classification_prob, segmentation_mask


# Sidebar controls
st.sidebar.header("Segmentation Settings")
threshold = st.sidebar.slider("Mask threshold", 0.0, 1.0, 0.5, 0.05, 
                               help="Higher = less sensitive, Lower = more sensitive")

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
                    bin_mask = postprocess_mask(mask_array, pil.size, threshold=threshold)
                    
                    # Calculate coverage
                    coverage = (bin_mask.sum() / bin_mask.size) * 100
                    st.metric("Predicted oil coverage", f"{coverage:.2f}%")
                    
                    # Create overlay (only show oil regions)
                    overlay = pil.convert('RGBA')
                    mask_img = Image.fromarray((bin_mask * 255).astype('uint8')).convert('L')
                    color_mask = Image.new('RGBA', pil.size, (255, 0, 0, 120))
                    overlay.paste(color_mask, (0, 0), mask_img)

                    st.image(overlay, caption='Overlay: red = predicted oil region', use_container_width=True)
                    
                    # Binary mask - BLACK for oil spill, WHITE for background
                    inverted_mask = (1 - bin_mask) * 255  # Invert: oil=0 (black), background=255 (white)
                    st.image(inverted_mask, caption='Binary mask (black = predicted oil)', use_container_width=True)

                    # Download buttons
                    buf = io.BytesIO()
                    overlay.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button('📥 Download overlay PNG', data=buf, file_name='overlay.png', mime='image/png')

                    buf2 = io.BytesIO()
                    Image.fromarray(inverted_mask.astype('uint8')).save(buf2, format='PNG')
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
- Use the threshold slider in the sidebar to adjust sensitivity
- Binary mask: **BLACK = oil spill**, WHITE = background
- If the overlay covers everything, try adjusting the threshold
"""
)






