 # Streamlit app for DeepLabV3+ Oil Spill Detection
# This app uses a dual-head DeepLabV3+ model for classification and segmentation

import os
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import gdown
from tensorflow import keras

st.set_page_config(layout="wide", page_title="Oil Spill Detector - DeepLabV3+")

st.title("🛢️ Oil Spill Detector — DeepLabV3+")
st.write("Upload a satellite/aerial image to detect and segment oil spills using state-of-the-art deep learning.")

# Configuration
IMG_SIZE = 256
MODEL_PATH = "models/deeplabv3_oil_spill.h5"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1jf6OW-jDqKgNGLkYJttNUd_yTo4GvlV3"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.info("📥 Model not found locally. Downloading from Google Drive...")
    os.makedirs("models", exist_ok=True)
    try:
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.info("Please upload your model file manually or update the GOOGLE_DRIVE_URL")

# Load model
@st.cache_resource
def load_model(path: str):
    """Load the DeepLabV3+ model"""
    try:
        # Load model without custom objects
        model = keras.models.load_model(path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model_obj = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Preprocessing
def preprocess_image(img: Image.Image, size: int):
    """Preprocess image for model input"""
    img = img.convert('RGB')
    img = ImageOps.fit(img, (size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr, np.expand_dims(arr, 0)

def postprocess_mask(mask: np.ndarray, target_size, threshold=0.5):
    """Process segmentation mask to match original image size"""
    # Remove batch and channel dimensions
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    
    # Ensure 2D
    if mask.ndim != 2:
        st.error(f"Unexpected mask shape: {mask.shape}")
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    # Normalize to [0, 1]
    if mask.max() > 1:
        mask = mask / 255.0
    
    # Resize to original size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_img.resize(target_size, resample=Image.BILINEAR)
    
    # Threshold
    mask_arr = np.array(mask_resized).astype(np.float32) / 255.0
    binary_mask = (mask_arr >= threshold).astype(np.uint8)
    
    return binary_mask

def predict(model, image: Image.Image, size: int):
    """Run inference on image"""
    _, img_tensor = preprocess_image(image, size)
    
    # Get predictions
    predictions = model.predict(img_tensor, verbose=0)
    
    # Parse outputs
    if isinstance(predictions, (list, tuple)) and len(predictions) == 2:
        seg_mask, classification = predictions
        class_prob = float(np.squeeze(classification))
    else:
        # Fallback if single output
        seg_mask = predictions
        class_prob = None
    
    return class_prob, seg_mask

# Sidebar settings
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Segmentation Threshold", 
    0.0, 1.0, 0.5, 0.05,
    help="Higher threshold = less sensitive (fewer false positives)"
)

opacity = st.sidebar.slider(
    "Overlay Opacity",
    0, 255, 120, 5,
    help="Transparency of red overlay on detected oil"
)

show_confidence = st.sidebar.checkbox("Show confidence heatmap", value=False)

# File uploader
st.header("📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a satellite or aerial image",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    help="Supported formats: JPG, PNG, TIFF"
)

# Main layout
col1, col2 = st.columns([1, 1])

if uploaded_file:
    # Load image
    pil_image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("📷 Input Image")
        st.image(pil_image, use_container_width=True)
        st.caption(f"Size: {pil_image.size[0]} × {pil_image.size[1]} pixels")
    
    if model_obj is None:
        st.error("⚠️ Model not loaded. Please check the model path.")
    else:
        with st.spinner("🔍 Analyzing image..."):
            try:
                # Run prediction
                class_prob, seg_mask = predict(model_obj, pil_image, IMG_SIZE)
                
                # Process segmentation mask
                binary_mask = postprocess_mask(seg_mask, pil_image.size, threshold)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                class_prob, binary_mask = None, None
        
        with col2:
            st.subheader("📊 Results")
            
            # Classification results
            if class_prob is not None:
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    st.metric("🎯 Oil Spill Probability", f"{class_prob*100:.1f}%")
                
                with col_b:
                    if class_prob >= 0.5:
                        st.error("⚠️ **OIL SPILL DETECTED**")
                    else:
                        st.success("✅ **NO OIL SPILL**")
            else:
                st.warning("⚠️ Classification unavailable")
            
            # Segmentation results
            if binary_mask is not None:
                # Calculate statistics
                total_pixels = binary_mask.size
                oil_pixels = binary_mask.sum()
                coverage_pct = (oil_pixels / total_pixels) * 100
                
                st.metric("🔴 Detected Oil Coverage", f"{coverage_pct:.2f}%")
                
                if oil_pixels > 0:
                    st.metric("📍 Affected Pixels", f"{oil_pixels:,} / {total_pixels:,}")
                
                # Visualization tabs
                tab1, tab2, tab3 = st.tabs(["🎨 Overlay", "⬛ Binary Mask", "🌡️ Heatmap"])
                
                with tab1:
                    # Create overlay
                    overlay = pil_image.convert('RGBA')
                    mask_img = Image.fromarray((binary_mask * 255).astype('uint8')).convert('L')
                    color_mask = Image.new('RGBA', pil_image.size, (255, 0, 0, opacity))
                    overlay.paste(color_mask, (0, 0), mask_img)
                    
                    st.image(overlay, caption='Red overlay = detected oil spill', use_container_width=True)
                
                with tab2:
                    # Inverted binary mask (black = oil)
                    inverted = (1 - binary_mask) * 255
                    st.image(inverted, caption='Black = oil spill, White = clean water', use_container_width=True)
                
                with tab3:
                    # Show confidence heatmap
                    if show_confidence and seg_mask is not None:
                        confidence_map = np.squeeze(seg_mask[0])
                        if confidence_map.max() > 1:
                            confidence_map = confidence_map / 255.0
                        
                        # Resize to original
                        conf_img = Image.fromarray((confidence_map * 255).astype(np.uint8))
                        conf_resized = conf_img.resize(pil_image.size, Image.BILINEAR)
                        
                        st.image(conf_resized, caption='Confidence map (brighter = higher confidence)', use_container_width=True)
                    else:
                        st.info("Enable 'Show confidence heatmap' in sidebar")
                
                # Download section
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Download overlay
                    buf_overlay = io.BytesIO()
                    overlay.save(buf_overlay, format='PNG')
                    buf_overlay.seek(0)
                    st.download_button(
                        '📥 Download Overlay',
                        data=buf_overlay,
                        file_name='oil_spill_overlay.png',
                        mime='image/png',
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Download binary mask
                    buf_mask = io.BytesIO()
                    Image.fromarray(inverted.astype('uint8')).save(buf_mask, format='PNG')
                    buf_mask.seek(0)
                    st.download_button(
                        '📥 Download Mask',
                        data=buf_mask,
                        file_name='oil_spill_mask.png',
                        mime='image/png',
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Segmentation unavailable")

# Information section
st.markdown("---")
st.subheader("ℹ️ About")

with st.expander("📖 How it works"):
    st.markdown("""
    This application uses **DeepLabV3+**, a state-of-the-art semantic segmentation model, combined with 
    a classification head to detect and segment oil spills in satellite/aerial imagery.
    
    **Model Architecture:**
    - **Backbone**: ResNet50 (pre-trained on ImageNet)
    - **Decoder**: DeepLabV3+ with ASPP (Atrous Spatial Pyramid Pooling)
    - **Dual Outputs**: 
        1. Binary classification (oil spill present/absent)
        2. Pixel-wise segmentation mask
    
    **Input**: RGB images (automatically resized to 256×256)
    
    **Output**: 
    - Classification probability
    - Segmentation mask highlighting oil-affected regions
    """)

with st.expander("🎯 Tips for best results"):
    st.markdown("""
    - **Image quality**: Higher resolution images generally give better results
    - **Threshold adjustment**: 
        - Lower threshold (0.3-0.4): More sensitive, may include false positives
        - Higher threshold (0.6-0.7): More conservative, may miss small spills
    - **Interpretation**: The model highlights areas with high probability of oil contamination
    - **Limitations**: Performance may vary with different lighting conditions, weather, and image sources
    """)

with st.expander("⚙️ Model Details"):
    if model_obj:
        st.write(f"**Total Parameters**: {model_obj.count_params():,}")
        st.write(f"**Input Shape**: {IMG_SIZE} × {IMG_SIZE} × 3")
        st.write(f"**Framework**: TensorFlow/Keras + Segmentation Models")

st.markdown("---")
st.caption("🛢️ Oil Spill Detection System | Powered by DeepLabV3+ | TensorFlow/Keras")

