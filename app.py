# Streamlit app to classify and segment oil spills using a dual-head UNet model
# Save this file as streamlit_app.py and run: streamlit run streamlit_app.py
# Assumptions & notes:
# - The model file is stored on Google Drive (single file .pt/.pth/.h5 or .zip containing the model).
# - Set the environment variable DRIVE_FILE_ID with the Google Drive file ID OR paste a public shareable link in the UI.
# - The model is either a PyTorch state_dict / scripted model (.pt/.pth) or a Keras .h5 model. The loader tries both.
# - Default image size (IMG_SIZE) is 256. Change if your model expects a different input size.

import os
import io
import zipfile
import tempfile
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# try import torch and tensorflow when available
try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Oil-spill Detector & Segmenter")

st.title("Oil-spill Detector — Dual-head UNet")
st.write("Upload an image; the app will predict whether it contains an oil spill and show the segmented region.")

# Sidebar configuration
st.sidebar.header("Model configuration")
IMG_SIZE = st.sidebar.number_input("Model input size (square)", min_value=32, max_value=2048, value=256, step=32)
MODEL_FILENAME = st.sidebar.text_input("Local model filename after download (optional)", value="model.pt")
DRIVE_FILE_ID = st.sidebar.text_input("Google Drive file ID or shareable URL (optional)")
TRY_DOWNLOAD = st.sidebar.checkbox("Download model from Google Drive now", value=False)

@st.cache_data
def download_from_gdrive(drive_id_or_url: str, dest: str):
    """Download file from Google Drive using gdown (will install if missing).
    Accepts either a raw file ID or a full shareable URL.
    """
    try:
        import gdown
    except Exception:
        # try to install gdown
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"]) 
        import gdown

    # extract id from url if necessary
    fid = drive_id_or_url
    if "drive.google.com" in drive_id_or_url:
        # try to parse
        if "id=" in drive_id_or_url:
            fid = drive_id_or_url.split("id=")[-1].split("&")[0]
        else:
            # handle /d/<id>/
            parts = drive_id_or_url.split("/d/")
            if len(parts) > 1:
                fid = parts[1].split("/")[0]

    url = f"https://drive.google.com/uc?id={fid}&export=download"
    gdown.download(url, dest, quiet=False)
    return dest


def try_unzip_if_needed(path: str) -> str:
    """If path is a zip, unzip and return path to first candidate model file inside."""
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path, 'r')
        tmpdir = tempfile.mkdtemp()
        z.extractall(tmpdir)
        # find common model file types
        for ext in ('.pt', '.pth', '.h5', '.keras'):
            for p in Path(tmpdir).rglob(f'*{ext}'):
                return str(p)
        # if none found, return first file
        files = list(Path(tmpdir).glob('*'))
        if files:
            return str(files[0])
    return path


@st.cache_resource
def load_model(path: str):
    """Attempt to load a model. Returns a tuple (framework, model)
    framework in {'torch','keras'} or (None, None) on failure."""
    path = str(path)
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}")
        return None, None

    # Try PyTorch
    if TORCH_AVAILABLE:
        try:
            # try to load scripted or state_dict
            model = torch.load(path, map_location=torch.device('cpu'))
            model.eval()
            st.success("Loaded model with PyTorch loader")
            return 'torch', model
        except Exception as e:
            st.warning(f"PyTorch loader failed: {e}")

    # Try Keras
    if TF_AVAILABLE:
        try:
            model = keras.models.load_model(path, compile=False)
            st.success("Loaded model with Keras loader")
            return 'keras', model
        except Exception as e:
            st.warning(f"Keras loader failed: {e}")

    st.error("Failed to load model: unsupported format or missing frameworks (install torch or tensorflow)")
    return None, None


def preprocess_image_pil(img: Image.Image, size: int):
    """Convert PIL image to array tensor for model input.
    Returns numpy array and torch tensor (if torch available)."""
    img = img.convert('RGB')
    # resize while keeping aspect ratio and pad to square
    img = ImageOps.fit(img, (size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW
    chw = np.transpose(arr, (2,0,1))
    if TORCH_AVAILABLE:
        t = torch.from_numpy(chw).unsqueeze(0)  # 1 x C x H x W
        return arr, t
    else:
        # Keras expects NHWC
        nhwc = np.expand_dims(arr, 0)
        return arr, nhwc


def postprocess_mask(mask: np.ndarray, orig_size):
    """Postprocess predicted mask (assumes model outputs single-channel probability map).
    Resize to orig_size and threshold to binary mask.
    """
    # mask: HxW float
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize(orig_size, resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    bin_mask = (arr >= 0.5).astype(np.uint8)
    return bin_mask


def predict(framework, model, pil_image: Image.Image, size: int):
    """Run the model and return (is_oil_prob, mask_prob_map)
    This function expects the model to have two outputs for the dual-head UNet:
    - classification head: scalar or 1-d array with probability of oil spill
    - segmentation head: HxW probability map (same size as input)
    For PyTorch: expect model(input) -> (class_logits/probs, mask_logits)
    For Keras: expect model.predict(input) -> [class, mask] or a single array that can be split.
    """
    orig_size = pil_image.size[::-1]  # PIL size is (W,H) -> want (H,W) for resize functions
    arr, tensor = preprocess_image_pil(pil_image, size)

    if framework == 'torch':
        with torch.no_grad():
            out = model(tensor)
            # Handle various output shapes
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                class_out, mask_out = out[0], out[1]
            else:
                # try to guess: if out has 2 channels it's probably mask+class
                out_t = out
                if out_t.ndim == 4:
                    # assume batch x channels x H x W
                    # take first channel as class? risky. Instead assume model returns tuple.
                    mask_out = out_t
                    class_out = None
                else:
                    class_out = out_t
                    mask_out = None

            # process class
            is_oil_prob = None
            if class_out is not None:
                # convert logits to probability
                if class_out.ndim == 2 or class_out.ndim == 1:
                    # could be [batch, 1] or [batch]
                    v = class_out.squeeze()
                    # if multi-class, try softmax
                    if v.numel() > 1:
                        probs = torch.softmax(v, dim=0)
                        # assume index 1 corresponds to positive class
                        is_oil_prob = float(probs[-1].item())
                    else:
                        is_oil_prob = float(torch.sigmoid(v).item())
            # process mask
            mask_prob = None
            if mask_out is not None:
                # take first item in batch, first channel
                mo = mask_out.squeeze()
                if mo.ndim == 3:
                    # C x H x W
                    mo = mo[0]
                elif mo.ndim == 2:
                    pass
                mask_prob = mo.cpu().numpy()
                # normalize if logits
                if mask_prob.max() > 1.01:
                    mask_prob = 1.0 / (1.0 + np.exp(-mask_prob))

            return is_oil_prob, mask_prob, orig_size

    elif framework == 'keras':
        inp = tensor
        # model.predict returns list or array
        out = model.predict(inp)
        class_out = None
        mask_out = None
        if isinstance(out, list) or isinstance(out, tuple):
            if len(out) >= 2:
                class_out = out[0]
                mask_out = out[1]
        else:
            # single output: try to split
            if out.ndim == 4 and out.shape[-1] == 1:
                mask_out = np.squeeze(out, axis=0)[...,0]
            elif out.ndim == 2:
                class_out = out

        is_oil_prob = None
        if class_out is not None:
            v = np.squeeze(class_out)
            if v.size > 1:
                # softmax
                p = np.exp(v) / np.sum(np.exp(v))
                is_oil_prob = float(p[-1])
            else:
                is_oil_prob = float(1.0 / (1.0 + np.exp(-v)))

        mask_prob = None
        if mask_out is not None:
            mask_prob = np.squeeze(mask_out)
            # if in 0-255 range
            if mask_prob.max() > 1.01:
                mask_prob = mask_prob / 255.0

        return is_oil_prob, mask_prob, orig_size

    else:
        raise RuntimeError("Unsupported framework for prediction")


# UI: Model download / load
model_path = None
if TRY_DOWNLOAD and DRIVE_FILE_ID:
    with st.spinner("Downloading model from Google Drive..."):
        try:
            dest = MODEL_FILENAME if MODEL_FILENAME else os.path.basename(DRIVE_FILE_ID)
            downloaded = download_from_gdrive(DRIVE_FILE_ID, dest)
            model_candidate = try_unzip_if_needed(downloaded)
            model_path = model_candidate
            st.success(f"Downloaded model to {downloaded}; using {model_candidate}")
        except Exception as e:
            st.error(f"Download failed: {e}")

# Allow manual upload of model file
uploaded_model = st.file_uploader("Or upload a model file directly (.pt/.pth/.h5/.zip)", type=['pt','pth','h5','zip','keras'])
if uploaded_model is not None:
    tpath = Path(tempfile.mkdtemp()) / uploaded_model.name
    tpath.write_bytes(uploaded_model.read())
    model_candidate = try_unzip_if_needed(str(tpath))
    model_path = model_candidate

# If user provided Drive ID in sidebar but didn't check download, still show a button
if DRIVE_FILE_ID and not model_path:
    if st.sidebar.button("Download now"):
        try:
            dest = MODEL_FILENAME if MODEL_FILENAME else 'model_downloaded'
            downloaded = download_from_gdrive(DRIVE_FILE_ID, dest)
            model_candidate = try_unzip_if_needed(downloaded)
            model_path = model_candidate
            st.success(f"Downloaded model to {downloaded}; using {model_candidate}")
        except Exception as e:
            st.error(f"Download failed: {e}")

# Load once
framework, model_obj = (None, None)
if model_path:
    framework, model_obj = load_model(model_path)

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
        st.warning("Model not loaded yet. Either download from Drive or upload the model file in the sidebar.")
    else:
        with st.spinner("Running prediction..."):
            try:
                prob, mask_prob, orig_size = predict(framework, model_obj, pil, IMG_SIZE)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                prob, mask_prob, orig_size = None, None, None

        with col2:
            st.subheader("Results")
            if prob is not None:
                st.metric("Oil spill probability", f"{prob*100:.1f}%")
                is_oil = prob >= 0.5
                st.write("Prediction:", "**OIL SPILL**" if is_oil else "**NO OIL SPILL**")
            else:
                st.write("Classification result not available.")

            if mask_prob is not None:
                # postprocess mask
                bin_mask = postprocess_mask(mask_prob, pil.size)
                # overlay mask on original image
                overlay = pil.convert('RGBA')
                mask_img = Image.fromarray((bin_mask*200).astype('uint8')).convert('L')
                color_mask = Image.new('RGBA', pil.size, (255,0,0,120))
                overlay.paste(color_mask, (0,0), mask_img)

                st.image(overlay, caption='Overlay: red = predicted oil region', use_column_width=True)

                # show mask alone
                st.image(bin_mask*255, caption='Binary mask (white = predicted oil)', use_column_width=True)

                # Download buttons
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
- If the app fails to load the model: ensure the model file is a PyTorch `.pt`/`.pth` or Keras `.h5`. If your model is saved as a checkpoint (state_dict), you might need a small wrapper file that defines the model class and loads the state_dict.
- If your model expects a different input size or normalization, change the `IMG_SIZE` in the sidebar and adapt `preprocess_image_pil` (mean/std normalization).
- If your model was trained on multispectral images or additional channels, this app will need adaptation.
- To deploy publicly (Heroku/GCP/AWS), add the model file to cloud storage and set the DRIVE_FILE_ID or modify the code to fetch from cloud storage.
""")

st.subheader("Developer notes")
st.write(
"This app attempts to be framework-agnostic. If you want, I can adapt the loader to your exact model file format (send me the model type: PyTorch state_dict, PyTorch scripted, Keras h5, and the expected input shape/normalization)."
)









