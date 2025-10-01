import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import gdown, zipfile, os

# --- App Configuration ---
st.set_page_config(
    page_title="Oil Spill Segmentation",
    page_icon="🌊",
    layout="wide"
)

# --- Constants ---
IMG_SIZE = (256, 256)
MODEL_PATH = "dual_head_best.h5"
ZIP_URL = "https://drive.google.com/file/d/1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg/view?usp=drive_link"   # 🔑 Replace with your Google Drive file ID
ZIP_PATH = "dual_head_model.zip"
MIN_SPILL_AREA_PIXELS = 500

# ===================================================================
#  1. Define the U-Net Model Architecture
# ===================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# ===================================================================
#  2. Preprocessing
# ===================================================================
base_transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# ===================================================================
#  3. Download & Load Model
# ===================================================================
def download_and_extract_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        gdown.download(ZIP_URL, ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("Model extracted successfully!")

@st.cache_resource
def load_pytorch_model():
    try:
        download_and_extract_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNET(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None, None

# ===================================================================
#  4. Prediction Function
# ===================================================================
def predict_and_analyze(model, device, image_bytes):
    pil_image = Image.open(image_bytes).convert('RGB')
    image_np = np.array(pil_image)
    transformed = base_transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        predicted_mask_tensor = (probs > 0.5).float()

    spill_pixel_count = torch.sum(predicted_mask_tensor).item()
    if spill_pixel_count > MIN_SPILL_AREA_PIXELS:
        status = f"🚨 Oil Spill Detected ({int(spill_pixel_count)} pixels)"
        status_color = "red"
    else:
        status = "✅ No Spill Detected"
        status_color = "green"
        predicted_mask_tensor = torch.zeros_like(predicted_mask_tensor)

    inverted_mask_tensor = 1 - predicted_mask_tensor
    visible_mask = inverted_mask_tensor.squeeze().cpu().numpy() * 255

    return pil_image.resize(IMG_SIZE), visible_mask.astype(np.uint8), status, status_color

# ===================================================================
#  5. Streamlit UI
# ===================================================================
st.title("🌊 Oil Spill Detection System (PyTorch)")
st.markdown("Upload a satellite image to detect potential oil spills.")

model, device = load_pytorch_model()

if model:
    uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with st.spinner("Analyzing image..."):
            original, mask, status, color = predict_and_analyze(model, device, uploaded_file)

        st.markdown(f'<h2 style="color:{color}; text-align:center;">{status}</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original Image", use_container_width=True)
        with col2:
            st.image(mask, caption="Predicted Mask", use_container_width=True, channels="GRAY")
        st.success("✅ Analysis Complete")
else:
    st.warning("⚠️ Model could not be loaded.")









