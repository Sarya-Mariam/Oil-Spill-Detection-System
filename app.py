import streamlit as st
from PIL import Image
from utils import predict_and_analyze

st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🌊 Oil Spill Detection App (TensorFlow)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    status, orig, mask, pixel_count, perc_area = predict_and_analyze(image)

    st.subheader(f"Detection Status: {status}")
    st.write(f"🟢 Spill pixels: {pixel_count} ({perc_area:.2f}% of image)")

    col1, col2 = st.columns(2)
    with col1:
        st.image(orig, caption="Original", use_column_width=True)
    with col2:
        st.image(mask, caption="Predicted Mask", use_column_width=True)









