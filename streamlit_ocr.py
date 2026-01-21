import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
# NEW IMPORT: For PDF support
from pdf2image import convert_from_bytes

# Page Setup
st.set_page_config(page_title="Paper & PDF OCR Pro", page_icon="📝", layout="wide")
st.title("📝 Advanced Paper & PDF Text Extractor")


# 1. Logic: Load and Cache the OCR Model
@st.cache_resource
def load_ocr_model(languages):
    return easyocr.Reader(languages, gpu=False)


# 2. UI: Sidebar Configuration
st.sidebar.header("OCR Settings")
langs = st.sidebar.multiselect("Select Languages", ["en", "hi"], default=["en"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2)
use_preprocessing = st.sidebar.checkbox("Apply Image Preprocessing", value=True)

reader = load_ocr_model(langs)


# 3. Logic: Image Preprocessing Function
def preprocess_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed


# 4. UI: File Uploader (Updated to allow PDF)
uploaded_file = st.file_uploader("Upload Paper Image or WT Assignment PDF", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file is not None:
    images_to_process = []

    # NEW LOGIC: Determine if it's a PDF or Image
    if uploaded_file.type == "application/