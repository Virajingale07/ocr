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
    if uploaded_file.type == "application/pdf":
        with st.spinner("Converting PDF pages to images..."):
            # Convert PDF bytes to PIL images
            pdf_pages = convert_from_bytes(uploaded_file.read())
            images_to_process = [np.array(page) for page in pdf_pages]
            st.success(f"Successfully loaded {len(images_to_process)} pages from PDF.")
    else:
        # Standard Image Logic
        image = Image.open(uploaded_file)
        images_to_process = [np.array(image)]

    # Display First Page/Image for Preview
    st.subheader("Document Preview")
    st.image(images_to_process[0], caption="First Page / Uploaded Image", width=500)

    if st.button("🚀 Start OCR Extraction"):
        with st.spinner("Extracting text logic across all pages..."):

            full_text_output = ""

            # Loop through each page (Logic for Multi-page assignments)
            for i, img_np in enumerate(images_to_process):
                processing_input = preprocess_image(img_np) if use_preprocessing else img_np

                # 5. Detection & Recognition Logic
                results = reader.readtext(processing_input)

                page_text = f"--- PAGE {i + 1} ---\n"
                for (bbox, text, prob) in results:
                    if prob >= conf_threshold:
                        page_text += text + "\n"

                full_text_output += page_text + "\n"

            # 6. Display & Download Results
            st.divider()
            st.subheader("📝 Extracted Assignment Text")
            st.text_area("Final Output", value=full_text_output, height=400)

            st.download_button(
                label="📥 Download as Text File",
                data=full_text_output,
                file_name="assignment_extracted.txt",
                mime="text/plain"
            )