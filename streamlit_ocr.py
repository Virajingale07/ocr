import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
from pdf2image import convert_from_bytes

# Page Setup
st.set_page_config(page_title="Paper & PDF OCR Pro", page_icon="📝", layout="wide")
st.title("📝 Advanced Paper & PDF Text Extractor")

# --- SETTINGS ---
# If on Windows Local: Set path to poppler bin folder, e.g., r"C:\poppler\bin"
# If on Streamlit Cloud: Keep as None and use packages.txt
POPPLER_PATH = None


# 1. Logic: Load and Cache the OCR Model
@st.cache_resource
def load_ocr_model(languages):
    return easyocr.Reader(languages, gpu=False)


# 2. UI: Sidebar Configuration (THIS DEFINES use_preprocessing)
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


# 4. UI: File Uploader
uploaded_file = st.file_uploader("Upload Assignment PDF or Image", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file is not None:
    images_to_process = []

    # PDF vs Image Logic
    if uploaded_file.type == "application/pdf":
        try:
            with st.spinner("Converting PDF..."):
                pdf_bytes = uploaded_file.read()
                pdf_pages = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
                images_to_process = [np.array(page) for page in pdf_pages]
        except Exception as e:
            st.error(f"Poppler Error: {e}")
            st.info("Check if 'poppler-utils' is in your packages.txt file.")
    else:
        image = Image.open(uploaded_file)
        images_to_process = [np.array(image)]

    # 5. Extraction Logic
    if len(images_to_process) > 0:
        st.subheader("Document Preview")
        st.image(images_to_process[0], caption="First Page / Uploaded Image", width=500)

        if st.button("🚀 Start OCR Extraction"):
            with st.spinner("Extracting text..."):
                full_text_output = ""

                for i, img_np in enumerate(images_to_process):
                    # use_preprocessing is now safely defined in the sidebar above
                    processing_input = preprocess_image(img_np) if use_preprocessing else img_np

                    results = reader.readtext(processing_input)

                    page_text = f"--- PAGE {i + 1} ---\n"
                    for (bbox, text, prob) in results:
                        if prob >= conf_threshold:
                            page_text += text + "\n"

                    full_text_output += page_text + "\n"

                st.divider()
                st.subheader("📝 Extracted Content")
                st.text_area("Final Output", value=full_text_output, height=400)

                st.download_button(
                    label="📥 Download as Text File",
                    data=full_text_output,
                    file_name="assignment_extracted.txt",
                    mime="text/plain"
                )