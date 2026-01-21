import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2

# Page Setup
st.set_page_config(page_title="Paper OCR Pro", page_icon="📝", layout="wide")
st.title("📝 Advanced Paper Text Extractor")


# 1. Logic: Load and Cache the OCR Model
@st.cache_resource
def load_ocr_model(languages):
    # We pass languages as a list, e.g., ['en', 'hi']
    return easyocr.Reader(languages, gpu=False)


# 2. UI: Sidebar Configuration
st.sidebar.header("OCR Settings")
langs = st.sidebar.multiselect("Select Languages", ["en", "hi"], default=["en"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2)
use_preprocessing = st.sidebar.checkbox("Apply Image Preprocessing", value=True)

reader = load_ocr_model(langs)


# 3. Logic: Image Preprocessing Function
def preprocess_image(image_np):
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Adaptive Thresholding to handle uneven lighting/shadows on paper
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed


# 4. UI: File Uploader
uploaded_file = st.file_uploader("Upload an image of text on paper", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Extract Text"):
        with st.spinner("Analyzing text logic..."):

            # Apply preprocessing logic if selected
            processing_input = preprocess_image(img_np) if use_preprocessing else img_np

            # 5. Detection & Recognition Logic
            results = reader.readtext(processing_input)

            full_text = ""
            annotated_image = img_np.copy()

            for (bbox, text, prob) in results:
                if prob >= conf_threshold:
                    full_text += text + "\n"

                    # Drawing logic
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))

                    # Color logic: Green for high confidence, Red for low
                    color = (0, 255, 0) if prob > 0.5 else (255, 0, 0)
                    cv2.rectangle(annotated_image, top_left, bottom_right, color, 2)

            with col2:
                st.subheader("Detection Logic")
                st.image(annotated_image, caption="Green: High Confidence | Red: Review Needed")

            # 6. Display & Download
            st.divider()
            st.subheader("Extracted Text Content")
            st.text_area("Final Output", value=full_text, height=300)

            st.download_button(
                label="📥 Download .txt file",
                data=full_text,
                file_name="paper_scan.txt",
                mime="text/plain"
            )