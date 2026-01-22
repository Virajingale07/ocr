import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2

# --- CONFIGURATION ---
# If on Windows, you MUST point to the tesseract.exe location
# Example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/Cloud, this line is usually not needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Tesseract OCR Pro", page_icon="🔍", layout="wide")
st.title("🔍 Tesseract Paper Text Extractor")


# 1. Logic: Image Preprocessing (Crucial for Tesseract)
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# 2. UI: Sidebar
st.sidebar.header("Tesseract Settings")
psm = st.sidebar.selectbox("Page Segmentation Mode (PSM)",
                           [3, 4, 6, 11, 12], index=0,
                           help="3: Default, 6: Uniform block of text, 11: Sparse text")

# 3. UI: File Uploader
uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Extract Text"):
        with st.spinner("Tesseract is processing..."):
            # 4. Logic: Preprocessing for better accuracy
            gray = get_grayscale(img_np)
            thresh = thresholding(gray)

            # 5. OCR Execution
            # PSM 6 is generally best for uniform handwritten or printed pages
            custom_config = f'--oem 3 --psm {psm}'
            extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

            with col2:
                st.subheader("Processed Logic View")
                st.image(thresh, caption="What Tesseract 'sees' (Binarized)", use_container_width=True)

            # 6. Display Results
            st.divider()
            st.subheader("📝 Extracted Text")
            st.text_area("Final Output", value=extracted_text, height=300)

            st.download_button("📥 Download .txt", extracted_text, "tesseract_output.txt")