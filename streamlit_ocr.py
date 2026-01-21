import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
from pdf2image import convert_from_bytes

# CONFIG: Set this ONLY if you are on Windows locally.
# On Streamlit Cloud, leave it as None.
POPPLER_PATH = None

st.set_page_config(page_title="Paper & PDF OCR Pro", page_icon="📝", layout="wide")
st.title("📝 Advanced Paper & PDF Text Extractor")


@st.cache_resource
def load_ocr_model(languages):
    return easyocr.Reader(languages, gpu=False)


# ... (Previous preprocessing logic remains the same) ...

uploaded_file = st.file_uploader("Upload Paper Image or WT Assignment PDF", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file is not None:
    images_to_process = []

    if uploaded_file.type == "application/pdf":
        try:
            with st.spinner("Converting PDF pages to images..."):
                pdf_bytes = uploaded_file.read()
                # Added poppler_path logic
                pdf_pages = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
                images_to_process = [np.array(page) for page in pdf_pages]
        except Exception as e:
            st.error(f"Poppler Error: {e}")
            st.info("Check if poppler-utils is in packages.txt (Cloud) or POPPLER_PATH is set (Local).")
    else:
        image = Image.open(uploaded_file)
        images_to_process = [np.array(image)]

    # ... (Rest of the OCR extraction loop) ...

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