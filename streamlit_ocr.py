import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2

# Page Setup
st.set_page_config(page_title="Paper OCR", page_icon="📄")
st.title("📄 Paper Text Extractor")


# 1. Logic: Load and Cache the OCR Model
@st.cache_resource
def load_ocr_model():
    # Adding 'en' for English. You can add 'hi' for Hindi support.
    return easyocr.Reader(['en'], gpu=False)


reader = load_ocr_model()

# 2. UI: File Uploader
uploaded_file = st.file_uploader("Upload an image of text on paper", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert uploaded file to an image PIL can read
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Extract Text"):
        with st.spinner("Processing image logic..."):
            # Convert PIL image to NumPy array for EasyOCR/OpenCV
            img_np = np.array(image)

            # 3. Detection & Recognition Logic
            results = reader.readtext(img_np)

            # Process results: Join text and draw boxes
            full_text = ""
            annotated_image = img_np.copy()

            for (bbox, text, prob) in results:
                full_text += text + "\n"

                # Visual Logic: Drawing bounding boxes on the result
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)

            # 4. Display Results
            st.subheader("Extracted Text")
            st.text_area("Final Output", value=full_text, height=250)

            st.subheader("Detection Logic Visualization")
            st.image(annotated_image, caption="Green boxes show where text was detected.")

            # 5. Download Logic
            st.download_button(
                label="📥 Download as .txt file",
                data=full_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )