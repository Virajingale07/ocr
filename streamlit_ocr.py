import streamlit as st
import base64
from groq import Groq
from PIL import Image
import io
import numpy as np
from pdf2image import convert_from_bytes

# --- CONFIGURATION ---
# Set this ONLY if you are on Windows locally.
# On Streamlit Cloud, leave it as None and use packages.txt
POPPLER_PATH = None

st.set_page_config(page_title="AI Assignment Parser", page_icon="📝", layout="wide")
st.title("📝 WT Assignment AI Extractor")

# 1. Logic: Retrieve API Key from Secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key not found! Go to Streamlit Settings -> Secrets and add GROQ_API_KEY.")
    st.stop()

client = Groq(api_key=groq_api_key)


# Helper: Convert Image to Base64 for Groq
def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# 2. UI: File Uploader (Accepts Image and PDF)
uploaded_file = st.file_uploader("Upload Assignment (PDF or Image)", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file is not None:
    images_to_process = []

    # NEW LOGIC: Multi-page PDF Handling
    if uploaded_file.type == "application/pdf":
        try:
            with st.spinner("Converting PDF pages to images..."):
                pdf_pages = convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
                images_to_process = pdf_pages
                st.success(f"Loaded {len(images_to_process)} pages.")
        except Exception as e:
            st.error(f"PDF Error: {e}. Check if poppler is installed.")
    else:
        images_to_process = [Image.open(uploaded_file)]

    if images_to_process:
        st.subheader("Document Preview")
        st.image(images_to_process[0], caption="Page 1 Preview", width=500)

        if st.button("🚀 Run AI OCR"):
            with st.spinner("Llama 4 Vision is analyzing..."):
                all_text = ""

                for i, page_img in enumerate(images_to_process):
                    try:
                        b64_img = pil_to_base64(page_img)

                        # 3. Logic: Call Llama 4 Scout Vision
                        response = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text",
                                         "text": "Transcribe this assignment page accurately. Keep the formatting (bullet points, arrows). Use technical context (WT, TF-IDF, BERT, Redis) to fix errors."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                    ],
                                }
                            ],
                        )

                        page_out = response.choices[0].message.content
                        all_text += f"--- PAGE {i + 1} ---\n{page_out}\n\n"

                    except Exception as e:
                        st.error(f"Error on Page {i + 1}: {e}")

                # 4. Results
                st.divider()
                st.subheader("📝 Final Transcription")
                st.text_area("Content:", value=all_text, height=400)
                st.download_button("📥 Download Result", all_text, "assignment_transcription.txt")