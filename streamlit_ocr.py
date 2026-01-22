import streamlit as st
import base64
from groq import Groq
from PIL import Image
import io
import numpy as np
from pdf2image import convert_from_bytes

# --- CONFIGURATION ---
POPPLER_PATH = None  # Set locally for Windows if needed

st.set_page_config(page_title="Strict AI OCR", page_icon="📝", layout="wide")
st.title("📝 Clean Transcription (No Explanations)")

# 1. Retrieve API Key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key not found! Add 'GROQ_API_KEY' to Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)


def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# 2. File Uploader
uploaded_file = st.file_uploader("Upload Assignment", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file is not None:
    images_to_process = []

    if uploaded_file.type == "application/pdf":
        try:
            pdf_pages = convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
            images_to_process = pdf_pages
        except Exception as e:
            st.error(f"PDF Error: {e}")
    else:
        images_to_process = [Image.open(uploaded_file)]

    if images_to_process:
        if st.button("🚀 Get Raw Text"):
            with st.spinner("Transcribing..."):
                final_output = ""

                for i, page_img in enumerate(images_to_process):
                    b64_img = pil_to_base64(page_img)

                    # 3. Logic: Strict Prompting for Transcription Only
                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Your task is to transcribe the text in this image. "
                                                "Return ONLY the raw transcription. "
                                                "Do NOT provide an explanation, introduction, or summary. "
                                                "Do NOT use bullet points or bolding unless they are in the image. "
                                                "Just output the text as it is written."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                                    }
                                ],
                            }
                        ],
                    )

                    final_output += response.choices[0].message.content + "\n\n"

                st.subheader("Transcription Results")
                st.text_area("Final Output", value=final_output, height=400)
                st.download_button("📥 Download .txt", final_output, "transcription.txt")