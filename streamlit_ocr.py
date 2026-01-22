import streamlit as st
import base64
from groq import Groq
from PIL import Image
import io

# Page Setup
st.set_page_config(page_title="Groq AI OCR", page_icon="⚡")
st.title("⚡ AI OCR (Powered by Groq)")

# 1. Logic: Retrieve API Key from Streamlit Secrets
# This replaces the hardcoded string for security
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key not found in Streamlit Secrets! Please add 'GROQ_API_KEY' to your app settings.")
    st.stop()

# Initialize Client
client = Groq(api_key=groq_api_key)


def encode_image(image_file):
    # Logic: Read the file buffer and convert to base64
    return base64.b64encode(image_file.read()).decode('utf-8')


# 2. UI: File Uploader
uploaded_file = st.file_uploader("Upload Image for Extraction", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=400)

    if st.button("🚀 Run AI OCR"):
        with st.spinner("Llama 3.2 Vision is processing..."):
            try:
                # Logic: Prepare the image for the Vision API
                base64_image = encode_image(uploaded_file)

                # 3. Logic: Call Groq Llama 3.2 Vision Model
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please transcribe the text in this image accurately."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                )

                extracted_text = response.choices[0].message.content

                # 4. Results
                st.subheader("Extracted Content")
                st.text_area("Final Transcription", value=extracted_text, height=300)
                st.download_button("📥 Download .txt", extracted_text, "extracted_text.txt")

            except Exception as e:
                st.error(f"Processing Error: {e}")