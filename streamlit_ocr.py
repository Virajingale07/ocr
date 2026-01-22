import streamlit as st
import base64
from groq import Groq
from PIL import Image
import io

# Page Setup
st.set_page_config(page_title="Groq AI OCR Pro", page_icon="⚡", layout="wide")
st.title("⚡ High-Speed AI OCR")

# 1. Logic: Securely Retrieve API Key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key not found! Add 'GROQ_API_KEY' to Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')


# 2. UI: File Uploader
uploaded_file = st.file_uploader("Upload Assignment Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Document Preview")
        st.image(uploaded_file, use_container_width=True)

    if st.button("🚀 Extract Text with Groq"):
        with st.spinner("Llama 3.2 Vision is analyzing the page..."):
            try:
                base64_image = encode_image(uploaded_file)

                # 3. Logic: Call the NEW Model (Instant)
                # This model is faster and currently supported
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Transcribe the handwritten text in this image. "
                                            "Keep the formatting (bullet points, arrows) exactly as they appear. "
                                            "If you see technical terms like 'TF-IDF' or 'BERT', ensure they are spelled correctly."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                )

                extracted_text = response.choices[0].message.content

                with col2:
                    st.subheader("Extracted Content")
                    st.text_area("Final Transcription", value=extracted_text, height=450)
                    st.download_button("📥 Download .txt", extracted_text, "transcription.txt")

            except Exception as e:
                st.error(f"Processing Error: {e}")