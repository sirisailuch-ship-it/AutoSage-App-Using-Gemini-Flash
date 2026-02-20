import streamlit as st
import os
import time
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 1. Configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("🚨 API Key not found! Please check your .env file.")

# 2. Model Selection (Using 2026 Stable Version)
# gemini-1.5-flash is retired. gemini-2.5-flash is the new free-tier standard.
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

def get_gemini_response(input_prompt, image_data):
    """Fetches response with a retry mechanism for the Free Tier."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # We use the standard generate_content which works with stable v1
            response = model.generate_content([input_prompt, image_data[0]])
            return response.text
        except Exception as e:
            # If we hit a Rate Limit (429), wait and retry
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                st.warning(f"⚠️ Free Tier busy. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return f"❌ Error: {str(e)}"

def input_image_setup(uploaded_file):
    """Prepares image for API upload."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No image uploaded.")

# 3. Streamlit Interface
st.set_page_config(page_title="AutoSage AI", page_icon="🚗")
st.header("🚗 AutoSage: AI Vehicle Expert")

uploaded_file = st.file_uploader("Upload an image of a vehicle...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Analyze Vehicle Details")

# This prompt is optimized for the 2.5 series reasoning
input_prompt = """
Identify the vehicle in this image. Provide a detailed report including:
- Brand and Specific Model
- Engine specs (displacement, cooling)
- Fuel System (Is it carbureted or fuel-injected?)
- Key Features (Wheels, Braking, Drive type)
- Estimated Mileage (km/l or mpg)
- Professional Buyer's Advice
"""

if submit:
    if uploaded_file:
        with st.spinner("🤖 AutoSage is analyzing..."):
            image_data = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt, image_data)
            st.subheader("📋 Vehicle Analysis Report")
            st.markdown(response)
    else:
        st.warning("Please upload an image first!")

st.divider()
st.caption(f"Running on {MODEL_NAME} | Free Tier Mode")