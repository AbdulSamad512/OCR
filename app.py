import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import os
import google.generativeai as genai

# Set up Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set up Google Generative AI
os.environ["GOOGLE_API_KEY"] = "AIzaSyDnMiK1zy4TkrSDx3o9pgFnxFNtf9leqKA"  # Replace with your API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the generative model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# Custom functions
def image_to_text_with_ocr(img ):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(img)

def image_to_text_with_genai():
    """Extract text using Google Generative AI."""
    prompt = "Extract details from this image."
    response = model.generate(messages=[{"content": prompt}])  # Correct method to generate
    return response.get("content", "No text returned.")

def image_and_query_with_genai(query):
    """Generate content based on user query using Google Generative AI."""
    response = model.generate(messages=[{"content": query}])  # Correct method to generate
    return response.get("content", "No text returned.")

# Streamlit App
st.title("Image to Text Extractor & Generator")
st.write("Upload an image and choose between Tesseract OCR or Google Generative AI for text extraction and content generation.")

# File uploader in Streamlit
uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

# Text input for additional user query
query = st.text_input("Write a story or blog for this image")

if st.button("Generate"):
    if uploaded_image is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            # Extract text using Tesseract OCR
            ocr_text = image_to_text_with_ocr(image)
            st.subheader("Extracted Text (OCR)")
            st.write(ocr_text)

            # Extract text using Google Generative AI
            genai_text = image_to_text_with_genai()
            st.subheader("Extracted Text (Generative AI)")
            st.write(genai_text)

            # Generate content based on image and query
            if query:
                generated_text = image_and_query_with_genai(query)
                st.subheader("Generated Content")
                st.write(generated_text)
            else:
                st.warning("Please provide a query for content generation.")

            # Save data to CSV
            data = {
                "OCR Extracted Text": [ocr_text],
                "Generative AI Extracted Text": [genai_text],
                "Generated Content": [generated_text] if query else [None]
            }
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)

            st.download_button(
                label="Download Extracted Data as CSV",
                data=csv,
                file_name="extracted_text_and_generated_content.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image to proceed.")
