import pytesseract
import cv2
from PIL import Image
import numpy as np
import requests

# Configure Tesseract path (Adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract text from an image
def extract_text_from_image(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            return "Error: Could not load image."

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding for better OCR
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Convert image to PIL format
        pil_image = Image.fromarray(gray)

        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(pil_image)

        return extracted_text.strip() if extracted_text.strip() else "No readable text found."
    except Exception as e:
        return f"Error processing image: {str(e)}"

# API call function
def detect_harassment(text):
    url = "http://127.0.0.1:5000/detect"
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to analyze text"}
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

# Example usage
image_path = "im2.png"  # Change to your screenshot's filename
extracted_text = extract_text_from_image(image_path)
print("Extracted Text:\n", extracted_text)

if extracted_text and "Error" not in extracted_text:
    result = detect_harassment(extracted_text)
    print("Prediction:", result)
else:
    print("No valid text found in image.")
