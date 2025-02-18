from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all requests

# Load the harassment detection model
harassment_detector = pipeline("text-classification", model="unitary/toxic-bert")

# Function to extract text from an image using OCR
def extract_text_from_image(image):
    try:
        image = Image.open(image)
        image = image.convert("RGB")  # Ensure it's RGB format
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(gray)

        if not extracted_text.strip():
            return "No readable text found."

        return extracted_text.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Text-based harassment detection
@app.route('/detect', methods=['POST'])
def detect_text():
    data = request.json
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = harassment_detector(text)[0]
    label = "toxic" if result["score"] > 0.5 else "non-toxic"

    return jsonify({"label": label, "score": result["score"]})

# Image-based harassment detection
@app.route('/detect-image', methods=['POST'])
def detect_image():
    if "image" not in request.files:
        print("❌ No image found in request!")
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    print("✅ Image received:", image.filename)

    extracted_text = extract_text_from_image(image)
    print("📝 Extracted Text:", extracted_text)

    if "Error" in extracted_text or extracted_text == "No readable text found.":
        print("❌ OCR failed:", extracted_text)
        return jsonify({"error": extracted_text}), 400

    result = harassment_detector(extracted_text)[0]
    label = "toxic" if result["score"] > 0.5 else "non-toxic"

    print("✅ Harassment Detection Result:", label, result["score"])
    return jsonify({"label": label, "score": result["score"], "extracted_text": extracted_text})


if __name__ == '__main__':
    app.run(debug=True)
