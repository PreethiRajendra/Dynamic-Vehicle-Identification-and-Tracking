import cv2
import numpy as np
import joblib
import pytesseract  # For OCR on the detected number plate
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image, ImageEnhance, ImageFilter
from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User

# Load the number plate detection model (replace with your model path)
model_path = 'detection/svm_model.pkl'
plate_model = joblib.load(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Adjust the size to match the training data
    img = ImageEnhance.Sharpness(img).enhance(2.0)  # Enhance sharpness
    img = img.filter(ImageFilter.MedianFilter())  # Apply median filter to reduce noise
    img_array = np.array(img).flatten()  # Flatten the image into a 1D array
    return img_array, np.array(img)

# Enhanced preprocessing function for better OCR accuracy
def enhanced_preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel)
    
    return sharpened

# Detect number plate and extract text using OCR
def detect_number_plate(image: InMemoryUploadedFile, plate_mappings):
    features, img = preprocess_image(image)
    # Predict number plate region
    plate_region = plate_model.predict([features])
    # Crop to the detected region (this part assumes `plate_region` gives bounding box coordinates)
    x, y, w, h = plate_region[0]  # Adjust this based on actual model output
    plate_img = img[y:y+h, x:x+w]
    # Enhanced preprocessing
    plate_img_enhanced = enhanced_preprocess_image(plate_img)
    # Perform OCR on the cropped plate region
    plate_text = pytesseract.image_to_string(plate_img_enhanced, config='--psm 8')  # Adjust config if needed
    # Process the plate text to determine state and district
    state = extract_state_district(plate_text, plate_mappings)
    return plate_text.strip(), state

# Map number plate text to state and district
def extract_state_district(plate_text, plate_mappings):
    # Extract state code from plate text
    plate_text = plate_text.replace(" ", "").upper()  # Remove spaces and convert to uppercase
    state_code = plate_text[1:3] if plate_text[0].isdigit() else plate_text[:2]
    if state_code in plate_mappings:
        state = plate_mappings[state_code]
        return state
    else:
        return "Unknown State"

# Example usage in a Django view
def handle_uploaded_image(image: InMemoryUploadedFile, plate_mappings):
    plate_text, state = detect_number_plate(image, plate_mappings)
    if state == "Unknown State":
        return f"Detected Vehicle Number: {plate_text}, State: {state}"
    else:
        return f"Detected Vehicle Number: {plate_text}, State: {state}"

# Function to detect and return license plate details from an image or video frame
def detect_number_plates(image_path, plate_mappings):
    """
    Function to detect and return license plate details from an image or video frame.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    dict: Information about the detected license plate (e.g., plate number, region).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not loaded correctly")

    # Enhanced preprocessing
    enhanced_image = enhanced_preprocess_image(image)

    # Perform OCR on the image
    plate_text = pytesseract.image_to_string(enhanced_image, config='--psm 8')

    # Process the plate text to determine state and district
    state = extract_state_district(plate_text, plate_mappings)

    # Return the results
    result = {
        "plate_number": plate_text.strip(),
        "state": state,
        "vehicle_type": "Car"  # This can be adjusted based on additional logic
    }
    return result