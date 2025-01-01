from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from .forms import CustomAuthenticationForm
from .forms import ImageUploadForm
from .forms import VideoUploadForm
from PIL import Image
import numpy as np
import pytesseract
import os
import cv2
import re
from .utils import detect_number_plates
from django.contrib.auth import views as auth_views
from .models import *

from django.contrib.auth.forms import PasswordResetForm
from django.utils.crypto import get_random_string
from django.core.mail import send_mail
from .forms import ForgotPasswordForm



# Replace 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' with the actual installation path on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load mappings of state and district codes (this should be a dictionary or function you define)
state_district_map = {
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CG': 'Chhattisgarh',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JH': 'Jharkhand',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra',
    'MN': 'Manipur',
    'ML': 'Meghalaya',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OR': 'Odisha',
    'PB': 'Punjab',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TS': 'Telangana',
    'TR': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UK': 'Uttarakhand',
    'WB': 'West Bengal',
    'AN': 'Andaman and Nicobar Islands',
    'CH': 'Chandigarh',
    'DN': 'Dadra and Nagar Haveli and Daman and Diu',
    'DL': 'Delhi',
    'JK': 'Jammu and Kashmir',
    'LA': 'Ladakh',
    'LD': 'Lakshadweep',
    'PY': 'Puducherry'
    # Add other mappings as needed
}



# Function to extract state and district from the vehicle number
def extract_state_and_district(plate_text):
    # Extract the first two characters to identify the state
    state_code = plate_text[:2].upper()
    state = state_district_map.get(state_code, "Unknown State")
    # Add custom logic here if you want to extract district codes or other details from the plate text
    return state

# Home view
def home_view(request):
    return render(request, 'detection/index.html')

@login_required(login_url='/login/')
def services_view(request):
    return render(request, 'detection/services.html')


def contact_view(request):
    return render(request, 'detection/contact.html')
# def Homepage(request):
#     return render(request,'index.html')

# Register view

def login_register_view(request):
    if request.method == 'POST':
        if request.POST.get('type') == "Register":
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists.")
            elif User.objects.filter(email=email).exists():
                messages.error(request, "Email already registered.")
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                #login(request, user)
                messages.success(request, "Registration successful.")
                return redirect('/login/')

        else:
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, "Login successful.")
                return redirect('services')
            else:
                messages.error(request, "Invalid username or password.")

    return render(request, 'detection/login.html')


'''def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect('upload_image')
        else:
            messages.error(request, "Unsuccessful registration. Invalid information.")
    else:
        form = UserCreationForm()
    return render(request, 'detection/login.html', {'form': form})

# Login view
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, "Login successful.")
                return redirect('upload_image')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid login details.")
    else:
        form = AuthenticationForm()
    return render(request, 'detection/login.html', {'form': form})
'''
# Logout view
@login_required(login_url='login')
def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')



# Upload Image view for vehicle number plate detection
@login_required(login_url='login')
def upload_image_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            uploaded_image = UploadedImage(user=request.user, image=image)
            uploaded_image.save()

            try:
                img = Image.open(uploaded_image.image.path)

                # Use pytesseract to extract text from the image
                plate_text = pytesseract.image_to_string(img, config='--psm 8')
                plate_text = plate_text.replace("\n", "").strip()  # Clean up the text

                if plate_text:
                    # Extract state from the detected text
                    state = extract_state_and_district(plate_text)  # Modify this function to return only the state
                    result = f"Detected Vehicle Number: {plate_text}, State: {state}"
                    uploaded_image.plate_text = plate_text
                    uploaded_image.state = state
                else:
                    result = "No text detected. Please try again with a clearer image."
                    uploaded_image.plate_text = "No text detected"
                    uploaded_image.state = ""

                uploaded_image.save()

                # Return the result or render a template with the result
                return render(request, 'detection/result.html', {'prediction': result})

            except Exception as e:
                messages.error(request, f"An error occurred while processing the image: {str(e)}")
                return redirect('upload_image')
        else:
            messages.error(request, "Please upload a valid image.")

    else:
        form = ImageUploadForm()

    return render(request, 'detection/upload_image.html', {'form': form})


@login_required(login_url='login')
def dashboard(request):

    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    uploaded_images = UploadedImage.objects.filter(user=request.user)
    Upload_History = UploadHistory.objects.filter(user=request.user)
    errors = {}

    if request.method == 'POST':
        form_type = request.POST.get('form_type')

        if form_type == 'profile':
            gender = request.POST.get('gender')
            age = request.POST.get('age')
            height = request.POST.get('height')
            weight = request.POST.get('weight')

            if not gender:
                errors['gender'] = 'Gender is required.'
            if not age or int(age) < 0 or int(age) > 120:
                errors['age'] = 'Please enter a valid age.'
            if not height or int(height) <= 0 or int(height) > 300:
                errors['height'] = 'Please enter a valid height in cm.'
            if not weight or int(weight) <= 0 or int(weight) > 500:
                errors['weight'] = 'Please enter a valid weight in kg.'

            if not errors:
                user_profile.gender = gender
                user_profile.age = age
                user_profile.height = height
                user_profile.weight = weight
                user_profile.save()
                return redirect('dashboard')

        elif form_type == 'upload_image':
            image = request.FILES.get('image')
            if image:
                uploaded_image = UploadedImage(user=request.user, image=image)
                uploaded_image.save()
                return redirect('dashboard')

    return render(request, 'detection/dashboard.html', {
        'user_profile': user_profile,
        'uploaded_images': uploaded_images,
        'Upload_History':Upload_History,
        'errors': errors
    })



# View to upload video for processing
@login_required(login_url='/login/')
def video_upload_view(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = os.path.join('media', filename)

        cap = cv2.VideoCapture(video_path)
        detected_numbers = set()
        detected_states = set()
        frame_counter = 0
        frame_interval = 30

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            plate_text = number_plate_detection(frame)
            if plate_text and frame_counter % frame_interval == 0:
                detected_numbers.add(plate_text)
                detected_states.add(extract_state_and_district(plate_text))

            frame_counter += 1

        cap.release()

        # Save to UploadHistory
        UploadHistory.objects.create(
            user=request.user,
            file_name=video_file.name,
            file_type="Video",
            detected_plate_numbers=", ".join(detected_numbers),
            detected_states=", ".join(detected_states),
        )

        request.session['detected_numbers'] = list(detected_numbers)
        return redirect('upload_video_result')

    return render(request, 'upload_video.html')

# Number plate detection function
def number_plate_detection(img):
    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
        num_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if num_contours:
            contour_area = [cv2.contourArea(c) for c in num_contours]
            max_cntr_index = np.argmax(contour_area)
            max_cnt = num_contours[max_cntr_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            if not ratio_check(contour_area[max_cntr_index], w, h):
                return plate, None

            final_img = thresh[y:y + h, x:x + w]
            return final_img, [x, y, w, h]
        else:
            return plate, None

    def ratio_check(area, width, height):
        ratio = float(width) / float(height) if height > 0 else 0
        if area < 1063.62 or area > 73862.5 or ratio < 3 or ratio > 6:
            return False
        return True

    img2 = cv2.GaussianBlur(img, (5, 5), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel=element)
    num_contours, _ = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    for cnt in num_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        plate_img = img[y:y + h, x:x + w]
        clean_plate, rect = clean2_plate(plate_img)
        if rect:
            plate_im = Image.fromarray(clean_plate)
            text = pytesseract.image_to_string(plate_im, lang='eng')
            return "".join(re.split("[^a-zA-Z0-9]*", text)).upper()
    return ""


def normalize_plate_text(plate_text):
    """Normalize the plate text to handle minor variations in OCR results."""
    # Remove any unwanted characters, extra spaces, and make the text uppercase
    normalized_text = ''.join(re.split(r'\W+', plate_text)).upper()  # Only keep alphanumeric characters
    return normalized_text.strip()

# Display detected results from video
@login_required(login_url='/login/')
def upload_video_result(request):
    detected_numbers = request.session.get('detected_numbers', [])
    vehicle_data = [
        (number, state_district_map.get(number[:2].upper(), "Unknown State"))
        for number in detected_numbers
    ]
    return render(request, 'upload_video_result.html', {'vehicle_data': vehicle_data})

