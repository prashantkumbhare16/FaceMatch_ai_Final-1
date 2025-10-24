import os
import json
import pickle
import face_recognition
import numpy as np
from PIL import Image
from datetime import datetime

# Logic for Face Matching
# Define dataset paths
DATASET_FOLDER = "dataset"
DETAILS_FILE = "details.json"
MODEL_FILE = "trained_model.pkl"

# Ensure dataset directory exists
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# Load user details
# if os.path.exists(DETAILS_FILE):
#     with open(DETAILS_FILE, "r") as f:
#         user_details = json.load(f)
# else:
#     user_details = {}
if os.path.exists(DETAILS_FILE):
    try:
        with open(DETAILS_FILE, "r") as f:
            user_details = json.load(f)
    except json.JSONDecodeError:
        user_details = {}  # Fallback to an empty dictionary if JSON is invalid
else:
    user_details = {}

# Load dataset
def load_dataset():
    known_encodings, known_names, known_images = [], [], []
    for person in os.listdir(DATASET_FOLDER):
        person_path = os.path.join(DATASET_FOLDER, person)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                image_path = os.path.join(person_path, file)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_encodings.append(encoding[0])
                    known_names.append(person)
                    known_images.append(image_path)
    return known_encodings, known_names, known_images

# Train and save model
def train_and_save_model():
    known_encodings, known_names, known_images = load_dataset()
    data = {"encodings": known_encodings, "names": known_names, "images": known_images}
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(data, file)
    return len(known_encodings)

## search for faces --logics--
# Function to calculate age
def calculate_age(birthdate):
    birth_date = datetime.strptime(birthdate, "%Y-%m-%d")  # Correct usage
    today = datetime.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

# Find matches in dataset
def find_match(image_array):
    # Ensure the input is a numpy array
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input to find_match must be a numpy.ndarray")
    
    # Extract face encodings from the image
    unknown_encoding = face_recognition.face_encodings(image_array)
    if not unknown_encoding:
        return None
    
    with open(MODEL_FILE, "rb") as file:
        data = pickle.load(file)
    
    known_encodings, known_names, known_images = data["encodings"], data["names"], data["images"]
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0])
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding[0])
    
    matched_results = {}
    for i, match in enumerate(matches):
        if match:
            name = known_names[i]
            if name not in matched_results:
                matched_results[name] = []
            matched_results[name].append((known_images[i], face_distances[i]))
    
    # Sort images by lowest face distance (better match) & earliest timestamp
    for name in matched_results:
        matched_results[name].sort(key=lambda x: (x[1], os.path.getctime(x[0])))
    
    return matched_results
