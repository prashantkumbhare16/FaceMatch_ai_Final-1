from django import db
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_cors import CORS  # For Flutter app
from authlib.integrations.flask_client import OAuth
import firebase_admin
from firebase_admin import credentials, firestore, auth
from face_match import * # Importing the face_match module for face recognition logic
import re
import os
import json
import numpy as np
from PIL import Image
import logging 
from flask import flash
import requests
from functools import wraps

logging.basicConfig(level=logging.DEBUG)

FIREBASE_API_KEY = "your_firebase_api_key"  # Replace with your Firebase API key

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:  # Check if the user is logged in
            flash("You need to log in to access this page.")
            return redirect(url_for('login'))  # Redirect to login page
        return f(*args, **kwargs)
    return decorated_function

# start Route----------------------------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        logging.debug(f"Login attempt with email: {email}")

        try:
            # Use Firebase REST API to verify email and password
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyCcHH-YOo0waAvJ9UzZY0q8YH28c6hkpyw"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            response = requests.post(url, json=payload)
            response_data = response.json()

            if response.status_code == 200:
                # Login successful
                session['user'] = response_data['email']
                logging.debug(f"Login successful for user: {email}")
                return redirect(url_for('dashboard'))
            else:
                # Login failed
                error_message = response_data.get("error", {}).get("message", "Login failed")
                logging.error(f"Login failed: {error_message}")
                flash("Invalid email or password. Please try again.")
        except Exception as e:
            logging.error(f"An error occurred during login: {e}")
            flash("An unexpected error occurred. Please try again.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Check if all required fields are present
        required_fields = ['name', 'email', 'password', 'mobile']
        for field in required_fields:
            if (field not in request.form) or (not request.form[field]):
                return f"Missing field: {field}", 400
            
        # Get form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        mobile = request.form['mobile']

        # Validate Mobile Number with Regex
        mobile_pattern = re.compile(r"^\+[1-9]{1}[0-9]{1,3}-[0-9]{10}$")
        if not mobile_pattern.match(mobile):
            return "Invalid mobile number format. Please use +<country-code>-<10-digit-number>.", 400

        try:
            # Create User in Firebase Authentication
            user = auth.create_user(email=email, password=password)

            # Store user data in Firestore
            user_data = {
                'name': name,
                'email': email,
                'mobile': mobile,
                'password': password  # Add password here (not recommended in plaintext)
            }
            db.collection('users').add(user_data)

            return "User created successfully!", 200
        except firebase_admin.exceptions.FirebaseError as e:
            return f"Firebase error: {e}", 500
        except Exception as e:
            return f"An unexpected error occurred: {e}", 500
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    return redirect(url_for('login'))

# Route: Upload and Train Model
@app.route("/upload_and_train", methods=["POST", "GET"]) #
@login_required
def upload_and_train():
    if request.method == "GET":
        return render_template("train.html")  
    try:
        name = request.form.get("name")
        birthdate = request.form.get("birthdate")
        birthplace = request.form.get("birthplace")
        files = request.files.getlist("images")

        if not name or not birthdate or not birthplace or not files:
            return jsonify({"error": "All fields are required"}), 400
        

        person_folder = os.path.join(DATASET_FOLDER, name.strip())
        os.makedirs(person_folder, exist_ok=True)
        for file in files:
            file_path = os.path.join(person_folder, file.filename)
            file.save(file_path)
        
        user_details[name] = {"name": name, "birthdate": birthdate, "birthplace": birthplace}
        with open(DETAILS_FILE, "w") as f:
            json.dump(user_details, f, indent=4)

        num_faces = train_and_save_model()

        if num_faces > 0:
            return jsonify({"message": f"Model trained successfully with {num_faces} faces!"}), 200
        else:
            return jsonify({"error": "No faces found in dataset."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for searching faces from an uploaded image
@app.route("/search_faces", methods=["GET", "POST"])
@login_required
def search_faces():
    if request.method == "GET":
        return render_template("search.html")
    
    try:
        uploaded_file = request.files.get("image")
        if not uploaded_file:
            return jsonify({"error": "No image uploaded"}), 400
        
        image = Image.open(uploaded_file)
        image_rgb = np.array(image.convert("RGB"))
        
        matched_results = find_match(image_rgb)
        if matched_results:
            results = []
            for name, images in matched_results.items():
                user_info = user_details.get(name, {})
                birthdate = user_info.get("birthdate", "N/A")
                age = calculate_age(birthdate) if birthdate != "N/A" else "Unknown"
                birthplace = user_info.get("birthplace", "N/A")
                best_img_path, best_distance = images[0]
                
                best_img_path = f"/dataset/{os.path.relpath(best_img_path, DATASET_FOLDER)}"
                
                match_percentage = round((1 - best_distance) * 100, 2)
                results.append({
                    "name": name,
                    "birthdate": birthdate,
                    "age": age,
                    "birthplace": birthplace,
                    "matches": len(images),
                    "match_percentage": match_percentage,
                    "best_image": best_img_path,
                    "images": [
                        {"url": f"/dataset/{os.path.relpath(img[0], DATASET_FOLDER)}", "match_percentage": round((1 - img[1]) * 100, 2)}
                        for img in images
                    ]
                })

            return jsonify({"matched_faces": results}), 200
        else:
            return jsonify({"error": "No match found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for webcam image search
@app.route("/webcam_search", methods=["POST"])
@login_required
def webcam_search():
    try:
        photo = request.files.get("photo")
        if not photo:
            logging.error("No image captured")
            return jsonify({"error": "No image captured"}), 400

        image = Image.open(photo)
        image_rgb = np.array(image.convert("RGB"))
        
        matched_results = find_match(image_rgb)
        if matched_results:
            results = []
            for name, images in matched_results.items():
                user_info = user_details.get(name, {})
                birthdate = user_info.get("birthdate", "N/A")
                age = calculate_age(birthdate) if birthdate != "N/A" else "Unknown"
                birthplace = user_info.get("birthplace", "N/A")
                best_img_path, best_distance = images[0]
                
                match_percentage = round((1 - best_distance) * 100, 2)
                logging.debug(f"Matched {len(images)} images for {name} with best match percentage {match_percentage}%")
                
                results.append({
                    "name": name,
                    "birthdate": birthdate,
                    "age": age,
                    "birthplace": birthplace,
                    "matches": len(images),
                    "match_percentage": match_percentage,
                    "best_image": best_img_path,
                    "images": [
                        {"url": f"/dataset/{os.path.relpath(img[0], DATASET_FOLDER)}", "match_percentage": round((1 - img[1]) * 100, 2)}
                        for img in images
                    ]
                })

            return jsonify({"matched_faces": results}), 200
        else:
            logging.info("No match found")
            return jsonify({"error": "No match found"}), 404
    except Exception as e:
        logging.error(f"Error during webcam search: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/dataset/<path:filename>")
@login_required
def serve_dataset_image(filename):
    return send_from_directory(DATASET_FOLDER, filename)

@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    flash("You have been logged out.")
    return redirect(url_for('login'))

# Route for user login
if __name__ == '__main__':
    app.run()
#     app.run(debug=True, port=5000)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(debug=True, port=5000)


