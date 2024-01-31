import cv2
import numpy as np
from sklearn.svm import SVC
from joblib import load
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the saved SVM model
model_filename = 'svm_model.joblib'
loaded_svm_model = load(model_filename)

# Function to extract faces from an image
def extract_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use a pre-trained face detector (like Haarcascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no faces detected or more than one face, return an empty array
    if len(faces_rect) != 1:
        return []

    # Extract the single face
    (x, y, w, h) = faces_rect[0]
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (50, 50))  # Resize for consistency

    return [face.flatten()]  # Return a list with a single flattened image array

# Function to handle the "Browse" button click event
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        update_image(file_path)

# Function to update the displayed image and predict age
def update_image(image_path):
    test_faces = extract_face(image_path)

    # Check if a face was successfully extracted
    if test_faces:
        # Make predictions on the test set
        predicted_ages = loaded_svm_model.predict(test_faces)

        # Display the image
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        # Update the predicted age label
        age_label.config(text=f"Predicted Age: {predicted_ages[0]}")
    else:
        age_label.config(text="No face detected or multiple faces found.")

# Create the main GUI window
root = tk.Tk()
root.title("Age Prediction App")

# Create and set up GUI components
browse_button = tk.Button(root, text="Browse", command=browse_image)
browse_button.pack(pady=10)

panel = tk.Label(root)
panel.pack(pady=10)

age_label = tk.Label(root, text="")
age_label.pack(pady=10)

# Run the GUI main loop
root.mainloop()