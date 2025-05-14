import cv2
import numpy as np
from sklearn.svm import SVC
from joblib import load
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.feature import local_binary_pattern as sk_lbp

# Load the saved SVM model
model_filename = 'svm_model.joblib'
loaded_svm_model = load(model_filename)

# Function to extract faces from an image
def extract_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return [], None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces_features = []
    for (x, y, w, h) in faces_rect:
        face = gray[y:y + h, x:x + w]
        if face.size == 0:
            continue
        face = cv2.resize(face, (50, 50))
        lbp = sk_lbp(face, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
        faces_features.append(hist)
    return faces_features, img

def age_to_bin(age):
    return min(age // 10, 8)

# Function to handle the "Browse" button click event
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        update_image(file_path)

# Function to update the displayed image and predict age
def update_image(image_path):
    test_faces, img = extract_faces(image_path)
    if img is not None:
        img_disp = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_disp.thumbnail((300, 300))
        img_disp = ImageTk.PhotoImage(img_disp)
        panel.configure(image=img_disp)
        panel.image = img_disp
    else:
        panel.configure(image=None)
        panel.image = None
    if test_faces:
        predicted_bins = loaded_svm_model.predict(test_faces)
        if len(predicted_bins) == 1:
            age_label.config(text=f"Predicted Age Bin: {predicted_bins[0]} (approx. {predicted_bins[0]*10}-{predicted_bins[0]*10+9})")
        else:
            age_label.config(text="Predicted Age Bins: " + ', '.join(f"{b} (approx. {b*10}-{b*10+9})" for b in predicted_bins))
    else:
        age_label.config(text="No face detected.")

# Create the main GUI window
root = tk.Tk()
root.title("Age Prediction App")
root.geometry("500x400")

# Create and set up GUI components
browse_button = tk.Button(root, text="Browse", command=browse_image, width=20, height=2)
browse_button.pack(pady=(200, 10))  # Adjust the pady to move the button down

panel = tk.Label(root)
panel.pack(pady=10)

age_label = tk.Label(root, text="")
age_label.pack(pady=10)

# Run the GUI main loop
root.mainloop()