import cv2
import numpy as np
from sklearn.svm import SVC
from joblib import load

# Load the saved SVM model
model_filename = 'svm_model.joblib'
loaded_svm_model = load(model_filename)

# Function to extract faces from an image
def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    return [(x, y, w, h), face.flatten()]  # Return face rectangle and flattened image array

# Initialize camera capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust if you have multiple cameras

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Extract face from the frame
    face_info = extract_face(frame)

    if face_info:
        # Unpack face information
        (x, y, w, h), face = face_info

        # Make prediction on the face
        predicted_age = loaded_svm_model.predict([face])[0]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the predicted age
        cv2.putText(frame, f"Age: {predicted_age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Face Age Prediction", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
