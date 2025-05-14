import cv2
import numpy as np
from sklearn.svm import SVC
from joblib import load
from skimage.feature import local_binary_pattern as sk_lbp

# Load the saved SVM model
model_filename = 'svm_model.joblib'
loaded_svm_model = load(model_filename)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract faces and their features from an image
def extract_faces(image):
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces_features = []
    for (x, y, w, h) in faces_rect:
        face = gray[y:y + h, x:x + w]
        if face.size == 0 or w < 10 or h < 10:
            continue
        face = cv2.resize(face, (50, 50))
        lbp = sk_lbp(face, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
        faces_features.append(((x, y, x + w, y + h), hist))
    return faces_features

# Initialize camera capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default camera, adjust if you have multiple cameras

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame. Exiting...")
        break

    # Extract faces and their features from the frame
    faces_info = extract_faces(frame)

    if faces_info:
        for (x1, y1, x2, y2), features in faces_info:
            try:
                # Predict age bin for the face
                predicted_bin = loaded_svm_model.predict([features])[0]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Display the predicted age bin and range
                cv2.putText(frame, f"Age Bin: {predicted_bin} ({predicted_bin * 10}-{predicted_bin * 10 + 9})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                cv2.putText(frame, f"Prediction error", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Face Age Prediction", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()