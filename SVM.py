# Import necessary libraries
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to extract faces from images with age information in filenames
def extract_faces(images_folder):
    faces = []
    ages = []

    for filename in os.listdir(images_folder):
        path = os.path.join(images_folder, filename)
        age = int(filename.split('A')[1][:2])  # Extract two digits after 'A'
        
        # Read the image
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use a pre-trained face detector (like Haarcascades)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Extract faces and ages
        for (x, y, w, h) in faces_rect:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))  # Resize for consistency
            faces.append(face.flatten())  # Flatten the image array
            ages.append(age)

    return faces, ages

# Load images and labels
images_folder = r'C:\Users\laith\Desktop\archive\FGNET\images'
faces, ages = extract_faces(images_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, ages, test_size=0.2, random_state=42)

# Create and train the Support Vector Machine model
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Test and predict on the last 10 images in the folder
last_10_images = sorted(os.listdir(images_folder))[-10:]

test_faces, true_ages = extract_faces(images_folder)

# Make predictions on the test set
predicted_ages = svm_model.predict(test_faces)
# Calculate accuracy
accuracy = accuracy_score(true_ages, predicted_ages)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize predictions for the last 10 images
for i in range(len(last_10_images)):
    plt.subplot(1, len(last_10_images), i + 1)
    plt.imshow(np.reshape(test_faces[i], (50, 50)), cmap='gray')
    plt.title(f"True: {true_ages[i]}, Predicted: {predicted_ages[i]}")
    plt.axis('off')

plt.show()
