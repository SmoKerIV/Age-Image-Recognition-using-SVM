# Import necessary libraries
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to extract faces from images
def extract_faces(images_folder):
    faces = []
    labels = []

    for filename in os.listdir(images_folder):
        path = os.path.join(images_folder, filename)
        label = int(filename.split('_')[0])  # Assuming file format is label_age.jpg

        # Read the image
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use a pre-trained face detector (like Haarcascades)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Extract faces and labels
        for (x, y, w, h) in faces_rect:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))  # Resize for consistency
            faces.append(face.flatten())  # Flatten the image array
            labels.append(label)

    return faces, labels

# Load images and labels
images_folder = 'path_to_your_dataset_folder'
faces, labels = extract_faces(images_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Create and train the Support Vector Machine model
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize a few predictions
sample_images = X_test[:5]
sample_labels = y_test[:5]
predictions = svm_model.predict(sample_images)

for i in range(len(sample_images)):
    plt.subplot(1, len(sample_images), i + 1)
    plt.imshow(np.reshape(sample_images[i], (50, 50)), cmap='gray')
    plt.title(f"True: {sample_labels[i]}, Predicted: {predictions[i]}")
    plt.axis('off')

plt.show()
