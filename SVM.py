import os
import numpy as np
import cv2
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Function to extract faces from images
def extract_faces(images_folder, current_year):
    faces = []
    labels = []

    for filename in os.listdir(images_folder):
        path = os.path.join(images_folder, filename)
        birth_year = int(filename.split('_')[1].split('-')[0])  # Assuming file format is label_birthdate.jpg

        # Read the image
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use a pre-trained face detector (like Haarcascades)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Extract faces and labels
        for (x, y, w, h) in faces_rect:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))  # Resize for consistency
            faces.append(face.flatten())  # Flatten the image array
            # Calculate age based on birth year and current year
            age = current_year - birth_year
            labels.append(age)

    return faces, labels

# Function to create a data generator with data augmentation
def create_data_generator(X_train, y_train, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(np.array(X_train), np.array(y_train), batch_size=batch_size)

# Load images and labels
images_folder = r'D:\faces\02'
current_year = datetime.now().year
faces, labels = extract_faces(images_folder, current_year)

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
sample_images = X_test[:100]
sample_labels = y_test[:100]
predictions = svm_model.predict(sample_images)

for i in range(len(sample_images)):
    plt.subplot(1, len(sample_images), i + 1)
    plt.imshow(np.reshape(sample_images[i], (50, 50)), cmap='gray')
    plt.title(f"True Age: {sample_labels[i]}, Predicted Age: {predictions[i]}")
    plt.axis('off')

plt.show()