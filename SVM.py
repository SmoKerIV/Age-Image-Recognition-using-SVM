import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
from joblib import dump, load
from tqdm import tqdm  # Import tqdm for progress bar

# Function to extract faces from images with age information in filenames
def extract_faces(images_folder):
    faces = []
    ages = []

    for filename in tqdm(os.listdir(images_folder), desc='Extracting Faces'):
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
images_folder = r'C:\Users\GTYaseen\Desktop\images'
faces, ages = extract_faces(images_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, ages, test_size=0.2, random_state=42)

# Create and train the Support Vector Machine model
svm_model = SVC(kernel='linear', C=1)
with tqdm(desc='Training SVM', total=len(X_train)) as pbar:
    svm_model.fit(X_train, y_train)
    pbar.update(len(X_train))

# Save the trained model to a file
model_filename = 'svm_model.joblib'
dump(svm_model, model_filename)
print(f"Trained model saved to {model_filename}")

# Test and predict on the last 5 images in the folder
last_5_images = sorted(os.listdir(images_folder))[-5:]

# Load the saved model
loaded_svm_model = load(model_filename)

test_faces, true_ages = extract_faces(images_folder)

# Make predictions on the test set
predicted_ages = loaded_svm_model.predict(test_faces)

# Calculate accuracy
accuracy = accuracy_score(true_ages, predicted_ages)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize predictions for the last 5 images
for i in range(len(last_5_images)):
    plt.subplot(1, len(last_5_images), i + 1)
    plt.imshow(np.reshape(test_faces[i], (50, 50)), cmap='gray')
    plt.title(f"True: {true_ages[i]}, Predicted: {predicted_ages[i]}")
    plt.axis('off')

plt.show()

# Write predictions to a CSV file
output_file = 'predictions.csv'
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['True Age', 'Predicted Age'])

    for true_age, predicted_age in zip(true_ages, predicted_ages):
        csv_writer.writerow([true_age, predicted_age])

print(f"Predictions written to {output_file}")
