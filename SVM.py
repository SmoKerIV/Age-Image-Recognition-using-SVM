import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
from joblib import dump, load
from tqdm import tqdm
from skimage.feature import local_binary_pattern as sk_lbp

# Function to extract LBP features and age labels from UTKFace dataset
# Filename format: [age]_[gender]_[race]_[date&time].jpg

def extract_faces(images_folder):
    faces = []
    ages = []
    image_extensions = ['.jpg', '.jpeg', '.png']
    for filename in tqdm(os.listdir(images_folder), desc='Extracting Faces'):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            path = os.path.join(images_folder, filename)
            try:
                age = int(filename.split('_')[0])
            except Exception as e:
                print(f"Skipping file (bad name): {filename}")
                continue
            img = cv2.imread(path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(gray, (50, 50))
                lbp = sk_lbp(face, 8, 1, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
                faces.append(hist)
                ages.append(age)
            else:
                print(f"Failed to load image: {path}")
    return faces, ages

def age_to_bin(age):
    # 0-2, 3-5, 6-8, ..., 114-116 (3-year bins, up to 116)
    return min(age // 3, 38)

# Load images and labels
images_folder = r'E:\archive\utkface_aligned_cropped\UTKFace'
faces, ages = extract_faces(images_folder)
# Convert ages to bins for classification
ages = [age_to_bin(a) for a in ages]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, ages, test_size=0.2, random_state=42)

# Create and train the Support Vector Machine model with GridSearchCV
param_grid = {
'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
svm_model = grid.best_estimator_
print(f"Best SVM parameters: {grid.best_params_}")

# Save the trained model to a file
model_filename = 'svm_model.joblib'
dump(svm_model, model_filename)
print(f"Trained model saved to {model_filename}")

# Evaluate on test set
predicted_ages = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, predicted_ages)
print(f"Test Accuracy (age bins): {accuracy * 100:.2f}%")

# Visualize predictions for 5 random test images
import random
sample_idxs = random.sample(range(len(X_test)), min(5, len(X_test)))
for i, idx in enumerate(sample_idxs):
    plt.subplot(1, len(sample_idxs), i + 1)
    plt.title(f"True bin: {y_test[idx]} ({y_test[idx]*3}-{y_test[idx]*3+2})\nPred bin: {predicted_ages[idx]} ({predicted_ages[idx]*3}-{predicted_ages[idx]*3+2})")
    plt.axis('off')
plt.suptitle('Sample Predictions (3-year Age Bins, LBP features)')
plt.show()

# Write predictions to a CSV file
output_file = 'predictions.csv'
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['True Age Bin', 'Predicted Age Bin', 'True Age Range', 'Predicted Age Range'])
    for true_bin, pred_bin in zip(y_test, predicted_ages):
        csv_writer.writerow([
            true_bin, pred_bin,
            f"{true_bin*3}-{true_bin*3+2}",
            f"{pred_bin*3}-{pred_bin*3+2}"
        ])
print(f"Predictions written to {output_file}")
