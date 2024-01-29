import os
import re
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image, ImageTk
import joblib

def extract_age_labels(file_paths):
    age_labels = []
    for path in file_paths:
        age_match = re.search(r'A(\d+)', path)
        if age_match:
            age_labels.append(int(age_match.group(1)))
        else:
            age_labels.append(-1)
    return age_labels

def load_and_preprocess_images(image_paths, target_size=(299, 299)):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
    return np.array(images)

def extract_features(images):
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    return base_model.predict(images)

def result_window(predicted_ages, true_ages, image_paths, root):
    result_window = tk.Toplevel(root)
    result_window.title("Prediction Results")

    tree = ttk.Treeview(result_window, columns=("Predicted Age", "True Age", "Image"), show="headings")
    tree.heading("Predicted Age", text="Predicted Age")
    tree.heading("True Age", text="True Age")
    tree.heading("Image", text="Image")

    for predicted_age, true_age, img_path in zip(predicted_ages, true_ages, image_paths):
        tree.insert("", "end", values=(predicted_age, true_age, img_path))

    tree.pack()

    # Display images
    def show_image(event):
        selected_item = tree.selection()[0]
        img_path = tree.item(selected_item, "values")[2]

        img = Image.open(img_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        img_label.config(image=img)
        img_label.image = img

    tree.bind("<ButtonRelease-1>", show_image)

    img_label = tk.Label(result_window)
    img_label.pack()

# Create the main root window
root = tk.Tk()
root.title("Age Prediction")

# Load your dataset and prepare labels
data_dir = r'D:\images'
image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.lower().endswith(('.jpg', '.jpeg'))]
labels = extract_age_labels(image_paths)

# Load and preprocess images
X = load_and_preprocess_images(image_paths)

# Extract features using InceptionV3
X_features = extract_features(X)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, image_paths_train, image_paths_test = train_test_split(X_features, y, image_paths, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_pca, y_train)

# Save the trained model to a file named 'model'
joblib.dump(svm_model, 'model')

# Predict using the trained model
y_pred = svm_model.predict(X_test_pca)

# Decode labels back to original age values
predicted_ages = le.inverse_transform(y_pred)
true_ages = le.inverse_transform(y_test)

# Print the predicted and true ages
for predicted_age, true_age, img_path in zip(predicted_ages[:10], true_ages[:10], image_paths_test[:10]):
    print(f"Predicted Age: {predicted_age}, True Age: {true_age}, Image Path: {img_path}")

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2%}".format(accuracy))

# Show the result window
result_window(predicted_ages[:10], true_ages[:10], image_paths_test[:10], root)

# Start the Tkinter event loop
root.mainloop()

# Load the trained model from the file
loaded_model = joblib.load('model')

# Continue training if needed (replace this part with your actual training loop)
# For example, train for additional epochs
# loaded_model.fit(X_additional_train_pca, y_additional_train)

# Predict using the reloaded model
y_pred_reloaded = loaded_model.predict(X_test_pca)

# Evaluate the accuracy after reloading the model
accuracy_reloaded = accuracy_score(y_test, y_pred_reloaded)
print("Reloaded Model Accuracy: {:.2%}".format(accuracy_reloaded))
