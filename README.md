Age Estimation with SVM in Python
Overview
Welcome to the Age Estimation with Support Vector Machines (SVM) program in Python! This project leverages SVM, a powerful machine learning algorithm, to predict the age of individuals based on facial features extracted from images. The program provides two modes of prediction: one for images and another for real-time age estimation using your device's camera.

How to Use
For Image Age Prediction:
Installation:

Ensure you have Python installed on your system.
Install the required dependencies using the following command:

pip install opencv-python numpy scikit-learn matplotlib joblib tqdm
Run the Image Prediction:

Clone the repository and navigate to the directory.
Update the images_folder variable in the script to point to the folder containing the images.
Run the script python age_estimation_image.py.
View the predictions and accuracy in the console and visualize predictions for the last 5 images.
For Camera-based Age Prediction:
Additional Installation:

Install the additional library for camera access:

pip install pillow
Run the Camera Prediction:

Execute the script python age_estimation_camera.py.
A GUI window will open; click the "Browse" button to select an image, and the predicted age will be displayed.
Note: Adjust the camera index in the script if you have multiple cameras.
Model Persistence:
The trained SVM model is saved in svm_model.joblib after the image-based training. This saved model is loaded for both image and camera predictions.
***Dependencies
The project depends on the following Python libraries:
OpenCV
NumPy
scikit-learn
Matplotlib
Joblib
tqdm
Pillow (for camera-based prediction)
