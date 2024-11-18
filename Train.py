import os
import cv2
import numpy as np
from PIL import Image

# Ensure the recognizer module is available
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("Make sure to install opencv-contrib-python to use LBPHFaceRecognizer.")
    exit()

# Define the path to the dataset
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print(f"Dataset directory '{dataset_path}' does not exist. Please create it and add training images.")
    exit()

def get_images_with_id(path):
    # List all image paths in the dataset directory
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faces = []
    ids = []

    for image_path in image_paths:
        # Convert each image to grayscale for consistency
        face_img = Image.open(image_path).convert('L')
        face_np = np.array(face_img, np.uint8)
        
        # Extract ID from the filename (assumes format user.<id>.<sample_number>.jpg)
        try:
            id = int(os.path.split(image_path)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"Skipping invalid file name format: {image_path}")
            continue

        print(f"ID: {id}, Image Path: {image_path}")

        faces.append(face_np)
        ids.append(id)

        # Display each face during the training (optional)
        cv2.imshow("Training", face_np)
        cv2.waitKey(10)

    return np.array(ids), faces

# Get images and their respective IDs
ids, faces = get_images_with_id(dataset_path)

# Train the recognizer and save the training data
if len(faces) > 0:
    recognizer.train(faces, ids)
    os.makedirs("recognizer", exist_ok=True)  # Ensure the directory exists
    recognizer.save("recognizer/trainingdata.yml")
    print("Training data saved successfully.")
else:
    print("No valid training data found. Please add images to the dataset directory.")

# Close any OpenCV windows
cv2.destroyAllWindows()