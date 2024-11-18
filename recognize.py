import cv2
import sqlite3
import os

# Load the face detection model
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Check if the dataset directory exists, if not, create it
if not os.path.exists("dataset"):
    os.makedirs("dataset")

def insert_update(id, name, age):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    # Check if record exists
    cursor.execute("SELECT * FROM students WHERE ID = ?", (id,))
    is_record_exist = cursor.fetchone() is not None

    # Update or insert the record based on existence
    if is_record_exist:
        cursor.execute("UPDATE students SET name = ?, age = ? WHERE id = ?", (name, age, id))
    else:
        cursor.execute("INSERT INTO students (id, name, age) VALUES (?, ?, ?)", (id, name, age))

    conn.commit()
    conn.close()

# Get user input for ID, name, and age
id = input("Enter user ID: ")
name = input("Enter name: ")
age = input("Enter age: ")

# Update the database with the user's details
insert_update(id, name, age)

sample_num = 0

while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        sample_num += 1
        # Save the captured face in the dataset folder
        cv2.imwrite(f"dataset/user.{id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)  # Wait briefly to avoid duplicating the same image

    # Display the resulting frame
    cv2.imshow("Face", img)

    # Press 'q' to quit or stop if we have enough samples
    if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 20:
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()