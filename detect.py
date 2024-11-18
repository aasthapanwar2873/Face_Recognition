import cv2
import sqlite3

# Load the face detection model
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize the face recognizer and load training data
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read("recognizer/trainingdata.yml")
except cv2.error as e:
    print("Error loading training data:", e)
    exit(1)

# Function to retrieve profile information from the database
def get_profile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM students WHERE id = ?", (id,))
    profile = None
    for row in cursor:
        profile = row
        print("Profile data:", profile)
    conn.close()
    return profile

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predict the ID of the face
        id, confidence = recognizer.predict(gray[y: y+h, x: x+w])
        
        # Fetch profile information if ID is detected
        profile = get_profile(id)

        # Display profile information on the video frame
        if profile is not None:
            cv2.putText(img, f"Roll No. : {profile[0]}", (x, y+h+65), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 129), 2)
            cv2.putText(img, f"Name : {profile[1]}", (x, y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 129), 2)
        else:
            cv2.putText(img, "Unknown", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Display the video frame with face rectangles and profile information
    cv2.imshow("FACE", img)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
