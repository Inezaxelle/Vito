import cv2
import sqlite3
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np

# Function to ensure the database has the required table and optionally add test data
def initialize_database():
    try:
        conn = sqlite3.connect('models/customer_faces_data.db')
        c = conn.cursor()
        # Create table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS customers (
                        customer_uid INTEGER PRIMARY KEY,
                        customer_name TEXT NOT NULL
                    )''')
        conn.commit()
    except sqlite3.Error as e:
        print("SQLite error:", e)
    finally:
        conn.close()

# Initialize the database
initialize_database()

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load the sign recognition model
signModel = load_model("keras_model.h5", compile=False)

# Load the labels for sign recognition
class_names = open("labels.txt", "r").readlines()

# Parameters for displaying text
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30

nametagColor = (100, 180, 0)
nametagHeight = 50

faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Disable scientific notation for clarity in sign recognition
np.set_printoptions(suppress=True)

# Start looping
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face found
    for (x, y, w, h) in faces:
        # Recognize the face
        customer_uid, Confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
        # Connect to SQLite database
        try:
            conn = sqlite3.connect('customer_faces_data.db')
            c = conn.cursor()
        except sqlite3.Error as e:
            print("SQLite error:", e)

        c.execute("SELECT customer_name FROM customers WHERE customer_uid LIKE ?", (f"{customer_uid}%",))
        row = c.fetchone()
        if row:
            customer_name = row[0].split(" ")[0]
        else:
            customer_name = "Unknown"

        if 45 < Confidence < 100:
            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleBorderColor, faceRectangleBorderSize)

            # Display name tag
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, str(customer_name) + ": " + str(round(Confidence, 2)) + "%", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

    # Sign recognition
    # Resize the raw image into (224-height,224-width) pixels
    sign_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape
    sign_image = np.asarray(sign_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    sign_image = (sign_image / 127.5) - 1

    # Predict the sign
    prediction = signModel.predict(sign_image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Debugging information
    print(f"Raw predictions: {prediction}")
    print(f"Predicted class: {class_name}, Confidence: {confidence_score}")

    # Display the sign prediction
    cv2.putText(frame, f"Sign: {class_name} ({str(np.round(confidence_score * 100))[:-2]}%)", (10, 30), fontFace, fontScale, fontColor, fontWeight)

    # Display the resulting frame
    cv2.imshow('Face and Sign Detection', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
