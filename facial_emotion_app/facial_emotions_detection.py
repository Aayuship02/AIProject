import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

# Load the pre-trained facial cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize basic emotions based on face landmarks
def recognize_emotion(roi, gray):
    # Calculate the mean pixel intensity of the ROI
    mean_intensity = roi.mean()

    # Define emotion labels and their intensity thresholds
    emotions = {
        "Happy": mean_intensity > 100,
        "Sad": mean_intensity <= 100,
        "Angry": mean_intensity < 50,
    }

    # Get the emotion label with the highest intensity
    predicted_emotion = max(emotions, key=emotions.get)

    return predicted_emotion

# Function to generate video frames with detected emotions
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the region of interest (ROI) for the face
            roi = gray[y:y + h, x:x + w]

            # Recognize the emotion for the detected face
            emotion = recognize_emotion(roi, gray)

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to display the video feed with detected emotions
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
