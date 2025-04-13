import os
import sys
import time
import cv2
import dlib
import numpy as np
import pyttsx3
import pygame
import threading
from scipy.spatial import distance
from twilio.rest import Client
from datetime import datetime
from flask import Flask, render_template, Response, request
from dotenv import load_dotenv

# Global flag to control detection
detection_active = False

# Default emergency contact (will be updated via form submission)
EMERGENCY_CONTACT = "+911234567890"

# Flask App
app = Flask(__name__)

# Initialize Pygame for alarm sound
pygame.mixer.init()
ALARM_SOUND_PATH = "alarm1.wav"
if not os.path.isfile(ALARM_SOUND_PATH):
    raise FileNotFoundError(f"Alarm sound file not found: {ALARM_SOUND_PATH}")
pygame.mixer.music.load(ALARM_SOUND_PATH)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Load environment variables
load_dotenv()

# Twilio credentials for SMS alert
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Load dlib shape predictor
predictor_path = os.path.join(os.getcwd(), "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(predictor_path):
    raise FileNotFoundError(f"Model file not found: {predictor_path}")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Landmark indices
RIGHT_EYE = list(range(36, 42))
LEFT_EYE  = list(range(42, 48))

# Thresholds and buffers
EYE_AR_THRESH = 0.21
HEAD_PITCH_THRESH = 25
HEAD_PITCH_BUFFER_SIZE = 5
head_pitch_buffer = []

eye_closed_start = None
alarm_active = False
sms_alert_sent = False

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(shape, frame_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye left corner
        (shape.part(45).x, shape.part(45).y),  # Right eye right corner
        (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = frame_size[0]
    center = (frame_size[0] / 2, frame_size[1] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x_angle = np.arctan2(rmat[2, 1], rmat[2, 2])
        y_angle = np.arctan2(-rmat[2, 0], sy)
        z_angle = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x_angle = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y_angle = np.arctan2(-rmat[2, 0], sy)
        z_angle = 0
    euler_angles = np.degrees(np.array([x_angle, y_angle, z_angle]))
    return success, rotation_vector, translation_vector, euler_angles

def send_sms_alert():
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_body = f"🚨 Alert: Driver drowsiness detected at {timestamp}! Please check on them immediately."
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Twilio error: {e}")

def speak_alert():
    tts_engine.say("Wake up! Please take a break.")
    tts_engine.runAndWait()

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device. Check camera connection or device index.")
    exit()

def generate_frames():
    global eye_closed_start, alarm_active, sms_alert_sent, head_pitch_buffer, detection_active
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        avg_head_pitch = 0

        # Run detection logic only if detection is active
        if detection_active and len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)

            # Head pose estimation
            success, rvec, tvec, euler_angles = get_head_pose(landmarks, (frame.shape[1], frame.shape[0]))
            if success:
                current_pitch = euler_angles[0]
                head_pitch_buffer.append(current_pitch)
                if len(head_pitch_buffer) > HEAD_PITCH_BUFFER_SIZE:
                    head_pitch_buffer.pop(0)
                avg_head_pitch = sum(head_pitch_buffer) / len(head_pitch_buffer)
                cv2.putText(frame, f"Pitch: {avg_head_pitch:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Eye landmarks and EAR calculation
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Drowsiness detection logic
            if avg_EAR > 0.27:
                eye_closed_start = None
                if alarm_active:
                    alarm_active = False
                    pygame.mixer.music.stop()
                sms_alert_sent = False
                head_pitch_buffer = []
            else:
                if avg_EAR < EYE_AR_THRESH or (avg_EAR < 0.3 and abs(avg_head_pitch) > HEAD_PITCH_THRESH):
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    else:
                        time_closed = time.time() - eye_closed_start
                        if time_closed >= 3 and not alarm_active:
                            alarm_active = True
                            pygame.mixer.music.play(-1)
                            threading.Thread(target=speak_alert, daemon=True).start()
                        if time_closed >= 10 and not sms_alert_sent:
                            send_sms_alert()
                            sms_alert_sent = True
                else:
                    eye_closed_start = None
                    if alarm_active:
                        alarm_active = False
                        pygame.mixer.music.stop()
                    sms_alert_sent = False
                    head_pitch_buffer = []
        else:
            head_pitch_buffer = []

        if alarm_active:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        if sms_alert_sent:
            cv2.putText(frame, "SMS ALERT SENT!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def phone_entry():
    # Render the landing page with the phone input
    return render_template('phone.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global EMERGENCY_CONTACT, detection_active
    phone = request.form.get('phone_number')
    if phone:
        EMERGENCY_CONTACT = phone
    detection_active = True  # Activate detection and camera feed
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Run Flask without debug reload to avoid threading issues
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        cap.release()
        pygame.mixer.music.stop()
        pygame.mixer.quit()
