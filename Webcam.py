import numpy as np
import cv2
import dlib
import time
import pygame
from scipy.spatial import distance as dist
from imutils import face_utils

# Function to calculate lip distance
def calculate_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance

# Function to update yawn status and duration
def update_yawn_status(yawn_status):
    global prev_yawn_status, yawn_start_time, yawn_end_time, total_yawns
    if yawn_status != prev_yawn_status:
        if yawn_status:  # Yawn started
            yawn_start_time = time.time()
        else:  # Yawn ended
            yawn_end_time = time.time()
            yawn_duration = yawn_end_time - yawn_start_time
            if yawn_duration > 1:  # Consider yawn if duration > 1 second
                total_yawns += 1
                play_alert_sound()  # Play sound alert
        prev_yawn_status = yawn_status

# Function to play sound alert
def play_alert_sound():
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load the sound file
    sound_file = "beep-warning-6387.mp3"  
    # Load the sound
    sound = pygame.mixer.Sound(sound_file)
    # Play the sound
    sound.play()

# Function to process frames
def process_frames():
    global cam, face_model, landmark_model, yawn_threshold, total_yawns
    while True:
        suc, frame = cam.read()

        if not suc:
            break

        # Convert frame to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_model(img_gray)
        for face in faces:
            shapes = landmark_model(img_gray, face)
            shape = face_utils.shape_to_np(shapes)

            # Calculate lip distance
            lip_distance = calculate_yawn(shape)
            if lip_distance > yawn_threshold:
                update_yawn_status(True)
            else:
                update_yawn_status(False)

        # Display total yawns count
        cv2.putText(frame, f'Total Yawning: {total_yawns}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load face and landmark detection models
    face_model = dlib.get_frontal_face_detector()
    landmark_model = dlib.shape_predictor("data\shape_predictor_68_face_landmarks.dat")

    # Initialize variables
    yawn_threshold = 29
    prev_yawn_status = False
    yawn_start_time = None
    yawn_end_time = None
    total_yawns = 0

    # Initialize webcam
    cam = cv2.VideoCapture(0)

    # Process frames
    process_frames()

