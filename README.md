# Yawning-Detection-
Yawn Detection using OpenCV and dlib

# Yawn Detection using OpenCV and dlib

This project demonstrates real-time yawn detection using OpenCV and dlib in Python. It detects facial landmarks and calculates the distance between the top and bottom lip to determine if a yawn is occurring. When a yawn is detected, it plays an alert sound.

## Installation

To run this project, you need to have Python installed on your system. You also need to install the required dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Additionally, you need to download the pre-trained facial landmark detection model (`shape_predictor_68_face_landmarks.dat`) from the [dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

## Usage

Run the `yawn_detection.py` script to start the yawn detection system.

```bash
python yawn_detection.py
```

Press 'q' to exit the application.

## Configuration

You can adjust the yawn detection threshold and sound alert in the `yawn_detection.py` script. Modify the `yawn_threshold` variable to set the lip distance threshold for detecting a yawn, and change the sound file path in the `play_alert_sound()` function to customize the alert sound.

## Dependencies

- numpy
- opencv-python
- dlib
- pygame
- imutils
- scipy



## Acknowledgments

This project was inspired by the work of [PyImageSearch](https://www.pyimagesearch.com/) and utilizes the dlib library for facial landmark detection.

---
