# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import time

import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.signal import find_peaks

import mediapipe as mp

# facemesh model left and right eye landmarks indices
# https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(eye):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the eye aspect ratio (EAR) of a given eye.
    Introduced by Soukupová and Čech in their 2016 paper,
    Real-Time Eye Blink Detection Using Facial Landmarks.


    Parameters:
    ............
    eye : array
        Array of 6 tuples containing the coordinates of the eye landmarks

    Returns:
    ............
    ear : float
        The eye aspect ratio (EAR) of the given eye

    ---------------------------------------------------------------------------------------------------
    """

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def eye_blink_counter(video_directory):
    """
    ---------------------------------------------------------------------------------------------------

    This function counts the number of eye blinks in a given video.

    Parameters:
    ............
    video_directory : string
        The directory of the video to be analyzed

    Returns:
    ............
    framewise: np.ndarray
        An array containing the frame number and the eye aspect ratio (EAR) of each frame
    blinks : np.ndarray
        An array containing for each blink the frame number and the time (in seconds)
         start of the blink and end of the blink
    summary : list
        The number of eye blinks and blink rate (blinks per minute)

    ---------------------------------------------------------------------------------------------------
    """

    # Initialize MediaPipe Face Mesh
    print("[INFO] initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(video_directory)
    fps = vs.get(cv2.CAP_PROP_FPS)
    time.sleep(1.0)

    framewise = []
    blinks = []
    frame_n = 0

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame_n += 1
        frame = cv2.resize(frame, (450, int(frame.shape[0] * (450. / frame.shape[1]))))

        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            leftEye = np.array([(face_landmarks.landmark[lidx].x, face_landmarks.landmark[lidx].y) for lidx in LEFT_EYE_INDICES], dtype=np.float32)
            rightEye = np.array([(face_landmarks.landmark[ridx].x, face_landmarks.landmark[ridx].y) for ridx in RIGHT_EYE_INDICES], dtype=np.float32)

            # Calculate the EAR for each eye
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
        else:
            continue

        framewise.append([frame_n, ear])

    vs.release()

    framewise = np.array(framewise)
    troughs, properties = find_peaks(-framewise[:, 1], prominence=0.1, width=3)
    left_ips = properties["left_ips"]
    right_ips = properties["right_ips"]
    left_ips = np.round(left_ips).astype(int)
    right_ips = np.round(right_ips).astype(int)

    # convert the frame number to time
    troughs_time = troughs/fps
    left_ips_time = left_ips/fps
    right_ips_time = right_ips/fps
    blinks = np.array([troughs, left_ips, right_ips, troughs_time, left_ips_time, right_ips_time]).T
    summary = [len(troughs), len(troughs)/(frame_n/fps)*60]

    return framewise, blinks, summary
