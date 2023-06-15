# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import time
import json
import os
import logging

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from scipy.signal import find_peaks

import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

dir_name = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(dir_name, 'config/eye.json'))

file = open(config_path)
CONFIG = json.load(file)

# facemesh model left and right eye landmarks indices
# https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LEFT_EYE_INDICES = CONFIG['LEFT_EYE_INDICES']
RIGHT_EYE_INDICES = CONFIG['RIGHT_EYE_INDICES']


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


def initialize_facemesh():
    """
    ---------------------------------------------------------------------------------------------------

    This function initializes the MediaPipe Face Mesh model.

    Returns:
    ............
    face_mesh : object
        The MediaPipe Face Mesh model

    ---------------------------------------------------------------------------------------------------
    """
    logger.info("Initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh()


def get_video_capture(video_directory):
    """
    ---------------------------------------------------------------------------------------------------

    This function initializes the video capture.

    Parameters:
    ............
    video_directory : string
        The directory of the video to be analyzed

    Returns:
    ............
    vs : object
        The video capture object
    fps : float
        The fps of the video

    ---------------------------------------------------------------------------------------------------
    """
    logger.info("Starting video stream thread...")
    vs = cv2.VideoCapture(video_directory)
    fps = vs.get(cv2.CAP_PROP_FPS)
    time.sleep(1.0)
    return vs, fps


def process_frame(face_mesh, frame):
    """
    ---------------------------------------------------------------------------------------------------

    This function processes a frame with the MediaPipe Face Mesh model.

    Parameters:
    ............
    face_mesh : object
        The MediaPipe Face Mesh model
    frame : array
        The frame to be processed

    Returns:
    ............
    leftEye : array
        Array of 6 tuples containing the coordinates of the left eye landmarks
    rightEye : array
        Array of 6 tuples containing the coordinates of the right eye landmarks

    ---------------------------------------------------------------------------------------------------
    """
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        leftEye = np.array([(face_landmarks.landmark[lidx].x, face_landmarks.landmark[lidx].y) for lidx in LEFT_EYE_INDICES], dtype=np.float32)
        rightEye = np.array([(face_landmarks.landmark[ridx].x, face_landmarks.landmark[ridx].y) for ridx in RIGHT_EYE_INDICES], dtype=np.float32)
        return leftEye, rightEye
    return None, None


def detect_blinks(framewise, prominence, width):
    """
    ---------------------------------------------------------------------------------------------------

    This function detects the blinks from a given EAR array.

    Parameters:
    ............
    framewise : pd.DataFrame
        Contains the frame number and the eye aspect ratio (EAR) of each frame
    prominence : float
        The prominence of the peaks
    width : float
        The width of the peaks

    Returns:
    ............
    troughs : array
        Array containing the frame number of each blink
    left_ips : array
        Array containing the frame number of the start of each blink
    right_ips : array
        Array containing the frame number of the end of each blink

    ---------------------------------------------------------------------------------------------------
    """
    troughs, properties = find_peaks(-framewise['EAR'], prominence=prominence, width=width)
    left_ips = properties["left_ips"]
    right_ips = properties["right_ips"]
    left_ips = np.round(left_ips).astype(int)
    right_ips = np.round(right_ips).astype(int)
    return troughs, left_ips, right_ips


def convert_frame_to_time(troughs, left_ips, right_ips, fps):
    """
    ---------------------------------------------------------------------------------------------------

    This function converts the frame number to time.

    Parameters:
    ............
    troughs : array
        Array containing the frame number of each blink
    left_ips : array
        Array containing the frame number of the start of each blink
    right_ips : array
        Array containing the frame number of the end of each blink
    fps : float
        The fps of the video

    Returns:
    ............
    blinks : pd.DataFrame
        Contains for each blink the frame number and the time (in seconds)
         start of the blink and end of the blink

    ---------------------------------------------------------------------------------------------------
    """
    troughs_time = troughs/fps
    left_ips_time = left_ips/fps
    right_ips_time = right_ips/fps

    blinks = pd.DataFrame({'Blink Peak Frame': troughs, 'Blink Starting Frame': left_ips, 'Blink Ending Frame': right_ips,
                            'Blink Peak Time': troughs_time, 'Blink Starting Time': left_ips_time, 'Blink Ending Time': right_ips_time})
    return blinks


def eye_blink_rate(video_directory, device='laptop'):
    """
    ---------------------------------------------------------------------------------------------------

    This function counts the number of eye blinks in a given video.

    Parameters:
    ............
    video_directory : string
        The directory of the video to be analyzed
    device : string
        The device used to record the video. It can be either 'laptop' or 'mobile'

    Returns:
    ............
    framewise: pd.DataFrame
        Contains the frame number and the eye aspect ratio (EAR) of each frame
    blinks : pd.DataFrame
        Contains for each blink the frame number and the time (in seconds)
         start of the blink and end of the blink
    summary : pd.DataFrame
        The number of eye blinks and blink rate (blinks per minute)

    Raises:
    ............
    ValueError
        If device is not 'laptop' or 'mobile'

    ---------------------------------------------------------------------------------------------------
    """

    framewise, blinks, summary = None, None, None

    try:
        prominence, width = 2, .01

        face_mesh = initialize_facemesh()
        vs, fps = get_video_capture(video_directory)

        framewise = []
        frame_n = 0

        while True:
            ret, frame = vs.read()
            if not ret:
                break

            frame_n += 1
            frame = cv2.resize(frame, (450, int(frame.shape[0] * (450. / frame.shape[1]))))

            leftEye, rightEye = process_frame(face_mesh, frame)
            if leftEye is None:
                continue

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            framewise.append([frame_n, ear])

        framewise = pd.DataFrame(framewise, columns=['frame', 'EAR'])
        # z-score normalization
        framewise['EAR'] = (framewise['EAR'] - framewise['EAR'].mean()) / framewise['EAR'].std()
        # detect blinks from EAR array
        troughs, left_ips, right_ips = detect_blinks(framewise, prominence, width)
        # convert frame number to time and create blinks dataframe
        blinks = convert_frame_to_time(troughs, left_ips, right_ips, fps)
        # create summary dataframe
        summary_list = [len(troughs), len(troughs)/(frame_n/fps)*60]
        summary = pd.DataFrame(summary_list, index=['blinks', 'blink_rate'], columns=['value'])
    
    except Exception as e:
        logger.error(e)

    finally:
        if vs is not None:
            vs.release()
        return framewise, blinks, summary
