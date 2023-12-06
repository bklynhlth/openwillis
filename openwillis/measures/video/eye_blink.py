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


def create_empty_dataframes(config):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates empty dataframes for the output of the measures.

    Parameters:
    ...........
    config : dict
        The configuration dictionary

    Returns:
    ...........
    ear : pd.DataFrame
        Contains the frame number and the eye aspect ratio (EAR) of each frame
    blinks : pd.DataFrame
        Contains for each blink the frame number and the time (in seconds)
         start of the blink and end of the blink
    summary : pd.DataFrame
        The number of eye blinks and blink rate (blinks per minute)

    ------------------------------------------------------------------------------------------------------
    """

    ear = pd.DataFrame(columns=[config['frame'], config['ear']])
    blinks = pd.DataFrame(columns=[config['peak_frame'], config['start_frame'], config['end_frame'], config['peak_time'], config['start_time'], config['end_time']])
    summary = pd.DataFrame(columns=[config['blinks'], config['blink_rate']])

    return ear, blinks, summary


def get_config(filepath, json_file):
    """
    ------------------------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes,
    and returns the contents of the file as a dictionary.

    Parameters:
    ...........
    filepath : str
        The path to the configuration file.
    json_file : str
        The name of the configuration file.

    Returns:
    ...........
    measures: A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(filepath)
    measure_path = os.path.abspath(os.path.join(dir_name, f"config/{json_file}"))

    file = open(measure_path)
    measures = json.load(file)
    return measures


def eye_aspect_ratio(eye_landmarks):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the eye aspect ratio (EAR) of a given eye.
    Introduced by Soukupová and Čech in their 2016 paper,
    Real-Time Eye Blink Detection Using Facial Landmarks.

    Parameters:
    ............
    eye_landmarks : array
        Array of 6 tuples containing the coordinates of the eye landmarks

    Returns:
    ............
    ear : float
        The eye aspect ratio (EAR) of the given eye

    ---------------------------------------------------------------------------------------------------
    """

    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

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


def get_video_capture(video):
    """
    ---------------------------------------------------------------------------------------------------

    This function initializes the video capture.

    Parameters:
    ............
    video : string
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
    vs = cv2.VideoCapture(video)
    fps = vs.get(cv2.CAP_PROP_FPS)
    time.sleep(1.0)
    return vs, fps


def process_frame(face_mesh, frame, config):
    """
    ---------------------------------------------------------------------------------------------------

    This function processes a frame with the MediaPipe Face Mesh model.

    Parameters:
    ............
    face_mesh : object
        The MediaPipe Face Mesh model
    frame : array
        The frame to be processed
    config : dict
        The configuration dictionary

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

        # facemesh model left and right eye landmarks indices
        # https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        left_eye_indices = config['left_eye_indices']
        right_eye_indices = config['right_eye_indices']

        leftEye = np.array([(face_landmarks.landmark[lidx].x, face_landmarks.landmark[lidx].y) for lidx in left_eye_indices], dtype=np.float32)
        rightEye = np.array([(face_landmarks.landmark[ridx].x, face_landmarks.landmark[ridx].y) for ridx in right_eye_indices], dtype=np.float32)
        return leftEye, rightEye

    return None, None


def detect_blinks(framewise, prominence, width, config):
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
    config : dict
        The configuration dictionary

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
    troughs, properties = find_peaks(-framewise[config["ear"]], prominence=prominence, width=width)

    left_ips = properties["left_ips"]
    right_ips = properties["right_ips"]

    # round to nearest integer and add 1 to match frame number
    left_ips = np.round(left_ips).astype(int) + 1
    right_ips = np.round(right_ips).astype(int) + 1
    troughs = np.round(troughs).astype(int) + 1

    return troughs, left_ips, right_ips


def convert_frame_to_time(blinks, troughs, left_ips, right_ips, fps, config):
    """
    ---------------------------------------------------------------------------------------------------

    This function converts the frame number to time.

    Parameters:
    ............
    blinks : pd.DataFrame
        Contains for each blink the frame number and the time (in seconds)
            start of the blink and end of the blink
    troughs : array
        Array containing the frame number of each blink
    left_ips : array
        Array containing the frame number of the start of each blink
    right_ips : array
        Array containing the frame number of the end of each blink
    fps : float
        The fps of the video
    config : dict
        The configuration dictionary

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

    blinks[config['peak_frame']] = troughs
    blinks[config['start_frame']] = left_ips
    blinks[config['end_frame']] = right_ips
    blinks[config['peak_time']] = troughs_time
    blinks[config['start_time']] = left_ips_time
    blinks[config['end_time']] = right_ips_time

    return blinks


def calculate_framewise(ear_df, vs, face_mesh, config):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the eye aspect ratio (EAR) of each frame.

    Parameters:
    ............
    ear_df : pd.DataFrame
        Contains the frame number and the eye aspect ratio (EAR) of each frame
    vs : object
        The video capture object
    face_mesh : object
        The MediaPipe Face Mesh model
    config : dict
        The configuration dictionary

    Returns:
    ............
    ear_df : pd.DataFrame
        Contains the frame number and the eye aspect ratio (EAR) of each frame
    frame_n : int
        The total number of frames

    ---------------------------------------------------------------------------------------------------
    """

    framewise = []
    frame_n = 0

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame_n += 1
        ear = np.nan
        try:
            frame = cv2.resize(frame, (450, int(frame.shape[0] * (450. / frame.shape[1]))))

            leftEye, rightEye = process_frame(face_mesh, frame, config)
            if leftEye is None:
                continue

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
        except Exception as e:
            logger.error(e)

        framewise.append([frame_n, ear])

    ear_df[config['frame']] = [x[0] for x in framewise]
    ear_df[config['ear']] = [x[1] for x in framewise]

    # z-score normalization
    ear_df[config['ear']] = (
        ear_df[config['ear']] - ear_df[config['ear']].mean()
    ) / ear_df[config['ear']].std()

    return ear_df, frame_n


def eye_blink_rate(video):
    """
    ---------------------------------------------------------------------------------------------------

    This function counts the number of eye blinks in a given video.

    Parameters:
    ............
    video : string
        The directory of the video to be analyzed

    Returns:
    ............
    framewise: pd.DataFrame
        Contains the frame number and the eye aspect ratio (EAR) of each frame
    blinks : pd.DataFrame
        Contains for each blink the frame number and the time (in seconds)
         start of the blink and end of the blink
    summary : pd.DataFrame
        The number of eye blinks and blink rate (blinks per minute)

    ---------------------------------------------------------------------------------------------------
    """
    
    config = get_config(os.path.abspath(__file__), "eye.json")
    ear, blinks, summary = create_empty_dataframes(config)

    try:
        # validate video directory exists and is a video file
        if not os.path.isfile(video):
            raise ValueError(f"{video} does not exist")
        if not video.endswith(('.mp4', '.avi', '.mov')):
            raise ValueError(f"{video} is not a video file")

        prominence, width = config['prominence'], config['width']

        face_mesh = initialize_facemesh()

        vs, fps = get_video_capture(video)

        # calculate EAR of each frame
        ear, frame_n = calculate_framewise(ear, vs, face_mesh, config)

        # detect blinks from EAR array
        troughs, left_ips, right_ips = detect_blinks(ear, prominence, width, config)

        # convert frame number to time and create blinks dataframe
        blinks = convert_frame_to_time(blinks, troughs, left_ips, right_ips, fps, config)

        # create summary dataframe
        summary_list = [len(troughs), len(troughs)/(frame_n/fps)*60]
        summary.loc[0] = summary_list

    except Exception as e:
        logger.error(f'Error in eye blink rate calculation- file: {video} & Error: {e}')

    finally:
        if 'vs' in locals():
            if vs is not None:
                vs.release()

        if ear.empty:
            ear.loc[0] = np.nan
        if blinks.empty:
            blinks.loc[0] = np.nan
        if summary.empty:
            summary.loc[0] = np.nan

        return ear, blinks, summary
