import cv2
import numpy as np
import pandas as pd
import feat 

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def get_undected_facepose(frame_index):
    """
    Return a placeholder face pose when no face is detected in the frame.

    Parameters
    ----------
    frame_index : int
        The frame index in the video.

    Returns
    -------
    np.ndarray
        A numpy array with the following values:
        [frame_index, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    """
    return np.array([frame_index, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])  

def get_facepose(frame_index, frame, detector):
    """
    Extract bounding box and facial landmark coordinates for a frame using py-feat's Detector.

    Parameters
    ----------
    frame_index : int
        The frame index in the video.
    frame : np.ndarray
        The frame image.
    detector : feat.Detector
        An instance of py-feat's Detector class.

    Returns
    -------
    np.ndarray
        A numpy array with the following values:
        [frame_index, bb_x1, bb_y1, bb_x2, bb_y2, face_confidence, pitch, roll, yaw]
    """
    faceposes = detector.detect_facepose(frame)

    # (x1, y1, x2, y2, face_confidence)
    faces = faceposes['faces'][0][0]#assume only one face in the frame
    
    # pitch, roll, yaw
    poses = faceposes['poses'][0][0]

    facepose = np.hstack(([frame_index], faces, poses))

    return facepose

def extract_landmarks_and_bboxes(video_path, skip_frames=5):
    """
    Extract bounding boxes and facial landmark coordinates for each frame (or every nth frame)
    of a video using py-feat's Detector.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    skip_frames : int, optional
        Process only every n-th frame; frames in between are skipped. Default is 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            ['frame', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'face_confidence', 'landmarks']
        - 'frame': The frame index in the video.
        - 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2': The bounding box coordinates of the detected face.
        - 'face_confidence': A float indicating the detection confidence from py-feat.
        - 'landmarks': A list of (x, y) tuples for each facial landmark detected.
    """

    detector = feat.Detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_index = 0

    faces_list = []

   

            
    while True:

        try:
            ret, frame = cap.read()
            if not ret:
                break  

            if frame_index % skip_frames == 0:
                facepose = get_facepose(frame_index, frame, detector)
            else:
                facepose = get_undected_facepose(frame_index) 
            
        except Exception as e:
            logger.info(f'error processing frame: {frame_index} in file: {video_path} & Error: {e}')
            facepose = get_undected_facepose(frame_index) 
        
        frame_index += 1
        faces_list.append(facepose)
 

    cap.release()

    out_array = np.array(faces_list)
    
    out_df = pd.DataFrame(
        out_array,
        columns=[
        'frame',
        'bb_x1',
        'bb_y1',
        'bb_x2',
        'bb_y2',
        'face_confidence',
        'pitch',
        'roll',
        'yaw']
    )

    return out_df

def head_movement(video_path, skip_frames=5):
    """
    Extract bounding boxes and facial landmark coordinates for each frame (or every nth frame)
    of a video using py-feat's Detector.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    skip_frames : int, optional
        Process only every n-th frame; frames in between are skipped. Default is 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            ['frame', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'face_confidence', 'landmarks']
        - 'frame': The frame index in the video.
        - 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2': The bounding box coordinates of the detected face.
        - 'face_confidence': A float indicating the detection confidence from py-feat.
        - 'landmarks': A list of (x, y) tuples for each facial landmark detected.
    """

    out_df = extract_landmarks_and_bboxes(video_path, skip_frames=skip_frames)
    return out_df