import cv2
import numpy as np
import pandas as pd
import feat 
from .util import create_cropped_frame

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def get_undetected_facepose(frame_index):
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

def crop_and_get_facepose(frame_index, frame, detector, bbox):
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
    bbox : dict
        A dictionary containing the bounding box coordinates of the face.                           
        Keys: ['bb_x1', 'bb_y1', 'w', 'h']

    Returns
    -------
    np.ndarray
        A numpy array with the following values:
        [frame_index, bb_x1, bb_y1, bb_x2, bb_y2, face_confidence, pitch, roll, yaw]
    """
    
    frame = create_cropped_frame(frame, bbox)
    facepose = get_facepose(frame_index, frame, detector)
    bbox_face_pose_fmt = [bbox['bb_x'], bbox['bb_y'], bbox['bb_x']+bbox['bb_w'], bbox['bb_y']+bbox['bb_h'],1]
    facepose[1:6]=bbox_face_pose_fmt

    return facepose

def extract_landmarks_and_bboxes(video_path, frames_per_second=3, bbox_list = []):
    """
    Extract bounding boxes and facial landmark coordinates from a video using py-feat's Detector,
    sampling at a specified number of frames per second.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.

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

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    skip_interval = max(1, int(video_fps / frames_per_second))

    frame_index = 0
    faces_list = []
            
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break  

            if frame_index % skip_interval == 0:
                if len(bbox_list)!=0:
                    facepose = crop_and_get_facepose(
                        frame_index,
                        frame,
                        detector,
                        bbox_list[frame_index]
                    )
                else:
                    facepose = get_facepose(frame_index, frame, detector)
            else:
                facepose = get_undetected_facepose(frame_index) 
            
        except Exception as e:
            logger.info(f'error processing frame: {frame_index} in file: {video_path} & Error: {e}')
            facepose = get_undetected_facepose(frame_index) 
        
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

    out_df['time'] = out_df['frame']/video_fps
    
    return out_df

def get_fw_xy_displacement(sampled_frames):
    """Calculates the frame-wise displacement in the x-y plane.

    Args:
        sampled_frames (pd.DataFrame): A DataFrame containing the bounding box center coordinates
            in the 'bb_center_x' and 'bb_center_y' columns for each frame.

    Returns:
        pd.Series: A Series containing the frame-wise displacement magnitudes.
            Returns an empty series if the input DataFrame has fewer than 2 rows.
    """
    # First drop any rows where either x or y coordinate is null
    valid_frames = sampled_frames[['bb_center_x', 'bb_center_y']].dropna()
    
    if len(valid_frames) < 2:
        return pd.Series()
        
    displacement_xy = valid_frames.diff()
    
    displacement_magnitudes = (displacement_xy**2).sum(axis=1)**0.5
    
    return displacement_magnitudes

def compute_rotation_angles_vectorized(pitch: np.ndarray, yaw: np.ndarray, roll: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """
    Computes the total rotation angles for multiple sets of pitch, yaw, and roll angles in degrees,
    with correct matrix multiplication order. Calculates total angle following openface convention:
        https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format#featureextraction

    Args:
    - pitch (np.ndarray): Array of rotations about the X-axis in degrees (pitch).
    - yaw (np.ndarray): Array of rotations about the Y-axis in degrees (yaw).
    - roll (np.ndarray): Array of rotations about the Z-axis in degrees (roll).
    - order (str): Rotation order (e.g., "XYZ" means pitch->yaw->roll).

    Returns:
    - np.ndarray: Array of total rotation angles in degrees.
    """
    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)

    # Compute cosines and sines in batch
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

    # Define batch rotation matrices (shape: (N, 3, 3))
    # Pitch around X-axis
    R_x = np.stack([
        np.stack([np.ones_like(pitch_rad), np.zeros_like(pitch_rad), np.zeros_like(pitch_rad)], axis=-1),
        np.stack([np.zeros_like(pitch_rad), cos_p, -sin_p], axis=-1),
        np.stack([np.zeros_like(pitch_rad), sin_p, cos_p], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Yaw around Y-axis
    R_y = np.stack([
        np.stack([cos_y, np.zeros_like(yaw_rad), sin_y], axis=-1),
        np.stack([np.zeros_like(yaw_rad), np.ones_like(yaw_rad), np.zeros_like(yaw_rad)], axis=-1),
        np.stack([-sin_y, np.zeros_like(yaw_rad), cos_y], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Roll around Z-axis
    R_z = np.stack([
        np.stack([cos_r, -sin_r, np.zeros_like(roll_rad)], axis=-1),
        np.stack([sin_r, cos_r, np.zeros_like(roll_rad)], axis=-1),
        np.stack([np.zeros_like(roll_rad), np.zeros_like(roll_rad), np.ones_like(roll_rad)], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Apply rotations in specified order correctly
    # Convention: R = Rx * Ry * Rz (pitch -> yaw -> roll)
    order_map = {
        "XYZ": lambda: np.einsum("nij,njk,nkl->nil", R_x, R_y, R_z),  # pitch -> yaw -> roll
        "YXZ": lambda: np.einsum("nij,njk,nkl->nil", R_y, R_x, R_z),  # yaw -> pitch -> roll
        "ZXY": lambda: np.einsum("nij,njk,nkl->nil", R_z, R_x, R_y),  # roll -> pitch -> yaw
        "ZYX": lambda: np.einsum("nij,njk,nkl->nil", R_z, R_y, R_x),  # roll -> yaw -> pitch
    }

    if order not in order_map:
        raise ValueError("Invalid rotation order. Use 'XYZ', 'YXZ', 'ZXY', or 'ZYX'.")

    R = order_map[order]()  # Shape: (N, 3, 3)

    # Compute the total rotation angles from the trace of R
    trace_R = np.einsum("nii->n", R)  
    rotation_angles = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))  

    return np.degrees(rotation_angles)  # Convert to degrees

def head_movement(video_path, frames_per_second=3, normalize_by_bb_size=False, bbox_list=[]):
    """
    Extract bounding boxes and facial landmark coordinates from a video using py-feat's Detector,
    sampling at a specified number of frames per second, and compute various head movement metrics.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    normalize_by_bb_size : bool, optional
        If True, normalize the xy displacement by the bounding box width. Default is False.
    bounding_boxes : list of dict, optional
        A list of bounding boxes, each represented as a dictionary with keys:
        ['bb_x1', 'bb_y1', 'w', 'h']. If provided, these bounding boxes will be used 
        instead of detecting them.

    Returns
    -------
    out_df : pd.DataFrame
        DataFrame containing per-frame results including bounding boxes, landmarks,
        displacement, and computed rotation metrics.
    summary_df : pd.DataFrame
        A one-row DataFrame summarizing mean and std of selected metrics.
    """
    
    out_df = extract_landmarks_and_bboxes(
        video_path,
        frames_per_second=frames_per_second,
        bbox_list=bbox_list
    )
    
    out_df = _compute_bb_centers(out_df)

    sampled_frames = _sample_frames(out_df)
    sampled_frames = _compute_xy_disp(
        sampled_frames, 
        normalize_by_bb_size=normalize_by_bb_size
    )

    out_df['euclidean_angle'] = compute_rotation_angles_vectorized(
        out_df['pitch'], 
        out_df['yaw'], 
        out_df['roll']
    )
    
    sampled_angles = out_df['euclidean_angle'].dropna()
    
    out_df['euclidean_angle_disp'] = sampled_angles.diff().abs()

    out_df['xy_disp'] = sampled_frames['xy_disp']

    summary_df = _compute_summary_stats(out_df)

    return out_df, summary_df

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def _compute_bb_centers(df):
    """Compute the center of the bounding box for each frame."""
    df['bb_center_x'] = df[['bb_x1', 'bb_x2']].mean(axis=1)
    df['bb_center_y'] = df[['bb_y1', 'bb_y2']].mean(axis=1)
    return df

def _sample_frames(df):
    """Return all frames since we're already sampling in extract_landmarks_and_bboxes."""
    return df.copy()

def _compute_xy_disp(sampled_df, normalize_by_bb_size=False):
    """Compute the frame-wise XY displacement."""
    sampled_df['xy_disp'] = get_fw_xy_displacement(sampled_df)

    if normalize_by_bb_size:
        sampled_df['xy_disp'] /= (sampled_df['bb_x2'] - sampled_df['bb_x1']).abs()

    return sampled_df

def _compute_euclidean_angles(sampled_df):
    """
    Compute the total Euclidean rotation angle based on pitch, yaw, and roll.
    Also calculate the absolute frame-to-frame change in this angle.
    """
    angles_df = sampled_df[['pitch', 'yaw', 'roll']].dropna().copy()
    
    angles_df['euclidean_angle'] = compute_rotation_angles_vectorized(
        angles_df['pitch'], 
        angles_df['yaw'], 
        angles_df['roll']
    )
    
    # Map back the computed angles to the sampled DataFrame
    sampled_df['euclidean_angle'] = angles_df['euclidean_angle']
    
    sampled_df['euclidean_angle_disp'] = sampled_df['euclidean_angle'].diff().abs()
    
    return sampled_df

def _compute_summary_stats(df):
    """Compute summary statistics (mean and std) for key metrics."""
    stats_cols = ['xy_disp', 'pitch', 'yaw', 'roll', 'euclidean_angle_disp']
    mean_series = df[stats_cols].mean()
    std_series = df[stats_cols].std()

    summary_df = pd.DataFrame({
        'xy_disp_mean': [mean_series['xy_disp']],
        'pitch_mean': [mean_series['pitch']],
        'yaw_mean': [mean_series['yaw']],
        'roll_mean': [mean_series['roll']],
        'euclidean_angle_disp_mean': [mean_series['euclidean_angle_disp']],
        'xy_disp_std': [std_series['xy_disp']],
        'pitch_std': [std_series['pitch']],
        'yaw_std': [std_series['yaw']],
        'roll_std': [std_series['roll']],
        'euclidean_angle_disp_std': [std_series['euclidean_angle_disp']]
    })
    return summary_df
    



