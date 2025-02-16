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

    fps = cap.get(cv2.CAP_PROP_FPS)

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

    out_df['time'] = out_df['frame']/fps
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

    displacement_xy = sampled_frames[['bb_center_x','bb_center_y']].diff()
    displacement_xy = displacement_xy.dropna()
    sampled_frames = (displacement_xy**2).sum(axis=1)**0.5

    return sampled_frames

def compute_rotation_angles_vectorized(pitch: np.ndarray, yaw: np.ndarray, roll: np.ndarray, order: str = "ZYX") -> np.ndarray:
    """
    Computes the total rotation angles for multiple sets of pitch, yaw, and roll angles in degrees,
    with correct matrix multiplication order.

    Args:
    - pitch (np.ndarray): Array of rotations about the Y-axis in degrees.
    - yaw (np.ndarray): Array of rotations about the Z-axis in degrees.
    - roll (np.ndarray): Array of rotations about the X-axis in degrees.
    - order (str): Rotation order (e.g., "XYZ" means roll->pitch->yaw).

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
    R_x = np.stack([
        np.stack([np.ones_like(roll_rad), np.zeros_like(roll_rad), np.zeros_like(roll_rad)], axis=-1),
        np.stack([np.zeros_like(roll_rad), cos_r, -sin_r], axis=-1),
        np.stack([np.zeros_like(roll_rad), sin_r, cos_r], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    R_y = np.stack([
        np.stack([cos_p, np.zeros_like(pitch_rad), sin_p], axis=-1),
        np.stack([np.zeros_like(pitch_rad), np.ones_like(pitch_rad), np.zeros_like(pitch_rad)], axis=-1),
        np.stack([-sin_p, np.zeros_like(pitch_rad), cos_p], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    R_z = np.stack([
        np.stack([cos_y, -sin_y, np.zeros_like(yaw_rad)], axis=-1),
        np.stack([sin_y, cos_y, np.zeros_like(yaw_rad)], axis=-1),
        np.stack([np.zeros_like(yaw_rad), np.zeros_like(yaw_rad), np.ones_like(yaw_rad)], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Apply rotations in specified order correctly
    order_map = {
        "XYZ": lambda: np.einsum("nij,njk,nkl->nil", R_x, R_y, R_z),  # Roll -> Pitch -> Yaw
        "YXZ": lambda: np.einsum("nij,njk,nkl->nil", R_y, R_x, R_z),  # Pitch -> Roll -> Yaw
        "ZXY": lambda: np.einsum("nij,njk,nkl->nil", R_z, R_x, R_y),  # Yaw -> Roll -> Pitch
        "ZYX": lambda: np.einsum("nij,njk,nkl->nil", R_z, R_y, R_x),  # Yaw -> Pitch -> Roll
    }

    if order not in order_map:
        raise ValueError("Invalid rotation order. Use 'XYZ', 'YXZ', 'ZXY', or 'ZYX'.")

    R = order_map[order]()  # Shape: (N, 3, 3)

    # Compute the total rotation angles from the trace of R
    trace_R = np.einsum("nii->n", R)  
    rotation_angles = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))  

    return np.degrees(rotation_angles)  # Convert to degrees

def head_movement(video_path, skip_frames=5, normalize_by_bb_size=False):
    """
    Extract bounding boxes and facial landmark coordinates for each frame (or every nth frame)
    of a video using py-feat's Detector, and compute various head movement metrics.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    @TODO make this consistent with other functions and make about samples per second
    skip_frames : int, optional
        Process only every n-th frame; frames in between are skipped. Default is 5.
    normalize_by_bb_size : bool, optional
        If True, normalize the xy displacement by the bounding box width. Default is False.

    Returns
    -------
    out_df : pd.DataFrame
        DataFrame containing per-frame results including bounding boxes, landmarks,
        displacement, and computed rotation metrics.
    summary_df : pd.DataFrame
        A one-row DataFrame summarizing mean and std of selected metrics.
    """
    # Step 1: Extract raw landmarks and bounding boxes
    out_df = extract_landmarks_and_bboxes(video_path, skip_frames=skip_frames)

    out_df = _compute_bb_centers(out_df)

    sampled_frames = _sample_frames(out_df, skip_frames)
    sampled_frames = _compute_xy_disp(
        sampled_frames, 
        normalize_by_bb_size=normalize_by_bb_size
    )

    sampled_frames = _compute_euclidean_angles(sampled_frames)

    out_df[['xy_disp', 'euclidean_angle', 'euclidean_angle_disp']] = sampled_frames[
        ['xy_disp', 'euclidean_angle', 'euclidean_angle_disp']
    ]

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

def _sample_frames(df, skip_frames):
    """Return only every nth frame (based on the 'frame' column)."""
    return df.loc[df['frame'] % skip_frames == 0].copy()

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
    



