import cv2
import numpy as np
import pandas as pd
import feat  # pip install py-feat

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
        ret, frame = cap.read()
        if not ret:
            break  

        if frame_index % skip_frames == 0:

            faceposes = detector.detect_facepose(frame)

            # (x1, y1, x2, y2, face_confidence)
            faces = faceposes['faces'][0][0]#assume only one face in the frame
            
            # pitch, roll, yaw
            poses = faceposes['poses'][0][0]


            facepose = np.hstack((faces, poses))
            faces_list.append(facepose)

            print(frame_index)
        
        frame_index += 1

    cap.release()

    out_array = np.array(faces_list)
    
    out_df = pd.DataFrame(
        out_array,
        columns=[
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