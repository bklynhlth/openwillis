# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import pandas as pd

import cv2

import feat
from feat.utils import FEAT_EMOTION_COLUMNS
from feat.pretrained import AU_LANDMARK_MAP

import os
import logging

from .util import get_speaking_probabilities, split_speaking_df, get_summary, create_cropped_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def bb_dict_to_bb_list(bb_dict):
    """
    Convert a bounding box dictionary to a bounding box list.
    Args:
        bb_dict (dict): A dictionary containing bounding box coordinates with keys 'bb_x', 'bb_y', 'bb_w', and 'bb_h'.
    Returns:
        list: A nested list representing the bounding box in the format
              [[[bb_x, bb_y, bb_x + bb_w, bb_y + bb_h, 1]]], where 1 is the face confidence.
    """

    return [[[
        bb_dict['bb_x'],
        bb_dict['bb_y'],
        bb_dict['bb_x'] + bb_dict['bb_w'],
        bb_dict['bb_y'] + bb_dict['bb_y'],
        1  # this is to formatt bb_list to be compatible with pyfeat (this is face confidence)
        ]]]

def mouth_openness(
    landmarks,
    upper_lip_lmks=[61, 62, 63],
    lower_lip_lmks=[65, 66, 67]
):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the average distance between the upper and lower lip landmarks to measure the
    openness of the mouth.

    Parameters:
    ..........
    landmarks: numpy array
        An array containing the facial landmarks detected by the facial landmark detector.
    upper_lip_lmks: list
        A list containing the indices of the landmarks that make up the upper lip.
    lower_lip_lmks: list
        A list containing the indices of the landmarks that make up the lower lip.

    Returns:
    ..........
    lmk_dist: float
        The average distance between the upper and lower lip landmarks.
    ------------------------------------------------------------------------------------------------------
    """

    upper_lip = landmarks[upper_lip_lmks]
    lower_lip = landmarks[lower_lip_lmks]

    lmk_dist = []
    for (upper_lip_x, upper_lip_y), (lower_lip_x, lower_lip_y) in zip(upper_lip, lower_lip):
        lmk_dist.append(np.sqrt((upper_lip_x - lower_lip_x)**2 + (upper_lip_y - lower_lip_y)**2))
    return np.mean(lmk_dist)


def detect_emotions(detector, frame, emo_cols, threshold=.95):
    """
    ------------------------------------------------------------------------------------------------------
    This function takes a frame and a configuration object as input, and uses the py-feat package to fetch
    facial emotion data for the frame. It returns a pandas dataframe containing the facial emotion data.

    Parameters:
    ..........
    detector: pyfeat detector object
        The pyfeat detector object used to detect facial emotions.
    frame: numpy array
        The frame for which facial emotion data is being fetched.
    emo_cols: list
        A list of column names for the facial emotion measures.
    threshold: float
        The threshold for face detection.

    Returns:
    ..........
    df_emo: pandas dataframe
        A dataframe containing facial emotion data for the input frame.
    ------------------------------------------------------------------------------------------------------
    """

    faces = detector.detect_faces(
        frame,
        threshold=threshold,
    )

    landmarks = detector.detect_landmarks(
        frame,
        detected_faces=faces
    )

    aus = detector.detect_aus(frame, landmarks)

    emotions = detector.detect_emotions(
        frame,
        faces,
        landmarks
    )

    emotions = emotions[0][0] * 100  # convert from 0-1 to 0-100
    aus = aus[0][0]
    landmarks = landmarks[0][0]

    emos_aus = np.hstack([emotions, aus])
    df_emo = pd.DataFrame([emos_aus], columns=emo_cols)
    df_emo['mouth_openness'] = mouth_openness(landmarks)

    return df_emo

def crop_and_detect_emotions(
    img,
    bbox,
    detector,
    emo_cols,
    fps,
    frame,
    df_common
):
    """
    ------------------------------------------------------------------------------------------------------
    This function crops the input frame using the bounding box information and then uses the pyfeat package to
    fetch facial emotion data for the cropped frame. It returns a pandas dataframe containing the facial emotion
    data.

    Parameters:
    ..........
    img: numpy array
        The input frame for which facial emotion data is being fetched.
    bbox: dict
        A dictionary containing the bounding box information for the face in the frame.
    detector: pyfeat detector object
        The pyfeat detector object used to detect facial emotions.
    emo_cols: list
        A list of column names for the facial emotion measures.
    fps: float
        The frame rate of the video.
    frame: int
        The frame number for which facial emotion data is being fetched.
    df_common: pandas dataframe
        A dataframe containing the frame number and time for the input frame.

    Returns:
    ..........
    df_emo: pandas dataframe
        A dataframe containing facial emotion data for the input frame.
    ------------------------------------------------------------------------------------------------------
    """

    if not np.isnan(bbox['bb_x']):

        cropped_img = create_cropped_frame(
            img,
            bbox
        )

        df_emo = detect_emotions(
            detector,
            cropped_img,
            emo_cols
        )

        df_emo = pd.concat([df_common, df_emo], axis=1)

    else:

        df_emo = get_undected_emotion(frame, emo_cols, fps)

    return df_emo

def run_pyfeat(path, skip_frames=5, bbox_list=[]):
    """
    ------------------------------------------------------------------------------------------------------
    This function takes an image path and measures config object as input, and uses the py-feat package to
    fetch facial emotion for each frame of the video. It returns a list of dataframes, each containing facial
    emotion data for one frame.

    Parameters:
    ..........
    path: str
        The path of the image file
    measures: dict
        A configuration object containing keys for different facial emotion measures

    Returns:
    ..........
    df_list: list
        A list of pandas dataframes where each dataframe contains facial emotion data for a single frame of
        the video.
    ------------------------------------------------------------------------------------------------------
    """

    try:
        # init pyfeat
        emo_cols = FEAT_EMOTION_COLUMNS + AU_LANDMARK_MAP['Feat']
        detector = feat.Detector()

        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        len_bbox_list = len(bbox_list)
        fps = cap.get(cv2.CAP_PROP_FPS)

        df_list = []
        frame = 0
        n_frames_skipped = skip_frames

        bbox_list_passed = len_bbox_list > 0

        if bbox_list_passed & (num_frames != len_bbox_list):
            raise ValueError('Number of frames in video and number of bounding boxes do not match')

        while True:

            try:

                ret_type, img = cap.read()
                if ret_type is not True:
                    break

                if n_frames_skipped < skip_frames:

                    n_frames_skipped += 1
                    df_emotion = get_undected_emotion(frame, emo_cols, fps)

                elif n_frames_skipped == skip_frames:
                    n_frames_skipped = 0
                    df_common = pd.DataFrame([[frame, frame/fps]], columns=['frame', 'time'])
                    # if there is a bounding box list crop the frame (or return a black frame)

                    if bbox_list_passed:

                        bbox = bbox_list[frame]
                        df_emotion = crop_and_detect_emotions(
                            img,
                            bbox,
                            detector,
                            emo_cols,
                            fps,
                            frame,
                            df_common
                        )

                    else:

                        df_emo = detect_emotions(
                            detector,
                            img,
                            emo_cols
                        )

                        df_emotion = pd.concat([df_common, df_emo], axis=1)

            except Exception as e:
                logger.info(f'error processing frame: {frame} in file: {path} & Error: {e}')
                df_emotion = get_undected_emotion(frame, emo_cols, fps)

            df_list.append(df_emotion)

            frame += 1

    except Exception as e:
        logger.info(f'Face error process file in pyfeat for file:{path} & Error: {e}')

    finally:
        # Empty dataframe in case of insufficient datapoints
        if len(df_list) == 0:
            df_emotion = pd.DataFrame(columns=emo_cols)

            df_list.append(df_emotion)
            logger.info(f'Face not detected by pyfeat in: {path}')

    return df_list

def get_undected_emotion(frame, cols, fps):
    """
    ------------------------------------------------------------------------------------------------------
    This function returns a pandas dataframe with a single row of NaN values for the different facial emotion
    measures. It is used to fill in missing values in cases where py-feat is unable to detect facial emotion
    for a particular frame.

    Parameters:
    ..........
    frame: int
        The frame number for which the dataframe is being created
    cols: list
        A list of column names for the facial emotion measures

    Returns:
    ..........
    df_emotion: pandas dataframe
        A dataframe with a single row of NaN values for the different facial emotion measures.
    ------------------------------------------------------------------------------------------------------
    """
    df_common = pd.DataFrame([[frame, frame/fps]], columns=['frame', 'time'])
    value = [np.nan] * len(cols)

    df = pd.DataFrame([value], columns=cols)
    df_emotion = pd.concat([df_common, df], axis=1)
    return df_emotion

def get_emotion(path, skip_frames=5, bbox_list=[]):
    """
    ------------------------------------------------------------------------------------------------------
    This function fetches facial emotion data for each frame of the input video. It calls the run_pyfeat()
    function to get a list of dataframes containing facial emotion data for each frame and then concatenates
    these dataframes into a single dataframe.

    Parameters:
    ..........
    path: str
        The path of the input video file
    skip_frames: int
        number of frames to skip between samples
    bbox_list: list of dicts
        a list of dicts as long as the video that contain bounding box information

    Returns:
    ..........
    df_emo: pandas dataframe
        A dataframe containing facial emotion data for each frame of the input video.
    ------------------------------------------------------------------------------------------------------
    """

    emotion_list = run_pyfeat(
        path,
        bbox_list=bbox_list,
        skip_frames=skip_frames
    )

    if len(emotion_list) > 0:
        df_emo = pd.concat(emotion_list).reset_index(drop=True)
    else:
        df_emo = pd.DataFrame()

    return df_emo

def baseline(
    df,
    base_path,
    base_bbox_list=[],
    skip_frames=5
):
    """
    ------------------------------------------------------------------------------------------------------
    This function normalizes the facial emotion data in the input dataframe using the baseline video. If no
    baseline video is provided, the function simply returns the input dataframe. If a baseline video is
    provided, the function first calculates the mean of the facial emotion measures for the baseline video and
    then normalizes the facial emotion measures in the input dataframe by dividing them by the baseline mean.

    Parameters:
    ..........
    df: pandas dataframe
        The input dataframe containing facial emotion data
    base_path: str
        The path to the baseline video
    measures: dict
        A configuration object containing keys for different facial emotion measures.
    skip_frames: int
        number of frames to skip between frames processed

    Returns:
    ..........
    df_emotion: pandas dataframe
        The normalized facial emotion data.
    ------------------------------------------------------------------------------------------------------
    """

    df_emo = df.copy()

    if not os.path.exists(base_path):
        return df_emo

    df_common = df_emo[['frame', 'time', 'mouth_openness']]
    df_emo.drop(columns=['frame', 'time', 'mouth_openness'], inplace=True)

    base_emo = get_emotion(
        base_path,
        bbox_list=base_bbox_list,
        skip_frames=skip_frames
    )

    base_mean = base_emo.drop(
        columns=['frame', 'time', 'mouth_openness']).mean() + 1  # Normalization

    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf])]

    if len(base_df) > 0:
        df_emo += 1  # Normalization
        df_emo = df_emo.div(base_df.iloc[0])

    df_emotion = pd.concat([df_common, df_emo], axis=1)

    return df_emotion

def emotional_expressivity(
    filepath,
    baseline_filepath='',
    bbox_list=[],
    base_bbox_list=[],
    skip_frames=5,
    split_by_speaking=False,
    rolling_std_seconds=3
):
    """
    ------------------------------------------------------------------------------------------------------

    Facial emotion detection and analysis

    This function uses pyfeat to quantify the framewise expressivity of facial emotions,
    specifically happiness, sadness, anger, fear, disgust, and surprise – in addition to measuring
    the absence of any emotion (neutral). The summary output provides overall measurements for the input.

    Parameters
    ..........
    filepath : str
        path to video
    baseline_filepath : str, optional
        optional path to baseline video. see openwillis research guidelines on github wiki to read case
        for baseline video use, particularly in clinical research contexts. (default is 0, meaning no
        baseline correction will be conducted).
    bbox_list : list of dicts, optional
        list of bounding box dictionaries for each frame in the video. (default is 0, meaning pyfeat will
        be used to detect faces in each frame).
    base_bbox_list : list of dicts, optional
        list of bounding box dictionaries for each frame in the baseline video. (default is 0, meaning
        pyfeat will be used to detect faces in each frame).
    skip_frames : int, optional
        number of frames to skip between samples (default is 5).
    split_by_speaking : bool, optional
        split the output by speaking probability (default is False).
    rolling_std_seconds : int, optional
        window size in seconds for calculating rolling standard deviation of speaking probability
        (default is 3).

    Returns
    ..........
    df_norm_emo : pandas dataframe
        Dataframe with framewise output of facial emotion expressivity. first column is the frame number,
        second column is time in seconds, subsequent columns are emotional expressivity  measures,

    df_summ: pandas dataframe
        Dataframe with summary measurements. first column is name of statistic, subsequent columns are
        facial emotions, last column is overall expressivity i.e. the mean of all emotions except neutral.
        first row contains mean expressivity and second row contains standard deviation. In case an optional
        baseline video was provided all measures are relative to baseline values calculated from baseline.

    ------------------------------------------------------------------------------------------------------
    """
    try:

        df_emotion = get_emotion(
            filepath,
            bbox_list=bbox_list,
            skip_frames=skip_frames
        )

        df_norm_emo = baseline(
            df_emotion,
            baseline_filepath,
            base_bbox_list=base_bbox_list,
            skip_frames=skip_frames
        )

        if split_by_speaking:
            df_norm_emo['speaking_probability'] = get_speaking_probabilities(
                df_norm_emo,
                rolling_std_seconds
            )
            df_summ = split_speaking_df(df_norm_emo, 'speaking_probability', 2)
        else:
            df_summ = get_summary(df_norm_emo, 2)

        if os.path.exists(baseline_filepath):
            df_norm_emo -= 1
            df_norm_emo[['frame', 'time']] += 1

            if split_by_speaking:
                df_norm_emo['speaking_probability'] += 1

        df_norm_emo.dropna(inplace=True)
        df_norm_emo.reset_index(drop=True, inplace=True)

        return df_norm_emo, df_summ

    except Exception as e:
        logger.info(f'Error in facial emotion calculation- file: {filepath} & Error: {e}')
