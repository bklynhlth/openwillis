# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import pandas as pd

import cv2

import feat
from feat.utils import FEAT_EMOTION_COLUMNS
from feat.pretrained import  AU_LANDMARK_MAP

import os
import json
import logging

from openwillis.measures.video.util.speaking_utils import get_speaking_probabilities

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

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
        1# this is to formatt bb_list to be compatible with pyfeat (this is face confidence)
        ]]]

def get_faces(detector,frame,bb_dict,threshold=.95):
    '''
    detect faces in frame if bb_dict is empty else use bb_dict

    arguments:
    detector: pyfeat detector object
    frame: frame to detect faces in
    bb_dict: dictionary with bounding box coordinates
    threshold: threshold for face detection

    returns:
    faces: list of faces detected in frame
    '''
    if len(bb_dict.keys()):
        #may need to change this if its batch
        faces = bb_dict_to_bb_list(bb_dict)
    else:
        faces = detector.detect_faces(
                frame,
                threshold=.95,
            )
    return faces

def mouth_openness(
    landmarks,
    upper_lip_lmks=[61, 62, 63],
    lower_lip_lmks=[65, 66, 67]
):
    '''
    insert docstring
    '''
    upper_lip = landmarks[upper_lip_lmks]
    lower_lip = landmarks[lower_lip_lmks]

    lmk_dist = []
    for (upper_lip_x, upper_lip_y), (lower_lip_x, lower_lip_y) in zip(upper_lip, lower_lip):
        lmk_dist.append(np.sqrt((upper_lip_x - lower_lip_x)**2 + (upper_lip_y - lower_lip_y)**2))
    return np.mean(lmk_dist)


def detect_emotions(detector, frame, emo_cols, bb_dict={},threshold=.95):
    # if faces empty else - skip this step
    faces = get_faces(detector,frame,bb_dict,threshold=threshold)
    # ok so landmarks are much less intense than 
    if len(faces[0])<1:
        raise ValueError('No faces detected in frame')
    
    landmarks = detector.detect_landmarks(
        frame,
        detected_faces=faces
    )

    aus = detector.detect_aus(frame, landmarks)

    emotions = detector.detect_emotions(
        frame, faces, landmarks
    )

    emotions = emotions[0][0] * 100 # convert from 0-1 to 0-100
    aus = aus[0][0]
    landmarks = landmarks[0][0]

    
    emos_aus = np.hstack([emotions,aus])
    df_emo = pd.DataFrame([emos_aus],columns=emo_cols)
    df_emo['mouth_openness'] = mouth_openness(landmarks)

    return df_emo

def run_pyfeat(path, skip_frames=5, bbox_list=[]):
    """
    ------------------------------------------------------------------------------------------------------
    This function takes an image path and measures config object as input, and uses the DeepFace package to
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

    
    emo_cols = FEAT_EMOTION_COLUMNS + AU_LANDMARK_MAP['Feat']
    detector = feat.Detector()

    try:

        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        len_bbox_list = len(bbox_list)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        df_list = []
        frame = 0
        n_frames_skipped = skip_frames


        if (len_bbox_list>0) & (num_frames != len_bbox_list):
            raise ValueError('Number of frames in video and number of bounding boxes do not match')
        
        while True:

            try:

                ret_type, img = cap.read()
                if ret_type is not True:
                    break

                if n_frames_skipped < skip_frames:
                    
                    n_frames_skipped += 1
                    df_emotion = get_undected_emotion(frame, emo_cols)

                elif n_frames_skipped == skip_frames:
                    
                    n_frames_skipped = 0
                    print('frame:',frame,'n_frames_skipped:',n_frames_skipped,'skip_frames',skip_frames)
                    
                    bbox = bbox_list[frame] if len_bbox_list>0 else {}
                    df_common = pd.DataFrame([[frame,frame/fps]], columns=['frame','time'])
                    df_emo = detect_emotions(
                        detector,
                        img,
                        emo_cols,
                        bb_dict=bbox
                    ) #pyfeat converts image from bgr - to rgb
                    df_emotion = pd.concat([df_common, df_emo], axis=1)
            
            except Exception as e:
                print(e)
                df_emotion = get_undected_emotion(frame, emo_cols)
            
            df_list.append(df_emotion)
            
            frame +=1

    except Exception as e:
        logger.error(f'Face not detected by deepface for- file:{path} & Error: {e}')

    finally:
        #Empty dataframe in case of insufficient datapoints
        if len(df_list)==0:
            df_emotion = pd.DataFrame(columns = emo_cols)

            df_list.append(df_emotion)
            logger.info(f'Face not detected by deepface in : {path}')

    return df_list

def get_undected_emotion(frame, cols):
    """
    ------------------------------------------------------------------------------------------------------
    This function returns a pandas dataframe with a single row of NaN values for the different facial emotion
    measures. It is used to fill in missing values in cases where DeepFace is unable to detect facial emotion
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
    df_common = pd.DataFrame([[frame]], columns=['frame'])
    value = [np.nan] * len(cols)

    df = pd.DataFrame([value], columns = cols)
    df_emotion = pd.concat([df_common, df], axis=1)
    return df_emotion

def get_emotion(path, measures,skip_frames=5, bbox_list=[],):
    """
    ------------------------------------------------------------------------------------------------------
    This function fetches facial emotion data for each frame of the input video. It calls the run_pyfeat()
    function to get a list of dataframes containing facial emotion data for each frame and then concatenates
    these dataframes into a single dataframe.

    Parameters:
    ..........
    path: str
        The path of the input video file
    error_info: str
        A string that specifies the type of error that occurred (e.g., 'input' or 'baseline')
    measures: dict
        A configuration object containing keys for different facial emotion measures.

    Returns:
    ..........
    df_emo: pandas dataframe
        A dataframe containing facial emotion data for each frame of the input video.
    ------------------------------------------------------------------------------------------------------
    """

    emotion_list = run_pyfeat(path, bbox_list=bbox_list, skip_frames=skip_frames)

    if len(emotion_list)>0:
        df_emo = pd.concat(emotion_list).reset_index(drop=True)

    return df_emo

def baseline(df, base_path, measures, base_bbox_list=[]):
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

    Returns:
    ..........
    df_emotion: pandas dataframe
        The normalized facial emotion data.
    ------------------------------------------------------------------------------------------------------
    """

    df_emo = df.copy()

    if not os.path.exists(base_path):
        return df_emo

    df_common = df_emo[['frame']]
    df_emo.drop(columns=['frame'], inplace=True)

    base_emo = get_emotion(base_path,
                           # 'baseline', look back and see why these matter
                           # measures,
                            bbox_list=base_bbox_list)
    base_mean = base_emo.iloc[:,1:].mean() + 1 #Normalization

    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf]).any(1)]

    if len(base_df)>0:
        df_emo = df_emo + 1 #Normalization
        df_emo = df_emo.div(base_df.iloc[0])

    df_emotion = pd.concat([df_common, df_emo], axis=1)
    return df_emotion

def split_speaking_df(df_disp):
    """
    ---------------------------------------------------------------------------------------------------

    This function splits the displacement dataframe into two dataframes based on speaking probability.

    Parameters:
    ............
    df_disp : pandas.DataFrame
        displacement dataframe

    Returns:
    ............
    df_summ : pandas.DataFrame
        stat summary dataframe
    ---------------------------------------------------------------------------------------------------
    """

    speaking_df = df_disp[df_disp['speaking'] > 0.5]
    not_speaking_df = df_disp[df_disp['speaking'] <= 0.5]
    speaking_df = speaking_df.drop('speaking', axis=1)
    not_speaking_df = not_speaking_df.drop('speaking', axis=1)

    speaking_df_summ = get_summary(speaking_df)
    not_speaking_df_summ = get_summary(not_speaking_df)
    speaking_df_summ = speaking_df_summ.add_suffix('_speaking')
    not_speaking_df_summ = not_speaking_df_summ.add_suffix('_not_speaking')
    
    df_summ = pd.concat([speaking_df_summ, not_speaking_df_summ], axis=1)
    
    return df_summ

def get_summary(df):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the summary statistics for the input dataframe containing the normalized facial
    emotion data. It calculates the mean and standard deviation of each facial emotion measure and returns
    them as a dataframe.

    Parameters:
    ..........
    df: pandas dataframe
        The input dataframe containing the normalized facial emotion data.

    Returns:
    ..........
    df_summ: pandas dataframe
        A dataframe containing the summary statistics for the normalized facial emotion data.
    ------------------------------------------------------------------------------------------------------
    """

    df_summ = pd.DataFrame()
    if len(df)>0:
        
        df_mean = pd.DataFrame(df.mean()).T.iloc[:,1:].add_suffix('_mean')
        df_std = pd.DataFrame(df.std()).T.iloc[:,1:].add_suffix('_std')

        df_summ = pd.concat([df_mean, df_std], axis =1).reset_index(drop=True)
    return df_summ

def emotional_expressivity(
    filepath,
    baseline_filepath='',
    bbox_list=[],
    base_bbox_list=[],
    split_by_speaking=False,
    rolling_std_seconds=3
):
    """
    ------------------------------------------------------------------------------------------------------

    Facial emotion detection and analysis

    This function uses serengil/deepface to quantify the framewise expressivity of facial emotions,
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

    Returns
    ..........
    df_norm_emo : pandas dataframe
        Dataframe with framewise output of facial emotion expressivity. first column is the frame number,
        second column is time in seconds, subsequent columns are emotional expressivity  measures, and
        last column is overall emotional expressivity i.e. a mean of all individual emotions except
        neutral. all values are between 0 and 1, as outputted by deepface.

    df_summ: pandas dataframe
        Dataframe with summary measurements. first column is name of statistic, subsequent columns are
        facial emotions, last column is overall expressivity i.e. the mean of all emotions except neutral.
        first row contains mean expressivity and second row contains standard deviation. In case an optional
        baseline video was provided all measures are relative to baseline values calculated from baseline.

    ------------------------------------------------------------------------------------------------------
    """
    try:

        #Loading json config
        measures = get_config(os.path.abspath(__file__), 'facial.json')
        
        df_emotion = get_emotion(
            filepath,
            measures,
            bbox_list=bbox_list
        )
        
        df_norm_emo = baseline(
            df_emotion, 
            baseline_filepath,
            measures,
            base_bbox_list=base_bbox_list
        )

        comp_exp = df_norm_emo[FEAT_EMOTION_COLUMNS].mean(axis=1)

        df_norm_emo[measures['comp_exp']] = comp_exp

        df_norm_emo['speaking'] = get_speaking_probabilities(
            df_norm_emo,
            rolling_std_seconds
        )

        if split_by_speaking:
            df_summ = split_speaking_df(df_norm_emo)
        else:
            df_summ = get_summary(df_norm_emo)

        if os.path.exists(baseline_filepath):
            df_norm_emo = df_norm_emo - 1
            df_norm_emo['frame'] = df_norm_emo['frame'] + 1

        df_summ = get_summary(df_norm_emo)
        return df_norm_emo, df_summ

    except Exception as e:
        logger.error(f'Error in facial emotion calculation- file: {filepath} & Error: {e}')
