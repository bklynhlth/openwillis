# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import pandas as pd

from deepface import DeepFace
import cv2

import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()


def run_deepface(path, measures):
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

    df_list = []
    frame = 0
    cols = [measures['angry'], measures['disgust'], measures['fear'], measures['happy'], measures['sad'],
            measures['surprise'], measures['neutral']]

    try:
        cap = cv2.VideoCapture(path)

        while True:
            try:

                ret_type, img = cap.read()
                if ret_type is not True:
                    break

                df_common = pd.DataFrame([[frame]], columns=['frame'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                face_analysis = DeepFace.analyze(img_path = img_rgb, actions = ['emotion'])
                df_face = pd.DataFrame([face_analysis['emotion'].values()], columns=cols)/100

                frame +=1
                df_emotion = pd.concat([df_common, df_face], axis=1)
                df_list.append(df_emotion)

            except Exception as e:
                df_emotion = get_undected_emotion(frame, cols)
                df_list.append(df_emotion)
                frame +=1

    except Exception as e:
        logger.error(f'Face not detected by deepface for- file:{path} & Error: {e}')

    finally:
        #Empty dataframe in case of insufficient datapoints
        if len(df_list)==0:
            df_emotion = pd.DataFrame(columns = cols)

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

def get_emotion(path, error_info, measures):
    """
    ------------------------------------------------------------------------------------------------------
    This function fetches facial emotion data for each frame of the input video. It calls the run_deepface()
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

    emotion_list = run_deepface(path, measures)

    if len(emotion_list)>0:
        df_emo = pd.concat(emotion_list).reset_index(drop=True)

    return df_emo

def baseline(df, base_path, measures):
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

    base_emo = get_emotion(base_path, 'baseline', measures)
    base_mean = base_emo.iloc[:,1:].mean() + 1 #Normalization

    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf]).any(1)]

    if len(base_df)>0:
        df_emo = df_emo + 1 #Normalization
        df_emo = df_emo.div(base_df.iloc[0])

    df_emotion = pd.concat([df_common, df_emo], axis=1)
    return df_emotion

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

def emotional_expressivity(filepath, baseline_filepath=''):
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
        dir_name = os.path.dirname(os.path.abspath(__file__))
        measure_path = os.path.abspath(os.path.join(dir_name, 'config/facial.json'))

        file = open(measure_path)
        measures = json.load(file)

        df_emotion = get_emotion(filepath, 'input', measures)
        df_norm_emo = baseline(df_emotion, baseline_filepath, measures)

        cols = [measures['angry'], measures['disgust'], measures['fear'], measures['happy'], measures['sad'],
                measures['surprise']]
        comp_exp = df_norm_emo[cols].mean(axis=1)

        df_norm_emo[measures['comp_exp']] = comp_exp

        if os.path.exists(baseline_filepath):
            df_norm_emo = df_norm_emo - 1
            df_norm_emo['frame'] = df_norm_emo['frame'] + 1

        df_summ = get_summary(df_norm_emo)
        return df_norm_emo, df_summ

    except Exception as e:
        logger.error(f'Error in facial emotion calculation- file: {filepath} & Error: {e}')
