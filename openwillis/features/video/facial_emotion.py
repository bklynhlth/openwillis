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
    -----------------------------------------------------------------------------------------
    Calling deepface to fetch facial emotion
    
    Args:
        path: image path
        measures: measures config object
    
    Returns:
        df_list: framewise facial emotion dataframe list
    -----------------------------------------------------------------------------------------
    """
    
    df_list = []
    frame = 0
    
    try:
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret_type, img = cap.read()

            if ret_type is not True:
                break

            df_common = pd.DataFrame([[frame,frame/fps]], columns=['frame', 'timestamp'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_analysis = DeepFace.analyze(img_path = img_rgb, actions = ['emotion'])
            cols = [measures['angry'], measures['disgust'], measures['fear'], measures['happy'], measures['sad'],
                   measures['surprise'], measures['neutral']]
            df_face = pd.DataFrame([face_analysis['emotion'].values()], columns=cols)/100

            frame +=1
            df_emotion = pd.concat([df_common, df_face], axis=1)
            df_list.append(df_emotion)

    except Exception as e:
        logger.info('Face not detected by mediapipe')
        
    return df_list

def get_emotion(path, error_info, measures):
    """
    -----------------------------------------------------------------------------------------
    Fetching facial emotion
    
    Args:
        path: image path
        error_info: error location
        measures: measures config object
    
    Returns:
        df_emo: facial emotion dataframe
    -----------------------------------------------------------------------------------------
    """
    
    emotion_list = run_deepface(path, measures)
    
    if len(emotion_list)>0:
        df_emo = pd.concat(emotion_list).reset_index(drop=True)
        
    else:
        #Handle error in future
        df_emo = pd.DataFrame()
        logger.info('Face not detected by deepface in :'+ error_info)
    
    return df_emo

def baseline(df, base_path, measures):
    """
    -----------------------------------------------------------------------------------------
    Normalize raw data
    
    Args:
        df: facial emotion dataframe
        base_path: baseline input file path
        measures: measures config object
    
    Returns:
        df_emotion: Normalized facial emotion dataframe
    -----------------------------------------------------------------------------------------
    """
    
    df_emo = df.copy()
    
    if len(base_path)==0:
        return df_emo
    
    df_common = df_emo[['frame', 'timestamp']]
    df_emo.drop(columns=['frame', 'timestamp'], inplace=True)
    
    base_emo = get_emotion(base_path, 'baseline', measures)
    base_mean = base_emo.iloc[:,2:].mean() + 1 #update baseline logic (check with Anzar)
    
    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    if len(base_df)>0:
        df_emo = df_emo.div(base_df.iloc[0])
    
    df_emotion = pd.concat([df_common, df_emo], axis=1)
    return df_emotion

def get_summary(df):
    """
    -----------------------------------------------------------------------------------------
    Displacement summary  
    
    Args:
        df: Framewise euclidean displacement dataframe
    
    Returns:
        df_summ: stat summary dataframe
    -----------------------------------------------------------------------------------------
    """
    
    df_summ = pd.DataFrame()
    if len(df)>0:
        
        df_summ = df.iloc[:,2:].describe()
    return df_summ.iloc[1:,:]

def facialemotions(path, base_path):
    """
    -----------------------------------------------------------------------------------------
    
    Facial emotion detection and analysis

    This function uses serengil/deepface to quantify the framewise expressivity of facial emotions, 
    specifically happiness, sadness, anger, fear, disgust, and surprise – in addition to measuring 
    the absence of any emotion (neutral). The summary output provides overall measurements for the input.

    Parameters
        filename : str; path to video
        base_path : str; optional path to baseline video. see openwillis research guidelines on github 
                   wiki to read case for baseline video use, particularly in clinical research contexts. 
                   (default is 0, meaning no baseline correction will be conducted).

    Returns
        framewise : data-type: dataframe with framewise output of facial emotion expressivity. first column 
                    is the frame number, second column is time in seconds, subsequent columns are emotional 
                    expressivity  measures, and last column is overall emotional expressivity i.e. a mean of 
                    all individual emotions except neutral. all values are between 0 and 1, as outputted by 
                    deepface.
                    
        summary: data-type: dataframe with summary measurements. first column is name of statistic, subsequent 
                 columns are facial emotions, last column is overall expressivity i.e. the mean of all emotions 
                 except neutral. first row contains mean expressivity and second row contains standard deviation. 
                 in case an optional baseline video was provided all measures are relative to baseline values 
                 calculated from baseline.

    -----------------------------------------------------------------------------------------
    """
    
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/facial.json'))
    
    file = open(measure_path)
    measures = json.load(file)
    
    df_emotion = get_emotion(path, 'input', measures)
    df_norm_emo = baseline(df_emotion, base_path, measures)
    
    cols = [measures['angry'], measures['disgust'], measures['fear'], measures['happy'], measures['sad'], measures['surprise']]
    comp_exp = df_norm_emo[cols].sum(axis=1)
    df_norm_emo[measures['comp_exp']] = comp_exp
    
    df_summ = get_summary(df_norm_emo)
    
    return df_norm_emo, df_summ