# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import pandas as pd
import cv2
import math

import mediapipe as mp
from PIL import Image
from protobuf_to_dict import protobuf_to_dict
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def init_facemesh():
    """
    -----------------------------------------------------------------------------------------
    
    Mediapipe Facemesh onject
    
    Args:
        path: image path 
        
    Returns:
        results: facemesh object
        
    -----------------------------------------------------------------------------------------
    """
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    return face_mesh

def filter_landamrks(col_name, keypoints):
    """
    -----------------------------------------------------------------------------------------
    
    Creating landmark dataframe
    
    Args:
        col_name: column name
        keypoints: facemesh landmark points
        
    Returns: 
        df: facial landmark dataframe
        
    -----------------------------------------------------------------------------------------
    """
    
    col_list = list(range(0, 468))
    cols = [col_name + '_' + str(s) for s in col_list] 
    
    item = list(map(lambda d: d[col_name], keypoints['landmark']))
    df = pd.DataFrame([item], columns=cols)
    
    return df

def get_column():
    """
    -----------------------------------------------------------------------------------------
    
    Prepare landmark column
    
    Returns:
        col_name: empty dataframe with facial landmark columns
    
    -----------------------------------------------------------------------------------------
    """
    
    col_list = list(range(0, 468))
    col_name = []
    
    value = [np.nan] * 468 *3
    lmk_cord = ['x', 'y', 'z']
    
    for col in lmk_cord:
        cols = [col + '_' + str(s) for s in col_list] 
        col_name.extend(cols)
    
    df = pd.DataFrame([value], columns = col_name)
    return df

def filter_coord(result):
    """
    -----------------------------------------------------------------------------------------
    
    Filtering facial landmarks from facemesh output
    
    Args:
        result: Facemesh output
        
    Returns:
        df_coord: Facial landmaark dataframe
    
    -----------------------------------------------------------------------------------------
    """
    
    df_coord = get_column()
    
    for face_landmarks in result.multi_face_landmarks:
        keypoints = protobuf_to_dict(face_landmarks)

    if len(keypoints)>0:
        df_x = filter_landamrks('x', keypoints)

        df_y = filter_landamrks('y', keypoints)
        df_z = filter_landamrks('z', keypoints)
        df_coord = pd.concat([df_x, df_y, df_z], axis=1)

    return df_coord

def run_facemesh(path):
    """
    -----------------------------------------------------------------------------------------
    Calling facemesh to fetch facial landmarks
    
    Args:
        path: image path
    
    Returns:
        df_list: framewise facial landmark dataframe list
    -----------------------------------------------------------------------------------------
    """
    
    df_list = []
    frame = 0
    
    try:
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        face_mesh = init_facemesh()

        while True:
            ret_type, img = cap.read()

            if ret_type is not True:
                break

            df_common = pd.DataFrame([[frame,frame/fps]], columns=['frame', 'timestamp'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(img_rgb)
            df_coord = filter_coord(result)

            frame +=1
            df_landmark = pd.concat([df_common, df_coord], axis=1)
            df_list.append(df_landmark)

    except Exception as e:
        logger.info('Face not detected by mediapipe')
        
    return df_list

def get_landmarks(path, error_info):
    """
    -----------------------------------------------------------------------------------------
    Facial landmark 
    
    Args:
        path: image path
        error_info: error location
    
    Returns:
        df_landmark: facial landmark dataframe
    -----------------------------------------------------------------------------------------
    """
    
    landmark_list = run_facemesh(path)
    
    if len(landmark_list)>0:
        df_landmark = pd.concat(landmark_list).reset_index(drop=True)
        
    else:
        #Handle error in future
        df_landmark = pd.DataFrame()
        logger.info('Face not detected by mediapipe in :'+ error_info)
    
    return df_landmark

def baseline(df, base_path):
    """
    -----------------------------------------------------------------------------------------
    Normalize raw data
    
    Args:
        df: facial landmark dataframe
        base_path: baseline input file path
    
    Returns:
        df_landmark: Normalized facial landmark dataframe
    -----------------------------------------------------------------------------------------
    """
    
    df_landmark = df.copy()
    
    if len(base_path)==0:
        return df_landmark
    
    base_landmark = get_landmarks(base_path, 'baseline')
    base_mean = base_landmark.iloc[:,2:].mean()
    
    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    if len(base_df)>0:
        df_landmark = df_landmark.div(base_df.iloc[0])
    
    return df_landmark

def get_displacement(lmk_df, base_path):
    """
    -----------------------------------------------------------------------------------------
    Framewise euclidean displacement
    
    Args:
        lmk_df: facial landmark dataframe
        base_path: baseline input file path
    
    Returns:
        displacement_df: euclidean displacement dataframe
    -----------------------------------------------------------------------------------------
    """
    
    disp_list = []
    displacement_df = pd.DataFrame()
    
    try:
    
        df = baseline(lmk_df, base_path)
        df_meta = lmk_df[['frame','timestamp']]

        if len(df)>1:
            for col in range(0, 468):

                dist= np.sqrt(np.power(df['x_' + str(col)].shift() - df['x_' + str(col)], 2) + 
                             np.power(df['y_' + str(col)].shift() - df['y_' + str(col)], 2) + 
                             np.power(df['z_' + str(col)].shift() - df['z_' + str(col)], 2))

                df_dist = pd.DataFrame(dist, columns=['disp_' + str(col)])
                disp_list.append(df_dist)

            displacement_df = pd.concat([df_meta] + disp_list, axis=1).reset_index(drop=True)

    except Exception as e:
        logger.info('Error in displacement calculation')
    return displacement_df

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
        
        df_summ = df.describe()
    return df_summ

def faciallandmarks(path,baseline):
    """
    -----------------------------------------------------------------------------------------
    
    Facial landmark detection and facial expressivity analysis

    This function uses mediapipe???s facemesh solution to quantify the framewise
    3D positioning of 468 facial landmarks. It then calculates framewise 
    displacement of those landmarks to quantify movement in facial musculature 
    as a proxy measure of overall facial expressivity.

    Parameters:
        filename : str; path to video ;
        baseline : str, optional path to baseline video. see openwillis research 
                   guidelines on github wiki to read case for baseline video use, 
                   particularly in clinical research contexts. (default is 0, meaning
                   no baseline correction will be conducted).

    Returns:
        framewise_loc : data-type
                dataframe with framewise output of facial landmark 3D positioning.
                rows are frames in the input video. first column is frame number, 
                second column is time in seconds, and all subsequent columns are 
                landmark position variables, with each landmark numbered and further 
                split into its x, y, and z coordinates. all coordinate values are 
                between 0 and 1, relative to position in frame, as outputted by 
                mediapipe.

        framewise_disp : data-type
                dataframe with framewise euclidean displacement of each facial 
                landmark. rows are frames in input video (first row values are always 
                zero). first column is frame number, second column is time in 
                seconds, and subsequent columns are framewise displacement values for 
                each facial landmark. last column is the overall framewise 
                displacement measurement as a mean of all previous displacement columns.

        summary : data-type
                dataframe with summary measurements. first column is name of 
                statistic, subsequent columns are all facial landmarks, last column 
                is overall column with composite measures for all landmarks. first 
                row contains sum of framewise displacement values, second row 
                contains mean framewise displacement over the video, and third row 
                has standard deviation of framewise displacement. in case an optional 
                baseline video was provided, all summary measures are relative to
                baseline values calculated from baseline video.
    
    -----------------------------------------------------------------------------------------
    """
    
    
    df_landmark = get_landmarks(path, 'input')
    df_disp = get_displacement(df_landmark, baseline)
    df_summ = get_summary(df_disp)
    
    return df_landmark, df_disp, df_summ
    
    