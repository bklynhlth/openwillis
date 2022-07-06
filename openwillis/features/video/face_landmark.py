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

def run_facemesh(path):
    """
    -----------------------------------------------------------------------------------------
    
    Calling Mediapipe Facemesh for facial landmarks
    
    Args:
        path: image path 
        
    Returns:
        results: facial landmark dictionary
        
    -----------------------------------------------------------------------------------------
    """
    
    image = cv2.imread(path)
    face_mesh_mp = mp.solutions.face_mesh
    
    with face_mesh_mp.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results

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
    
    col_list = list(range(0, 478))
    cols = [col_name + '_' + str(s) for s in col_list] 
    
    item = list(map(lambda d: d[col_name], keypoints['landmark']))
    df = pd.DataFrame([item], columns=cols)
    
    return df

def get_column():
    """
    -----------------------------------------------------------------------------------------
    
    Prepare landmark column
    
    Returns:
        col_name: List of landmark column name
    
    -----------------------------------------------------------------------------------------
    """
    
    col_list = list(range(0, 478))
    col_name = []
    lmk_cord = ['x', 'y', 'z']
    
    for col in lmk_cord:
        cols = [col + '_' + str(s) for s in col_list] 
        col_name.extend(cols)
    
    return col_name

def get_landmarks(path):
    """
    -----------------------------------------------------------------------------------------
    Facial landmark 
    
    Args:
        path: image path
    
    Returns:
        df_landmark: facial landmark dataframe
    -----------------------------------------------------------------------------------------
    """
    try:
        
        df_common = pd.DataFrame([[0,1]], columns=['frame', 'timestamp'])
        df_landmark = pd.DataFrame(columns = get_column())
        face_mesh = run_facemesh(path)
        
        for face_landmarks in face_mesh.multi_face_landmarks:
            keypoints = protobuf_to_dict(face_landmarks)
            break

        if len(keypoints)>0:
            df_x = filter_landamrks('x', keypoints)

            df_y = filter_landamrks('y', keypoints)
            df_z = filter_landamrks('z', keypoints)
            df_landmark = pd.concat([df_common, df_x, df_y, df_z], axis=1)
            
    except Exception as e:
        logger.info('Face not detected by mediapipe')
        
    return df_landmark

def faciallandmarks(path,baseline):
    """
    -----------------------------------------------------------------------------------------
    
    Facial landmark detection and facial expressivity analysis

    This function uses mediapipe’s facemesh solution to quantify the framewise
    3D positioning of 478 facial landmarks. It then calculates framewise 
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
    
    df_landmark = get_landmarks(path)
    df_disp = pd.DataFrame()
    df_summ = pd.DataFrame()
    
    return df_landmark, df_disp, df_summ
    
    