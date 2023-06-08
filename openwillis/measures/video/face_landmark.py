# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import pandas as pd
import cv2
import os
import math

import mediapipe as mp
from PIL import Image
from protobuf_to_dict import protobuf_to_dict
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def init_facemesh():
    """
    ---------------------------------------------------------------------------------------------------

    This function initializes a Facemesh object from the Mediapipe library, with a minimum detection
    confidence of 0.5. It returns the Facemesh object.

    Parameters:
    ............
    None

    Returns:
    ............
    face_mesh : Mediapipe object
        Facemesh object with minimum detection confidence of 0.5

    ---------------------------------------------------------------------------------------------------
    """

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    return face_mesh

def filter_landmarks(col_name, keypoints):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes the column name and landmark keypoints detected by Facemesh as inputs, and
    returns a Pandas dataframe with the filtered landmarks in the specified column.

    Parameters:
    ............
    col_name : str
        Column name to filter landmarks into
    keypoints : dict
        Landmark keypoints detected by Facemesh

    Returns:
    ............
    df : pandas.DataFrame
        Dataframe with the filtered landmarks in the specified column

    ---------------------------------------------------------------------------------------------------
    """

    col_list = list(range(0, 468))
    cols = ['lmk' + str(s+1).zfill(3) + '_' + col_name for s in col_list]

    item = list(map(lambda d: d[col_name], keypoints['landmark']))
    df = pd.DataFrame([item], columns=cols)

    return df

def get_column():
    """
    ---------------------------------------------------------------------------------------------------

    This function returns an empty Pandas dataframe with columns corresponding to the 468 facial landmark
    coordinates, labeled with the landmark number and x/y/z coordinate.

    Parameters:
    ............
    None

    Returns:
    ............
    df : pandas.DataFrame
        Empty dataframe with columns for each facial landmark coordinate

    ---------------------------------------------------------------------------------------------------
    """

    col_list = list(range(0, 468))
    col_name = []

    value = [np.nan] * 468 *3
    lmk_cord = ['x', 'y', 'z']

    for col in lmk_cord:
        cols = [col + '_' + str(s+1) for s in col_list]
        col_name.extend(cols)

    df = pd.DataFrame([value], columns = col_name)
    return df

def filter_coord(result):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes the output from a Facemesh object and returns a Pandas dataframe with the filtered
    3D coordinates of each facial landmark detected.

    Parameters:
    ............
    result : Mediapipe object
        Output from a Facemesh object

    Returns:
    ............
    df_coord : pandas.DataFrame
        Dataframe with the filtered 3D coordinates of each facial landmark detected

    ---------------------------------------------------------------------------------------------------
    """

    df_coord = get_column()

    for face_landmarks in result.multi_face_landmarks:
        keypoints = protobuf_to_dict(face_landmarks)

    if len(keypoints)>0:
        df_x = filter_landmarks('x', keypoints)

        df_y = filter_landmarks('y', keypoints)
        df_z = filter_landmarks('z', keypoints)
        df_coord = pd.concat([df_x, df_y, df_z], axis=1)

    return df_coord

def run_facemesh(path):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file as input, runs Facemesh on the image, and returns a list
    of dataframes containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file

    Returns:
    ............
    df_list : list
        List of dataframes containing landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    df_list = []
    frame = 0

    try:

        cap = cv2.VideoCapture(path)
        face_mesh = init_facemesh()

        while True:
            try:

                ret_type, img = cap.read()
                if ret_type is not True:
                    break

                df_common = pd.DataFrame([[frame]], columns=['frame'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                result = face_mesh.process(img_rgb)
                df_coord = filter_coord(result)

                frame +=1
                df_landmark = pd.concat([df_common, df_coord], axis=1)
                df_list.append(df_landmark)

            except Exception as e:
                df_landmark = get_undected_markers(frame)
                df_list.append(df_landmark)
                frame +=1

    except Exception as e:
        logger.error(f'Face not detected by mediapipe file: {path} & Error: {e}')

    return df_list

def get_undected_markers(frame):
    """
    ---------------------------------------------------------------------------------------------------

    This function creates a dataframe with NaN values representing facial landmarks that were not detected
    in a frame of the video.

    Parameters:
    ............
    frame : int
        Frame number

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe with NaN values for undetected facial landmarks in a frame of the video

    ---------------------------------------------------------------------------------------------------
    """
    df_common = pd.DataFrame([[frame]], columns=['frame'])
    df_coord = get_column()

    col_list = list(range(0, 468))
    cols_x = ['lmk' + str(s+1).zfill(3) + '_x' for s in col_list]
    cols_y = ['lmk' + str(s+1).zfill(3) + '_y' for s in col_list]
    cols_z = ['lmk' + str(s+1).zfill(3) + '_z' for s in col_list]

    cols = cols_x + cols_y + cols_z
    df_coord.columns = cols

    df_landmark = pd.concat([df_common, df_coord], axis=1)
    return df_landmark

def get_landmarks(path, error_info):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file and an error location string as input, and returns a Pandas
    dataframe containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    error_info : str
        Error location string

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe containing the landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    landmark_list = run_facemesh(path)

    if len(landmark_list)>0:
        df_landmark = pd.concat(landmark_list).reset_index(drop=True)

    else:
        df_landmark = get_undected_markers(0)
        logger.info(f'Face not detected by mediapipe in file {path}')

    return df_landmark

def get_distance(df):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a Pandas dataframe of landmark coordinates as input, calculates the Euclidean distance
    between each landmark in consecutive frames, and returns a dataframe of the displacement values.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates

    Returns:
    ............
    displacement_df : pandas.DataFrame
        Dataframe containing displacement values for each landmark

    ---------------------------------------------------------------------------------------------------
    """
    disp_list = []

    for col in range(0, 468):
        dist= np.sqrt(np.power(df['lmk' + str(col+1).zfill(3) + '_x'].shift() - df['lmk' + str(col+1).zfill(3) + '_x'], 2) +
                     np.power(df['lmk' + str(col+1).zfill(3) + '_y'].shift() - df['lmk' + str(col+1).zfill(3) + '_y'], 2) +
                     np.power(df['lmk' + str(col+1).zfill(3) + '_z'].shift() - df['lmk' + str(col+1).zfill(3) + '_z'], 2))

        df_dist = pd.DataFrame(dist, columns=['lmk' + str(col+1).zfill(3)])
        disp_list.append(df_dist)

    displacement_df = pd.concat(disp_list, axis=1).reset_index(drop=True)
    return displacement_df

def baseline(base_path):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to a baseline input file and returns a normalized Pandas dataframe of
    landmark coordinates.

    Parameters:
    ............
    base_path : str
        Path to baseline input file

    Returns:
    ............
    base_df : pandas.DataFrame
        Normalized dataframe of landmark coordinates

    ---------------------------------------------------------------------------------------------------
    """

    base_landmark = get_landmarks(base_path, 'baseline')
    disp_base_df = get_distance(base_landmark)

    disp_base_df['overall'] = pd.DataFrame(disp_base_df.mean(axis=1))
    base_mean = disp_base_df.mean()

    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf]).any(1)]

    base_df = base_df + 1 #Normalization
    return base_df

def get_empty_dataframe():
    """
    ---------------------------------------------------------------------------------------------------

    This function creates an empty dataframe containing columns for frame number, landmark position
    variables, and overall displacement measurement.

    Parameters:
    ............
    None

    Returns:
    ............
    empty_df : pandas.DataFrame
        Empty displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """
    columns = ['frame'] + ['lmk' + str(col+1).zfill(3) for col in range(0, 468)] + ['overall']
    empty_df = pd.DataFrame(columns=columns)
    return empty_df

def get_displacement(lmk_df, base_path):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the framewise euclidean displacement of each facial landmark from the landmark
    data and a given baseline. It returns a dataframe containing the framewise displacement data.

    Parameters:
    ............
    lmk_df : pandas.DataFrame
        facial landmark dataframe
    base_path : str
        baseline input file path

    Returns:
    ............
    displacement_df : pandas.DataFrame
         euclidean displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """

    disp_list = []
    displacement_df = get_empty_dataframe()

    try:
        df_meta = lmk_df[['frame']]

        if len(lmk_df)>1:
            disp_actual_df = get_distance(lmk_df)
            disp_actual_df['overall'] = pd.DataFrame(disp_actual_df.mean(axis=1))

            if os.path.exists(base_path):
                disp_base_df = baseline(base_path)
                check_na = disp_base_df.iloc[:,1:].isna().all().all()

                if len(disp_base_df)> 0 and not check_na:
                    disp_actual_df = disp_actual_df + 1

                    disp_actual_df = disp_actual_df/disp_base_df.values
                    disp_actual_df = disp_actual_df - 1

            displacement_df = pd.concat([df_meta, disp_actual_df], axis=1).reset_index(drop=True)
    except Exception as e:

        logger.error(f'Error in displacement calculation is {e}')
    return displacement_df

def get_summary(df):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the summary measurements from the framewise displacement data.

    Parameters:
    ............
    df : pandas.DataFrame
        framewise euclidean displacement dataframe

    Returns:
    ............
    df_summ : pandas.DataFrame
         stat summary dataframe

    ---------------------------------------------------------------------------------------------------
    """

    df_summ = pd.DataFrame()
    if len(df.columns)>0:

        df_summ = df.mean(axis=1)
        df_summ = pd.DataFrame({'mean': df.mean(), 'stdev': df.std()}).T

        df_summ = df_summ.reset_index().rename(columns={'index': 'stats'})
        df_summ = df_summ.drop(columns=['frame'])
    return df_summ

def facial_expressivity(filepath, baseline_filepath=''):
    """
    ---------------------------------------------------------------------------------------------------

    Uses mediapipe's facemesh solution to quantify the framewise 3D positioning of 468 facial landmarks.
    Calculates the framewise displacement of those landmarks to quantify movement in facial musculature
    as a proxy measure of overall facial expressivity.

    Parameters:
        filepath : str
            path to video
        baseline_filepath : str, optional
            optional path to baseline video. see openwillis research guidelines on github wiki to
            read case for baseline video use, particularly in clinical research contexts.
            (default is 0, meaning no baseline correction will be conducted).

    Returns:
        framewise_loc : pandas.DataFrame
            dataframe with framewise output of facial landmark 3D positioning. rows are frames in the input
            video. first column is frame number, second column is time in seconds, and all subsequent columns
            are landmark position variables, with each landmark numbered and further split into its x, y, and
            z coordinates. all coordinate values are between 0 and 1, relative to position in frame, as
            outputted by mediapipe.

        framewise_disp : pandas.DataFrame
            dataframe with framewise euclidean displacement of each facial landmark. rows are frames in input
            video (first row values are always zero). first column is frame number, second column is time in
            seconds, and subsequent columns are framewise displacement values for each facial landmark. last
            column is the overall framewisedisplacement measurement as a mean of all previous displacement columns.

        summary : pandas.DataFrame
            dataframe with summary measurements. first column is name of statistic, subsequent columns are all
            facial landmarks, last column is overall column with composite measures for all landmarks. first
            row contains sum of framewise displacement values, second row contains mean framewise displacement
            over the video, and third row has standard deviation of framewise displacement. in case an optional
            baseline video was provided, all summary measures are relative to baseline values calculated from
            baseline video.

    ---------------------------------------------------------------------------------------------------
    """

    try:
        df_landmark = get_landmarks(filepath, 'input')
        df_disp = get_displacement(df_landmark, baseline_filepath)

        df_summ = get_summary(df_disp)
        return df_landmark, df_disp, df_summ

    except Exception as e:
        logger.error(f'Error in facial landmark calculation file: {filepath} & Error: {e}')
