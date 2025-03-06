# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import numpy as np
import pandas as pd
import cv2
import json
import os
import logging
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict

from .util import crop_with_padding_and_center, get_speaking_probabilities, split_speaking_df, get_summary

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

def process_and_format_face_mesh(img, face_mesh, df_common):
    """
    Process the given image using the face_mesh model and format the resulting face landmarks.

    Args:
        img (numpy.ndarray): The input image.
        face_mesh: The face_mesh model.
        df_common (pandas.DataFrame): The common dataframe.

    Returns:
        pandas.DataFrame: The formatted dataframe containing the face landmarks.
    """
    result = face_mesh.process(img)
    df_coord = filter_coord(result)
    df_landmark = pd.concat([df_common, df_coord], axis=1)
    return df_landmark

def crop_and_process_face_mesh(
    img,
    face_mesh,
    df_common,
    bbox,
    frame,
    fps
):
    """
    ---------------------------------------------------------------------------------------------------
    Crop and process the face mesh on the given image.
    .......
    Args:
        img (numpy.ndarray): The input image.
        face_mesh (object): The face mesh object.
        df_common (pd.Dataframe): The common dataframe.
        bbox (dict): The bounding box coordinates of the face.
        frame (int): The frame index
        fps (int): The frames per second of the video
    .......
    Returns:
        pandas.DataFrame: The processed face landmarks dataframe.
    ---------------------------------------------------------------------------------------------------
    """
    if bbox and not np.isnan(bbox['bb_x']):
        cropped_img = crop_with_padding_and_center(img, bbox)
        df_landmark = process_and_format_face_mesh(
            cropped_img,
            face_mesh,
            df_common
        )
    else:
        df_landmark = get_undected_markers(frame,fps)
    return df_landmark


def run_facemesh(path, frames_per_second=3, bbox_list=[]):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file as input, runs Facemesh on the image, and returns a list
    of dataframes containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    bbox_list : list, optional
        List of bounding boxes for each frame in the video

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
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        len_bbox_list = len(bbox_list)
        
        # Calculate skip interval to get desired frames per second
        skip_interval = max(1, int(video_fps / frames_per_second))
        
        face_mesh = init_facemesh()
        bbox_list_passed = len_bbox_list > 0

        if bbox_list_passed & (num_frames != len_bbox_list):
            raise ValueError('Number of frames in video and number of bounding boxes do not match')

        n_frames_skipped = skip_interval

        while True:
            try:
                ret_type, img = cap.read()
                
                if ret_type is not True:
                    break

                if n_frames_skipped < skip_interval:
                    n_frames_skipped += 1
                    df_landmark = get_undected_markers(frame, video_fps)

                elif n_frames_skipped == skip_interval:
                    n_frames_skipped = 0
                    df_common = pd.DataFrame([[frame, frame/video_fps]], columns=['frame', 'time'])
                    
                    if bbox_list_passed:
                        bbox = bbox_list[frame]
                        df_landmark = crop_and_process_face_mesh(
                            img,
                            face_mesh,
                            df_common,
                            bbox,
                            frame,
                            video_fps
                        )
                    else:
                        df_landmark = process_and_format_face_mesh(
                            img,
                            face_mesh,
                            df_common
                        )

            except Exception as e:
                logger.info(f'error processing frame: {frame} in file: {path} & Error: {e}')
                df_landmark = get_undected_markers(frame, video_fps)

            df_list.append(df_landmark)
            frame += 1

    except Exception as e:
        logger.info(f'Face error process file in facemesh for file:{path} & Error: {e}')

    finally:
        # Empty dataframe in case of insufficient datapoints
        if len(df_list) == 0:
            df_landmark = get_empty_dataframe()
            df_list.append(df_landmark)
            logger.info(f'Face not detected by facemesh in: {path}')

    return df_list

def get_undected_markers(frame,fps):
    """
    ---------------------------------------------------------------------------------------------------

    This function creates a dataframe with NaN values representing facial landmarks that were not detected
    in a frame of the video.

    Parameters:
    ............
    frame : int
        Frame number
    fps : int
        Frames per second of the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe with NaN values for undetected facial landmarks in a frame of the video

    ---------------------------------------------------------------------------------------------------
    """
    df_common = pd.DataFrame([[frame, frame/fps]], columns=['frame','time'])
    df_coord = get_column()

    col_list = list(range(0, 468))
    cols_x = ['lmk' + str(s+1).zfill(3) + '_x' for s in col_list]
    cols_y = ['lmk' + str(s+1).zfill(3) + '_y' for s in col_list]
    cols_z = ['lmk' + str(s+1).zfill(3) + '_z' for s in col_list]

    cols = cols_x + cols_y + cols_z
    df_coord.columns = cols

    df_landmark = pd.concat([df_common, df_coord], axis=1)
    return df_landmark

def get_landmarks(path, frames_per_second=3, bbox_list=[]):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file as input, runs Facemesh on the image, and returns a
    dataframe containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    bbox_list : list, optional
        List of bounding boxes for each frame in the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe containing landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    landmark_list = run_facemesh(
        path,
        bbox_list=bbox_list,
        frames_per_second=frames_per_second
    )

    if len(landmark_list) > 0:
        df_landmark = pd.concat(landmark_list).reset_index(drop=True)
    else:
        df_landmark = get_empty_dataframe()

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

def get_mouth_height(df, measures):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a Pandas dataframe of landmark coordinates as input, calculates the Euclidean distance
    between the upper and lower lips, and returns an array of the displacement values.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    mouth_height : numpy.array
        Array of displacement values for mouth height

    ---------------------------------------------------------------------------------------------------
    """

    upper_lip_indices = measures["upper_lip_simple_landmarks"]
    lower_lip_indices = measures["lower_lip_simple_landmarks"]

    upper_lip = ['lmk' + str(col+1).zfill(3) for col in upper_lip_indices]
    lower_lip = ['lmk' + str(col+1).zfill(3) for col in lower_lip_indices]

    mouth_height = 0
    for i in [8, 9, 10]:
        mouth_height += np.sqrt(
            (df[upper_lip[i] + '_x'] - df[lower_lip[18-i] + '_x'])**2
            + (df[upper_lip[i] + '_y'] - df[lower_lip[18-i] + '_y'])**2
        )
    
    return mouth_height

def get_lip_height(df, lip, measures):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a Pandas dataframe of landmark coordinates as input, calculates the Euclidean distance
    between the upper and lower parts of a lip, and returns an array of the displacement values.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates
    lip : str
        lip to calculate height for; must be either 'upper' or 'lower'
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    lip_height : numpy.array
        Array of displacement values for mouth height

    Raises:
    ............
    ValueError
        If lip is not 'upper' or 'lower'

    ---------------------------------------------------------------------------------------------------
    """

    lip = lip.lower()
    if lip not in ['upper', 'lower']:
        raise ValueError('lip must be either upper or lower')

    lip_indices = measures[f"{lip}_lip_simple_landmarks"]

    lip_landmarks = ['lmk' + str(col+1).zfill(3) for col in lip_indices]

    lip_height = 0
    for i in [2, 3, 4]:
        lip_height += np.sqrt(
            (df[lip_landmarks[i] + '_x'] - df[lip_landmarks[12-i] + '_x'])**2
            + (df[lip_landmarks[i] + '_y'] - df[lip_landmarks[12-i] + '_y'])**2
        )
    
    return lip_height

def get_mouth_openness(df, measures):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates whether the mouth openness as the ratio of the mouth height to the min of
     upper lip and lower lip height.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    mouth_openness : numpy.array
        Array of mouth openness values

    ---------------------------------------------------------------------------------------------------
    """

    upper_lip_height = get_lip_height(df, 'upper', measures)
    lower_lip_height = get_lip_height(df, 'lower', measures)
    mouth_height = get_mouth_height(df, measures)

    mouth_openness = mouth_height / np.minimum(upper_lip_height, lower_lip_height)

    return mouth_openness

def baseline(base_path, frames_per_second=3, bbox_list=[], normalize=True, align=False):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to a baseline video file as input, runs Facemesh on the video, and returns a
    dataframe containing the landmark coordinates for each frame of the video. The function can also normalize
    and align the landmarks if specified.

    Parameters:
    ............
    base_path : str
        Path to baseline video file
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    bbox_list : list, optional
        List of bounding boxes for each frame in the video
    normalize : bool, optional
        Whether to normalize the landmarks. Default is True.
    align : bool, optional
        Whether to align the landmarks. Default is False.

    Returns:
    ............
    base_df : pandas.DataFrame
        Normalized dataframe of landmark coordinates

    ---------------------------------------------------------------------------------------------------
    """
    config = get_config(os.path.abspath(__file__), "facial.json")

    base_landmark = get_landmarks(
        base_path, 
        frames_per_second=frames_per_second,
        bbox_list=bbox_list
    )
    
    if normalize:
        base_landmark = normalize_face_landmarks(base_landmark, align=align)
        
    disp_base_df = get_distance(base_landmark)

    disp_base_df['overall'] = pd.DataFrame(disp_base_df.mean(axis=1))
    base_mean = disp_base_df.mean()

    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf])]

    base_df += 1 #Normalization
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
    columns = ['frame','time'] + ['lmk' + str(col+1).zfill(3) for col in range(0, 468)] + ['overall']
    empty_df = pd.DataFrame(columns=columns)
    return empty_df

def get_displacement(
        lmk_df,
        base_path,
        measures,
        base_bbox_list=[],
        normalize=True,
        align=False
    ):
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
    measures : dict
        dictionary of landmark indices
    base_bbox_list : list, optional
        list of bounding boxes for each frame in the baseline video
    normalize : bool, optional
        whether to normalize the facial landmarks to a common reference point (default is True)
    align : bool, optional
        whether to align the facial landmarks based on the position of the eyes (default is False)

    Returns:
    ............
    displacement_df : pandas.DataFrame
         euclidean displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """

    displacement_df = get_empty_dataframe()

    try:
        df_meta = lmk_df[['frame','time']]
#TODO adjust for if there are missing frames 
        if len(lmk_df)>1:
            # here is where we need to adjust for missing frames
            # we need to drop the rows where the landmarks are not detected
            # but we need to maintain columns so we just need to index correclty
            disp_actual_df = get_distance(lmk_df)
            disp_actual_df['overall'] = pd.DataFrame(disp_actual_df.mean(axis=1))

            if os.path.exists(base_path):
                # add normalize flag
                disp_base_df = baseline(
                    base_path,
                    bbox_list=base_bbox_list,
                    normalize=normalize,
                    align=align
                )
                check_na = disp_base_df.iloc[:,1:].isna().all().all()

                if len(disp_base_df)> 0 and not check_na:
                    disp_actual_df = disp_actual_df + 1

                    disp_actual_df = disp_actual_df/disp_base_df.values
                    disp_actual_df = disp_actual_df - 1

            disp_actual_df = calculate_areas_displacement(disp_actual_df, measures)
            displacement_df = pd.concat([df_meta, disp_actual_df], axis=1).reset_index(drop=True)
    except Exception as e:

        logger.info(f'Error in displacement calculation is {e}')
    return displacement_df

def calculate_areas_displacement(displacement_df, measures):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the summary framewise displacement for upper face,
     lower face, lips and eyebros.

    Parameters:
    ............
    displacement_df : pandas.DataFrame
        euclidean displacement dataframe
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    displacement_df : pandas.DataFrame
        updated euclidean displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """

    lower_face_indices = measures["lower_face_landmarks"]
    upper_face_indices = [i for i in range(0, 468) if i not in lower_face_indices]
    lip_indices = measures["lips_landmarks"]
    eyebrow_indices = measures["eyebrows_landmarks"]

    lower_face_cols = ['lmk' + str(col+1).zfill(3) for col in lower_face_indices]
    upper_face_cols = ['lmk' + str(col+1).zfill(3) for col in upper_face_indices]
    lip_cols = ['lmk' + str(col+1).zfill(3) for col in lip_indices]
    eyebrow_cols = ['lmk' + str(col+1).zfill(3) for col in eyebrow_indices]

    displacement_df['lower_face'] = displacement_df[lower_face_cols].mean(axis=1)
    displacement_df['upper_face'] = displacement_df[upper_face_cols].mean(axis=1)
    displacement_df['lips'] = displacement_df[lip_cols].mean(axis=1)
    displacement_df['eyebrows'] = displacement_df[eyebrow_cols].mean(axis=1)

    return displacement_df

def apply_rotation_per_frame(norm_df, rotation_matrices):
    """
    Applies the corresponding rotation matrix to each row (frame) of the DataFrame.

    Parameters:
    norm_df (DataFrame): The DataFrame containing the centered landmarks.
    rotation_matrices (list of np.array): List of 3x3 rotation matrices for each frame.

    Returns:
    DataFrame: The rotated landmarks.
    """
    axes = ['x', 'y', 'z']
    landmark_cols = {
        axis: [col for col in norm_df.columns if col.endswith(f'_{axis}')]
        for axis in axes
    }

    stacked_coords = np.stack([
        norm_df[landmark_cols['x']].values,
        norm_df[landmark_cols['y']].values,
        norm_df[landmark_cols['z']].values
    ], axis=-1)  # Shape: (num_frames, num_landmarks, 3)

    rotated_coords = np.array([
        np.dot(stacked_coords[i], rotation_matrices[i].T) for i in range(len(rotation_matrices))
    ])

    for i, axis in enumerate(axes):
        norm_df[landmark_cols[axis]] = rotated_coords[..., i]

    return norm_df


def calculate_rotation_matrix_for_all_frames(left_eye_x, left_eye_y, right_eye_x, right_eye_y):
    """
    ---------------------------------------------------------------------------------------------------
    Calculates the rotation matrix for each frame based on eye positions to align the eyes horizontally.
    .................................
    Parameters:
    left_eye_x: int
        The x coordinate of the left eye.
    left_eye_y: int
        The y coordinate of the left eye.
    right_eye_x: int
        The x coordinate of the right eye.
    right_eye_y: int
        The y coordinate of the right eye.
    .................................
    Returns:
    list of np.array: List of 3x3 rotation matrices, one for each frame.
    ---------------------------------------------------------------------------------------------------
    """
    rotation_matrices = []
    
    for lx, ly, rx, ry in zip(left_eye_x, left_eye_y, right_eye_x, right_eye_y):
        delta_eye_x = lx - rx
        delta_eye_y = ly - ry

        # Compute 2D angle between eyes (ignore z-axis for horizontal alignment)
        theta = np.arctan2(delta_eye_y, delta_eye_x)

        # Create the rotation matrix to rotate landmarks around the Z-axis
        cos_theta = np.cos(-theta)  
        sin_theta = np.sin(-theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        rotation_matrices.append(rotation_matrix)

    return rotation_matrices

def center_landmarks(df, nose_tip):
    """
    ---------------------------------------------------------------------------------------------------
    Centers the landmarks by moving the nose tip to the origin.

    Parameters:
    df (DataFrame):
        The DataFrame containing the landmarks.
    nose_tip (str): 
        The column name of the nose tip landmark.
    .................................
    Returns:
    DataFrame: The centered landmarks.
    ---------------------------------------------------------------------------------------------------
    """

    axes = ['x', 'y', 'z']
    nose_tip_coords = {axis: f'{nose_tip}_{axis}' for axis in axes}

    norm_data = {
        axis: df.filter(like=f'_{axis}') - df[nose_tip_coords[axis]].values[:, None]
        for axis in axes
    }

    norm_df = pd.concat(norm_data.values(), axis=1)
    return norm_df

def get_vertices_for_col(df, col_name):
    """
    Extracts the x, y, and z coordinates for a given column.

    Parameters:
    df : DataFrame
         The DataFrame containing the columns.
    col_name : str
      The name of the column.

    Returns:
    x_col (Series): The x column.
    y_col (Series): The y column.
    z_col (Series): The z column.
    """
    x_col = df[f'{col_name}_x']
    y_col = df[f'{col_name}_y']
    z_col = df[f'{col_name}_z']

    return x_col, y_col, z_col

# Main function with the refactored parts included
def normalize_face_landmarks(
    df, 
    align=True,
    nose_tip='lmk001',
    left_eye='lmk144',
    right_eye='lmk373'
):
    """
    ---------------------------------------------------------------------------------------------------
    Normalize the face landmarks by centering them around the nose tip and aligning the eyes horizontally.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing the face landmarks.
    align : bool, optional
        Whether to align the landmarks based on the position of the eyes (default is True).
    nose_tip : str, optional
        The name of the nose tip landmark (default is 'lmk001'). Note landmarks are 1-indexed in openwillis and 0-indexed in mediapipe
    left_eye : str, optional
        The name of the left eye landmark (default is 'lmk144').
    right_eye : str, optional
        The name of the right eye landmark (default is 'lmk373').

    Returns:
    --------
    DataFrame: The normalized face landmarks.
    ---------------------------------------------------------------------------------------------------
    """
    left_eye_x, left_eye_y, left_eye_z = get_vertices_for_col(df, left_eye)
    right_eye_x, right_eye_y, right_eye_z = get_vertices_for_col(df, right_eye)

    # compute the eye distance (for scaling)
    eye_distance = np.sqrt(
        (right_eye_x - left_eye_x)**2 +
        (right_eye_y - left_eye_y)**2 +
        (right_eye_z - left_eye_z)**2
    )

    scaling_factor =  eye_distance

    norm_df = center_landmarks(df, nose_tip)

    if align:

        rotation_matrices = calculate_rotation_matrix_for_all_frames(left_eye_x, left_eye_y, right_eye_x, right_eye_y)

        norm_df = apply_rotation_per_frame(norm_df, rotation_matrices)
    
     # scale the landmarks
    landmark_cols = [f'lmk{str(i).zfill(3)}_{axis}' for i in range(1, 469) for axis in ['x', 'y', 'z']]
    norm_df[landmark_cols] = norm_df[landmark_cols].div(scaling_factor.values, axis=0)

    norm_df[['frame','time']] = df[['frame','time']]

    return norm_df

def facial_expressivity(
    filepath,
    baseline_filepath='',
    bbox_list=[],
    base_bbox_list=[],
    frames_per_second=3,
    normalize=True,
    align=False,
    rolling_std_seconds=3,
    split_by_speaking=False,
):
    """
    ---------------------------------------------------------------------------------------------------

    Facial landmark detection and analysis

    This function uses MediaPipe Face Mesh to detect and analyze facial landmarks in a video.
    It can optionally normalize the landmarks using a baseline video and split the analysis
    by speaking probability.

    Parameters
    ..........
    filepath : str
        Path to the input video file
    baseline_filepath : str, optional
        Path to a baseline video for normalization. Default is empty string.
    bbox_list : list of dicts, optional
        List of bounding box dictionaries for each frame in the video.
    base_bbox_list : list of dicts, optional
        List of bounding box dictionaries for each frame in the baseline video.
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    normalize : bool, optional
        Whether to normalize the landmarks. Default is True.
    align : bool, optional
        Whether to align the landmarks. Default is False.
    rolling_std_seconds : int, optional
        Window size in seconds for calculating rolling standard deviation of speaking probability.
        Default is 3.
    split_by_speaking : bool, optional
        Whether to split the analysis by speaking probability. Default is False.

    Returns
    ..........
    df_landmark : pandas.DataFrame
        Dataframe containing landmark coordinates for each frame of the video
    df_summ : pandas.DataFrame
        Summary statistics of the landmark movements

    ---------------------------------------------------------------------------------------------------
    """
    config = get_config(os.path.abspath(__file__), "facial.json")

    try:
        df_landmark = get_landmarks(
            filepath,
            bbox_list=bbox_list,
            frames_per_second=frames_per_second
        )

        if normalize:
            df_landmark = normalize_face_landmarks(df_landmark, align=align)
        
        df_disp = get_displacement(
            df_landmark,
            baseline_filepath,
            config,
            base_bbox_list=base_bbox_list,
            normalize=normalize,
            align=align
        )

        # use mouth height to calculate mouth openness
        df_disp['mouth_openness'] = get_mouth_openness(df_landmark, config)

        if split_by_speaking:
            df_disp['speaking_probability'] = get_speaking_probabilities(
                df_disp, 
                rolling_std_seconds
            )
            df_summ = split_speaking_df(df_disp, 'speaking_probability', 470)

        else:
            df_summ = get_summary(df_disp, 470)

        return df_landmark, df_disp, df_summ

    except Exception as e:
        logger.info(f'Error in facial landmark calculation file: {filepath} & Error: {e}')
