import os
import json
import cv2
import numpy as np

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


def crop_img(img, bb_dict):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes an RGB image and a dictionary containing bounding box coordinates as input. 
    It crops the image based on the provided bounding box coordinates and returns the cropped region.

    Parameters:
    ----------
    img : numpy.ndarray
        The RGB image to be cropped.
    bb_dict : dict
        A dictionary containing the bounding box coordinates.
        The dictionary should have the following keys:
            - 'bb_x': The x-coordinate of the top-left corner of the bounding box.
            - 'bb_y': The y-coordinate of the top-left corner of the bounding box.
            - 'bb_w': The width of the bounding box.
            - 'bb_h': The height of the bounding box.

    Returns:
    -------
    numpy.ndarray
        The cropped region of the image.

    ---------------------------------------------------------------------------------------------------
    """

    x = bb_dict['bb_x']
    y = bb_dict['bb_y']
    w = bb_dict['bb_w']
    h = bb_dict['bb_h']
    roi = img[y:y+h, x:x+w]
    return roi

def draw_bounding_boxes_sf(frame, bb_dict):
    """
    ---------------------------------------------------------------------------------------------------

    Draw bounding boxes on the given frame.

    Parameters:
    ............
        frame (numpy.ndarray): The frame on which to draw the bounding boxes.
        bb_dict (dict): A dictionary containing the bounding box coordinates.

    Returns:
    ............
        numpy.ndarray: The frame with the bounding boxes drawn on it.
    ---------------------------------------------------------------------------------------------------
    
    """
    x = bb_dict['bb_x']
    y = bb_dict['bb_y']
    w = bb_dict['bb_w']
    h = bb_dict['bb_h']
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def blacken_outside_bounding_box(frame, bb_dict):
    """
    ---------------------------------------------------------------------------------------------------

    Blackens the area outside a specified bounding box in the given frame.

    Parameters:
    ............
    frame : numpy.ndarray
        The input image frame in which the area outside the bounding box will be blackened.
    bb_dict : dict
        Dictionary containing bounding box coordinates and dimensions with keys:
        'bb_x' (int): The x-coordinate of the top-left corner of the bounding box.
        'bb_y' (int): The y-coordinate of the top-left corner of the bounding box.
        'bb_w' (int): The width of the bounding box.
        'bb_h' (int): The height of the bounding box.

    Returns:
    ............
    numpy.ndarray
        The resulting image frame with the area outside the bounding box blackened.
    ---------------------------------------------------------------------------------------------------

    """
    x = bb_dict['bb_x']
    y = bb_dict['bb_y']
    w = bb_dict['bb_w']
    h = bb_dict['bb_h']

    # Create a black image of the same size as the frame
    mask = np.zeros_like(frame)

    # Copy the bounding box area from the original frame to the mask
    mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

    return mask


def process_video_single_face(
    video_path,
    output_path,
    detections
):
    """
    ---------------------------------------------------------------------------------------------------

    Process the video by drawing bounding boxes based on the provided detections
    and save the processed video.

    Parameters:
    ............
        video_path (str): Path to the video file.
        detections (list): List of dictionaries containing bounding box and frame index information.
        output_path (str): Path to save the processed video.
        capture_n_per_sec (int, optional): Number of frames to capture per second. Defaults to 30.

    Returns:
    ............
        None    
    ---------------------------------------------------------------------------------------------------

    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    
    max_frame_index = len(detections)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index >= max_frame_index:
            break
            
        frame_dict = detections[frame_index]
        if len(frame_dict.keys()) != 0:
            #draw_bounding_boxes_sf(frame,frame_dict)
            face_frame = blacken_outside_bounding_box(frame,frame_dict)
            out.write(face_frame)
        else:
            out.write(frame)
        frame_index += 1

    cap.release()
    out.release()