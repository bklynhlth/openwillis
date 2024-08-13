import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

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

def calculate_padding(bb_dict, padding_percent):
    """
    Calculate the padding around the bounding box based on a percentage.

    Parameters:
    ----------
    bb_dict : dict
        A dictionary containing the bounding box coordinates.
        The dictionary should have the following keys:
            - 'bb_x': The x-coordinate of the top-left corner of the bounding box.
            - 'bb_y': The y-coordinate of the top-left corner of the bounding box.
            - 'bb_w': The width of the bounding box.
            - 'bb_h': The height of the bounding box.
    padding_percent : float
        The percentage of padding to add around the bounding box.

    Returns:
    -------
    dict
        A dictionary containing the new bounding box coordinates with padding.
    """
    x = bb_dict['bb_x']
    y = bb_dict['bb_y']
    w = bb_dict['bb_w']
    h = bb_dict['bb_h']
    
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)
    
    return {
        'bb_x': max(0, x - padding_x),
        'bb_y': max(0, y - padding_y),
        'bb_w': w + 2 * padding_x,
        'bb_h': h + 2 * padding_y
    }

def resize_to_fit(img, frame_size):
    """
    Resize the image to fit within the frame size while maintaining the aspect ratio.

    Parameters:
    ----------
    img : numpy.ndarray
        The image to be resized.
    frame_size : tuple
        The size of the frame (width, height).

    Returns:
    -------
    numpy.ndarray
        The resized image that fits within the specified frame size.
    """
    h, w = img.shape[:2]
    frame_w, frame_h = frame_size
    
    aspect_ratio = w / h
    frame_aspect_ratio = frame_w / frame_h
    
    if aspect_ratio > frame_aspect_ratio:
        # Width is the limiting factor
        new_w = frame_w
        new_h = int(new_w / aspect_ratio)
    else:
        # Height is the limiting factor
        new_h = frame_h
        new_w = int(new_h * aspect_ratio)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img

def center_in_frame(cropped_img, frame_size=(512, 512), background_color=(0, 0, 0)):
    """
    Center the cropped and padded image in a specified frame.

    Parameters:
    ----------
    cropped_img : numpy.ndarray
        The cropped image to be centered.
    frame_size : tuple, optional
        The size of the frame (default is 512x512 pixels).
    background_color : tuple, optional
        The background color of the frame (default is black, (0, 0, 0)).

    Returns:
    -------
    numpy.ndarray
        The image centered within the specified frame.
    """
    h_padded, w_padded = cropped_img.shape[:2]
    
    if w_padded > frame_size[0] or h_padded > frame_size[1]:
        cropped_img = resize_to_fit(cropped_img, frame_size)
        h_padded, w_padded = cropped_img.shape[:2]
    
    frame = np.full((frame_size[1], frame_size[0], 3), background_color, dtype=np.uint8)
    
    x_offset = (frame_size[0] - w_padded) // 2
    y_offset = (frame_size[1] - h_padded) // 2
    
    frame[y_offset:y_offset + h_padded, x_offset:x_offset + w_padded] = cropped_img
    
    return frame

def crop_with_padding_and_center(
    img, 
    bb_dict, 
    padding_percent=0.1, 
    frame_size=(512, 512),
    background_color=(0, 0, 0)
):
    """
    Crop an image with padding around the bounding box and center it in a frame.

    Parameters:
    ----------
    img : numpy.ndarray
        The RGB image to be cropped and centered.
    bb_dict : dict
        A dictionary containing the bounding box coordinates.
    padding_percent : float, optional
        The percentage of padding to add around the bounding box (default is 10%).
    frame_size : tuple, optional
        The size of the frame (default is 512x512 pixels).
    background_color : tuple, optional
        The background color of the frame (default is black, (0, 0, 0)).

    Returns:
    -------
    numpy.ndarray
        The padded and centered image within the specified frame.
    """
    padded_bb_dict = calculate_padding(bb_dict, padding_percent)
    padded_crop = crop_img(img, padded_bb_dict)
    final_img = center_in_frame(padded_crop, frame_size, background_color)
    
    return final_img

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

def create_cropped_frame(
    frame,
    frame_dict,
    crop=True,
    default_size=(512,512),
    ):
    """
    Blackens the area outside a specified bounding box in the given frame or crops the frame with padding and centering.

    Parameters:
    ----------
    frame : numpy.ndarray
        The input image frame in which the area outside the bounding box will be blackened or cropped.
    frame_dict : dict
        Dictionary containing bounding box coordinates and dimensions with keys:
        - 'bb_x' (int): The x-coordinate of the top-left corner of the bounding box.
        - 'bb_y' (int): The y-coordinate of the top-left corner of the bounding box.
        - 'bb_w' (int): The width of the bounding box.
        - 'bb_h' (int): The height of the bounding box.
    crop : bool, optional
        If False, blackens the area outside the bounding box. If True, crops the frame with padding and centering. Default is True.
    default_size : tuple, optional
        The size of the cropped frame if `crop` is True. Default is (512, 512).

    Returns:
    -------
    numpy.ndarray
        The resulting image frame with the area outside the bounding box blackened or the cropped frame.

    Notes:
    ------
    - If `crop` is False, the function uses the `blacken_outside_bounding_box` function to blacken the area outside the bounding box.
    - If `crop` is True, the function uses the `crop_with_padding_and_center` function to crop the frame with padding and centering.
    """
    
    if crop:
        face_frame = crop_with_padding_and_center(frame, frame_dict, frame_size=default_size)
    else:
        face_frame = blacken_outside_bounding_box(frame, frame_dict)
        
    return face_frame


def create_face_frame(
        frame_dict,
        frame,
        default_size=(512,512),
        trim=True,
        debug=False,
        crop=True
        ):
    """
    Create a face frame by cropping the input frame based on the provided frame dictionary and booleans for
    style of cropping.

    Parameters:
    ----------
        frame_dict (dict): A dictionary containing the coordinates of the face region in the frame.
        frame (numpy.ndarray): The input frame to be cropped.
        default_size (tuple, optional): The default size of the cropped face frame. Defaults to (512, 512).
        trim (bool, optional): Whether to the trim video so only frames with face present are retained. Defaults to True.
        crop (bool, optional): Whether to crop the cropped face frame. Defaults to True.

    Returns:
        numpy.ndarray: The cropped face frame.
    """
    
    if len(frame_dict.keys()) != 0:

        face_frame = create_cropped_frame(
            frame,
            frame_dict,
            crop=crop,
            default_size=default_size,
        )

    elif debug:

        face_frame = frame

    elif not trim:
        
        if crop:
            face_frame = np.zeros(default_size+(3,),dtype=np.uint8)     
        else:
            face_frame = np.zeros_like(frame,dtype=np.uint8)
            
    else:

        face_frame = np.array([])
    
    return face_frame

def create_cropped_video(
    video_path,
    detections,
    output_path,
    crop=False,
    trim=True,
    debug=False,
    default_size_for_cropped=(512, 512)
):
    """
    Process the video by drawing bounding boxes based on the provided detections
    and save the processed video as an mp4.

    Parameters:
    ----------
    video_path : str
        Path to the video file.
    detections : list
        List of dictionaries containing bounding box and frame index information.
    output_path : str
        Path to save the output video.
    crop : bool, optional
        Flag to crop frame so that only bounding box is retained versus setting all pixels outside bounxing box to black but retaining original framesize (crop == False). by default False.
    trim : bool, optional
        Flag to only return frames with bounding box information, effectively altering the length of the video by. If false black frames are appended when no bounding box info is present default True
    debug : bool, optional
        Flag to keep the original frames when no face is detected, by default False.
    default_size_for_cropped : tuple, optional
        Default size for the cropped frames, by default (512, 512).

    Raises:
    ------
    ValueError
        If there is an error opening the video file.

    """
    cap = cv2.VideoCapture(video_path)

    if debug:
        if crop:
            logger.warning("Debug mode is enabled. crop flag will be set to False.")
            crop = False


    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if crop:
        frame_width, frame_height = default_size_for_cropped

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    max_frame_index = len(detections)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index >= max_frame_index:
            break

        frame_dict = detections[frame_index]

        face_frame = create_face_frame(
            frame_dict,
            frame,
            default_size=default_size_for_cropped,
            trim=trim,
            debug=debug,
            crop=crop
        )

        # Check if face_frame is not empty (i.e., face detected)
        if face_frame.size != 0:
            out.write(face_frame)

        frame_index += 1

    cap.release()
    out.release()