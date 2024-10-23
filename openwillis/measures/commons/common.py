# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import json
import os
from pydub import AudioSegment
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def make_dir(dir_name):
    """
    ------------------------------------------------------------------------------------------------------

    Creates a directory if it doesn't already exist.

    Parameters:
    ...........
    dir_name : str
        The path to the directory

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def to_audio(filepath, speaker_dict, output_dir, cleaned=False):
    """
    ------------------------------------------------------------------------------------------------------

    Save a numpy signal into multiple speaker-specific audio files.

    Parameters:
    ----------
        filepath : str
            The path to the input audio file.
        speaker_dict : dict
            A dictionary containing speaker labels as keys and corresponding segments (NumPy arrays) as values.
        output_dir : str
            The directory where the output audio files will be saved.
        cleaned : bool
            A flag indicating whether the audio signal has been cleaned.

    ------------------------------------------------------------------------------------------------------
    """
    make_dir(output_dir)
    for key, value in speaker_dict.items():
        file_name, _ = os.path.splitext(os.path.basename(filepath))

        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        spk_signal = AudioSegment(value.tobytes(), frame_rate=44100 if cleaned else audio_signal.frame_rate,
                                  sample_width=audio_signal.sample_width, channels=audio_signal.channels)

        output_file = os.path.join(output_dir, file_name + '_' + key + '.wav')
        spk_signal.export(output_file, format="wav")

def from_audio(audio_dir):
    """
    ------------------------------------------------------------------------------------------------------
    
    Load multiple speaker-specific audio files and reconstruct a dictionary with speaker labels as keys 
    and corresponding segments (NumPy arrays) as values.

    Parameters:
    ----------
        audio_dir : str
            The directory containing the speaker-specific audio files.

    Returns:
    ----------
        speaker_dict : dict
            A dictionary with speaker labels as keys and corresponding audio data (NumPy arrays) as values.

    ------------------------------------------------------------------------------------------------------
    """
    speaker_dict = {}
    
    # Iterate over the files in the directory
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            try:
                # Extract speaker key from the file name (assuming the format 'filename_speaker.wav')
                speaker_key = file_name.split('_')[-1].replace('.wav', '')
                
                # Load the audio file
                audio_file_path = os.path.join(audio_dir, file_name)
                audio = AudioSegment.from_file(audio_file_path, format="wav")
                
                # Convert audio to raw bytes and then to a numpy array
                audio_array = np.array(audio.get_array_of_samples())
                
                # Store in the dictionary with speaker key
                speaker_dict[speaker_key] = audio_array
            except Exception as e:
                logger.info(f"Error while converting from audio to numpy array file: {file_name}, error: {e}")
    
    return speaker_dict

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
