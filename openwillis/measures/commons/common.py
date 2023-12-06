# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import json
import os
from pydub import AudioSegment

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

def to_audio(filepath, speaker_dict, output_dir):
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

    ------------------------------------------------------------------------------------------------------
    """
    make_dir(output_dir)
    for key, value in speaker_dict.items():
        file_name, _ = os.path.splitext(os.path.basename(filepath))

        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        spk_signal = AudioSegment(value.tobytes(), frame_rate=audio_signal.frame_rate,
                                  sample_width=audio_signal.sample_width, channels=audio_signal.channels)

        output_file = os.path.join(output_dir, file_name + '_' + key + '.wav')
        spk_signal.export(output_file, format="wav")

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
