# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
from openwillis.measures.audio.util import separation_util as sutil
from pydub import AudioSegment

import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def get_config():
    """
    ------------------------------------------------------------------------------------------------------

    This function loads and returns a dictionary containing the configuration settings for speech separation
    from a JSON file.

    Parameters:
    ...........
    None

    Returns:
    ...........
    measures : dict
        A dictionary containing the configuration settings for speech separation.

    ------------------------------------------------------------------------------------------------------
    """
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures

def speaker_separation_cloud(filepath, json_response):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs speaker separation using cloud based speech diarization techniques.

    Parameters:
    ...........
    filepath : str
        Path to the input audio file.
    json_response : json
        Speech transcription json response.

    Returns:
    ...........
    signal_label : pandas.DataFrame
        A pandas dataframe containing the speaker diarization information.

    ------------------------------------------------------------------------------------------------------
    """
    signal_label = {}
    measures = get_config()

    try:
        if not os.path.exists(filepath):
            return signal_label

        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        speaker_df, speaker_count = sutil.transcribe_response_to_dataframe(json_response)

        if len(speaker_df)>0 and speaker_count>1:
            signal_label = sutil.generate_audio_signal(speaker_df , audio_signal, '', measures)

    except Exception as e:
        logger.error(f'Error in diard processing: {e} & File: {filepath}')

    return signal_label
