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

def is_amazon_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------
    This function checks if the json response object is from Amazon Transcribe.
    Parameters:
    ...........
    json_conf: dict
        JSON response object.
    Returns:
    ...........
    bool: True if the json response object
     is from Amazon Transcribe, False otherwise.
    ------------------------------------------------------------------------------------------------------
    """
    return "jobName" in json_conf and "results" in json_conf

def speaker_separation_labels(filepath, transcript_json):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs speaker separation using cloud based speech diarization techniques.

    Parameters:
    ...........
    filepath : str
        Path to the input audio file.
    transcript_json : json
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
        if not is_amazon_transcribe(transcript_json):

            speaker_df, speaker_count = sutil.whisperx_to_dataframe(transcript_json)
        else:
            speaker_df, speaker_count = sutil.transcribe_response_to_dataframe(transcript_json)
            
        if len(speaker_df)>0 and speaker_count>1:
            signal_label = sutil.generate_audio_signal(speaker_df , audio_signal, '', measures)

    except Exception as e:
        logger.error(f'Error in diard processing: {e} & File: {filepath}')

    return signal_label
