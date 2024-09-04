# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
from openwillis.measures.audio.util import separation_util as sutil
from pydub import AudioSegment

import os
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

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

def is_whisper_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks if the json response object is from Whisper Transcribe.

    Parameters:
    ...........
    json_conf: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object
     is from Whisper Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    if "segments" in json_conf:
        if len(json_conf["segments"])>0:

            if "words" in json_conf["segments"][0]:
                return True
    return False

def phonation_extraction(filepath, transcript_json):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs phonation extraction from the audio signal based on the transcripted information.

    Parameters:
    ...........
    filepath : str
        Path to the input audio file.
    transcript_json : json
        Speech transcription json response.

    Returns:
    ...........
    phonation_dict : pandas.DataFrame
        A pandas dataframe containing the phonation information.

    ------------------------------------------------------------------------------------------------------
    """
    phonation_dict = {}

    try:
        if not os.path.exists(filepath):
            return phonation_dict

        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        if is_whisper_transcribe(transcript_json):
            speaker_df, _ = sutil.whisperx_to_dataframe(transcript_json)
        elif is_amazon_transcribe(transcript_json):
            speaker_df, _ = sutil.transcribe_response_to_dataframe(transcript_json)
        else:
            speaker_df = sutil.vosk_to_dataframe(transcript_json)

        phonation_df = sutil.extract_phonation(speaker_df)

        if len(phonation_df)>0:
            phonation_dict = sutil.segment_phonations(audio_signal, phonation_df)

    except Exception as e:
        logger.error(f'Error phonation extraction: {e} & File: {filepath}')

    return phonation_dict
