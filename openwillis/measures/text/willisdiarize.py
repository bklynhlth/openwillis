# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

from openwillis.measures.text.util import diarization_utils as dutil
from openwillis.measures.text.util import huggingface_utils as hutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_config():
    """
    ------------------------------------------------------------------------------------------------------

    Load the configuration settings for the speech transcription.

    Parameters:
    ...........
    None

    Returns:
    ...........
    measures : dict
        A dictionary containing the configuration settings.

    ------------------------------------------------------------------------------------------------------
    """
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures


def read_kwargs(kwargs):
    """
    ------------------------------------------------------------------------------------------------------

    Reads keyword arguments and returns a dictionary containing input parameters.

    Parameters:
    ...........
    kwargs : dict
        Keyword arguments to be processed.

    Returns:
    ...........
    input_param: dict A dictionary containing input parameters with their corresponding values.

    ------------------------------------------------------------------------------------------------------
    """
    input_param = {}
    input_param['huggingface_token'] = kwargs.get('huggingface_token', '')

    return input_param


def is_amazon_transcribe(transcript_json):
    """
    ------------------------------------------------------------------------------------------------------
    This function checks if the json response object is from Amazon Transcribe.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object
     is from Amazon Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    return "jobName" in transcript_json and "results" in transcript_json


def is_whisper_transcribe(transcript_json):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks if the json response object is from Whisper Transcribe.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object
     is from Whisper Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    if "segments" in transcript_json:
        if len(transcript_json["segments"]) > 0:

            if "words" in transcript_json["segments"][0]:
                return True
    return False


def diarization_correction(transcript_json, model_name, **kwargs):
    """
    ------------------------------------------------------------------------------------------------------
    This function corrects the speaker diarization of a transcript
     using a fine-tuned speaker diarization LLM.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.
    model_name: str
        The name of the model used for transcription in HuggingFace.
    kwargs: dict
        Additional arguments for the AWS API call.

    Returns:
    ...........
    dict: JSON response object with corrected speaker diarization.

    ------------------------------------------------------------------------------------------------------
    """
    transcript_json_corrected = transcript_json.copy()
    input_param = read_kwargs(kwargs)
    measures = get_config()

    try:

        if bool(transcript_json):

            if is_whisper_transcribe(transcript_json):
                asr = "whisperx"
            elif is_amazon_transcribe(transcript_json):
                asr = "aws"
            else:
                raise Exception("Transcript source not supported")

            prompts, translate_json = dutil.extract_prompts(transcript_json, asr)
            results = hutil.call_diarization_hf(prompts, model_name, input_param)
            transcript_json_corrected = dutil.correct_transcription(
                transcript_json, prompts, results, translate_json, asr
            )

    except Exception as e:
        logger.error(f"Error in Speaker Diarization Correction {e}")

    return transcript_json_corrected
