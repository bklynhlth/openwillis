# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import logging

from openwillis.measures.text.util import diarization_utils as dutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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


def diarization_correction(transcript_json, **kwargs):
    """
    ------------------------------------------------------------------------------------------------------
    This function corrects the speaker diarization of a transcript
     using a fine-tuned speaker diarization LLM.

    Parameters:
    ...........
    transcript_json: dict
        JSON response object.
    kwargs: dict
        Additional arguments for the AWS API call.

    Returns:
    ...........
    dict: JSON response object with corrected speaker diarization.

    ------------------------------------------------------------------------------------------------------
    """
    transcript_json_corrected = transcript_json.copy()

    try:

        if bool(transcript_json):

            if is_whisper_transcribe(transcript_json):
                asr = "whisperx"
            elif is_amazon_transcribe(transcript_json):
                asr = "aws"
            else:
                raise Exception("Transcript source not supported")

            prompts, translate_json = dutil.extract_prompts(transcript_json, asr)
            results = dutil.call_diarization(prompts, **kwargs)
            transcript_json_corrected = dutil.correct_transcription(
                transcript_json, prompts, results, translate_json, asr
            )

    except Exception as e:
        logger.error(f"Error in Speaker Diarization Correction {e}")

    return transcript_json_corrected
