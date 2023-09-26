# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import boto3
import urllib
import time
import random

import json
import logging
from openwillis.measures.audio.util import separation_util as sutil

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def replace_speaker_labels(data, check_labels, speaker_labels):
    """
    ------------------------------------------------------------------------------------------------------

    Recursively replaces speaker labels in a nested dictionary or list.

    Parameters:
    ...........
    data : dict or list
        The nested dictionary or list containing speaker labels.
    check_labels: list
        Check on input speaker labels
    speaker_labels: list
        Expected speaker labels

    Returns:
    ...........
    data : dict or list
        The modified nested dictionary or list with replaced speaker labels.

    ------------------------------------------------------------------------------------------------------
    """

    if isinstance(data, dict):
        for key, value in data.items():

            if key == 'speaker_label':
                if value == check_labels[0]:
                    data[key] = speaker_labels[0]

                elif value == check_labels[1]:
                    data[key] = speaker_labels[1]

                else:
                    data[key] = value
            else:
                replace_speaker_labels(value, check_labels, speaker_labels)
    elif isinstance(data, list):
        for item in data:
            replace_speaker_labels(item, check_labels, speaker_labels)

    return data

def extract_content(data):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts content from a nested dictionary and returns it in a speaker-based dictionary.

    Parameters:
    ...........
    data: json
        Speech transcription response

    Returns:
    ...........
    content_dict : dict
        A dictionary where the keys are speaker labels and the values are the corresponding content spoken by each

    ------------------------------------------------------------------------------------------------------
    """
    content_dict = {}

    if 'results' in data:           # Check if 'results' key is present
        item_data = data['results']['items']

        item_spk_0 = [item for item in item_data if item.get('speaker_label', '') == 'speaker0']
        item_spk_1 = [item for item in item_data if item.get('speaker_label', '') == 'speaker1']

        spk_0_text = [item['alternatives'][0]['content'] for item in item_spk_0 if 'alternatives' in item]
        spk_1_text = [item['alternatives'][0]['content'] for item in item_spk_1 if 'alternatives' in item]

        content_dict['speaker0'] = " ".join(spk_0_text)
        content_dict['speaker1'] = " ".join(spk_1_text)

    return content_dict

def get_clinical_labels(scale, measures, content_dict, json_response):
    """
    ------------------------------------------------------------------------------------------------------

    Replaces speaker labels in a JSON response based on clinical measures.

    Parameters:
    ...........
    scale : str
        Clinical scale
    measures : object
        A configuration object.
    content_dict: dict
        A dictionary containing speaker-based content.
    json_response: json
        Speech transcription response

    Returns:
    ...........
    json_response : json
        The modified JSON response with replaced speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    #Check if content is available for all the speaker
    if content_dict and content_dict['speaker0'] and content_dict['speaker1']:
        if scale.lower() not in measures['scale'].split(","):
            return json_response
        
        score_string = scale.lower()+'_string'
        spk1_score = sutil.match_transcript(measures[score_string], content_dict['speaker0'])
        spk2_score = sutil.match_transcript(measures[score_string], content_dict['speaker1'])

        if spk1_score > spk2_score:
            json_response = replace_speaker_labels(json_response, ['speaker0', 'speaker1'], ['clinician', 'participant'])

        else:
            json_response = replace_speaker_labels(json_response, ['speaker0', 'speaker1'], ['participant', 'clinician'])

    return json_response

def get_job_status(transcribe, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    Get transcriptopn job status

    Parameters:
    ...........
    transcribe : object
        Transcription client object
    input_param: dict
        A dictionary containing input parameters with their corresponding values.

    Returns:
    ...........
    status : str
        Transcription job status

    ------------------------------------------------------------------------------------------------------
    """
    status = json.loads("{}")
    while True:
        try:

            sleep_time = int(random.uniform(5, 10))
            status = transcribe.get_transcription_job(TranscriptionJobName=input_param['job_name'])

            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break

            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Get Transcription job failed with exception: {e}")

    return status

def filter_transcript_response(status, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    Filter transcriptopn job response

    Parameters:
    ...........
    status : object
        Transcription job response
    input_param: dict
        A dictionary containing input parameters with their corresponding values.

    Returns:
    ...........
    tuple : Two strings, the first string is the response object in JSON format, and the second string is
    the transcript of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    read_data = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
    response = json.loads(read_data.read().decode('utf-8'))

    transcript = response['results']['transcripts'][0]['transcript']
    if input_param['ShowSpeakerLabels'] == True:#replace speaker labels with standard names

        response = replace_speaker_labels(response, ['spk_0', 'spk_1'], ['speaker0', 'speaker1'])
    return response, transcript

def transcribe_audio(s3uri, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribe an audio file using AWS Transcribe.

    Parameters:
    ...........
    s3uri : str
        The S3 uri for the recording to be transcribed.
    input_param: dict
        A dictionary containing input parameters with their corresponding values.

    Raises:
    ...........
    Exception: If the transcription job fails.

    Returns:
    ...........
    tuple : Two strings, the first string is the response object in JSON format, and the second string is
    the transcript of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    response = json.loads("{}")
    transcript = ""

    try:
        if input_param['access_key'] and input_param['secret_key']:
            transcribe = boto3.client('transcribe', region_name = input_param['region'], aws_access_key_id = input_param['access_key'], aws_secret_access_key = input_param['secret_key'])
        else:
            transcribe = boto3.client('transcribe', region_name = input_param['region'])

        settings = {'ShowSpeakerLabels': input_param['ShowSpeakerLabels'], 'MaxSpeakerLabels': input_param['MaxSpeakerLabels']}
        transcribe.start_transcription_job(
            TranscriptionJobName=input_param['job_name'],
            Media={'MediaFileUri': s3uri},

            #IdentifyMultipleLanguages=True,
            LanguageCode=input_param['language'],
            Settings=settings
        )

        status = get_job_status(transcribe, input_param)
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            response, transcript = filter_transcript_response(status, input_param)

    except Exception as e:
        logger.error(f"AWS Transcription job failed with file: {s3uri} exception: {e}")

    finally:
        return response, transcript
