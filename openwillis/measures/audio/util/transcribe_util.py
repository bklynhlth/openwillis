# author:    Vijay Yadav
# website:   http://www.brooklyn.health

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

def filter_labels_aws(data):
    """
    ------------------------------------------------------------------------------------------------------

    replaces speaker labels in AWS JSON.

    Parameters:
    ...........
    data : JSON
        The JSON containing speaker labels.

    Returns:
    ...........
    data : JSON
        The modified JSON with replaced speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    if 'results' in data:
        speaker_labels = data['results'].get('speaker_labels', {})
        segments = speaker_labels.get('segments', [])

        for segment in segments:
            seg_speaker_label = segment.get('speaker_label', '')
            
            if 'spk_' in seg_speaker_label:
                segment['speaker_label'] = seg_speaker_label.replace("spk_", "speaker")
            
            seg_items = segment.get('items', [])
            for seg_item in seg_items:
                
                seg_item_speaker_label = seg_item.get('speaker_label', '')
                if 'spk_' in seg_item_speaker_label:
                    
                    seg_item['speaker_label'] = seg_item_speaker_label.replace("spk_", "speaker")
        items = data['results'].get('items', [])
        
        for item in items:
            item_speaker_label = item.get('speaker_label', '')
            
            if 'spk_' in item_speaker_label:
                item['speaker_label'] = item_speaker_label.replace("spk_", "speaker")

    return data

def filter_labels_whisper(data):
    """
    ------------------------------------------------------------------------------------------------------

    replaces speaker labels in Whisper JSON.

    Parameters:
    ...........
    data : JSON
        The JSON containing speaker labels.

    Returns:
    ...........
    data : JSON
        The modified JSON with replaced speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    for segment in data.get('segments', []):
        current_speaker = segment.get('speaker', '')

        if 'SPEAKER_0' in current_speaker:
            segment["speaker"] = current_speaker.replace("SPEAKER_0", "speaker")

        for word in segment["words"]:
            word_speaker = word.get('speaker', '')
            
            if 'SPEAKER_0' in word_speaker:
                word["speaker"] = word_speaker.replace("SPEAKER_0", "speaker")

    for word_segment in data.get('word_segments', []):
        word_seg_speaker = word_segment.get('speaker', '')
        
        if 'SPEAKER_0' in word_seg_speaker: 
            word_segment["speaker"] = word_seg_speaker.replace("SPEAKER_0", "speaker")

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
    if len(content_dict) <2:
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
    if input_param['speaker_labels'] == True:#replace speaker labels with standard names

        response = filter_labels_aws(response)
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
    response = json.dumps({})
    settings = {}
    transcript = ""

    try:
        if input_param['access_key'] and input_param['secret_key']:
            transcribe = boto3.client('transcribe', region_name = input_param['region'], 
                                      aws_access_key_id = input_param['access_key'], 
                                      aws_secret_access_key = input_param['secret_key'])
        else:
            transcribe = boto3.client('transcribe', region_name = input_param['region'])

        if input_param['speaker_labels'] == True and input_param['max_speakers']>=2:
            settings = {'ShowSpeakerLabels': input_param['speaker_labels'], 'MaxSpeakerLabels': input_param['max_speakers']}

        transcribe.start_transcription_job(
            TranscriptionJobName=input_param['job_name'],
            Media={'MediaFileUri': s3uri},
            LanguageCode=input_param['language'],
            Settings=settings)

        status = get_job_status(transcribe, input_param)
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            response, transcript = filter_transcript_response(status, input_param)

    except Exception as e:
        logger.error(f"AWS Transcription job failed with file: {s3uri} exception: {e}")

    finally:
        return response, transcript

def get_whisperx_content(data):
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

    if 'segments' in data:           # Check if 'segments' key is present
        item_data = data['segments']
        
        spk_0_text = [item['text'] for item in item_data if item.get('speaker', '') == 'speaker0']
        spk_1_text = [item['text'] for item in item_data if item.get('speaker', '') == 'speaker1']
        
        spk0_check = any(not char.isspace() for char in spk_0_text)
        spk1_check = any(not char.isspace() for char in spk_1_text)

        if spk0_check and spk1_check:
            content_dict['speaker0'] = " ".join(spk_0_text)
            content_dict['speaker1'] = " ".join(spk_1_text)

    return content_dict

def replace_whisperx_speaker_labels(json_data, check_label, replace_label):
    """
    ------------------------------------------------------------------------------------------------------

    Replaces speaker labels in json response.

    Parameters:
    ...........
    data : json
        The json containing speaker labels.
    check_labels: list
        Check on input speaker labels
    speaker_labels: list
        Expected speaker labels

    Returns:
    ...........
    data : json
        The modified json with replaced speaker labels.

    ------------------------------------------------------------------------------------------------------
    """
    updated_data = json_data.copy()
    if 'segments' in updated_data:
        
        for segment in updated_data['segments']:
            if 'speaker' not in segment:
                continue
                
            if segment['speaker'] == check_label[0]:
                segment['speaker'] = replace_label[0]
    
            elif segment['speaker'] == check_label[1]:
                segment['speaker'] = replace_label[1]

    return updated_data

def get_whisperx_clinical_labels(scale, measures, content_dict, json_response):
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
    if len(content_dict) <2:
        return json_response
    
    score_string = scale.lower()+'_string'
    spk1_score = sutil.match_transcript(measures[score_string], content_dict['speaker0'])
    spk2_score = sutil.match_transcript(measures[score_string], content_dict['speaker1'])
    
    if spk1_score > spk2_score:
        json_response = replace_whisperx_speaker_labels(json_response, ['speaker0', 'speaker1'], ['clinician', 'participant'])

    else:
        json_response = replace_whisperx_speaker_labels(json_response, ['speaker0', 'speaker1'], ['participant', 'clinician'])

    return json_response
