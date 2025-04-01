# author:    Vijay Yadav
# website:   http://www.brooklyn.health

# import the required packages
import os
import json
import logging
from .util import transcribe_util as tutil
from .willisdiarize_aws import diarization_correction_aws

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

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
    input_param['language'] = kwargs.get('language', 'en-US')
    input_param['region'] = kwargs.get('region', 'us-east-1')

    input_param['job_name'] = kwargs.get('job_name', 'transcribe_job_01')
    input_param['speaker_labels'] = kwargs.get('speaker_labels', False)
    input_param['max_speakers'] = kwargs.get('max_speakers', 2)

    input_param['context'] = kwargs.get('context', '')
    input_param['context_model'] = kwargs.get('context_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    input_param['access_key'] = kwargs.get('access_key', '')
    input_param['secret_key'] = kwargs.get('secret_key', '')

    input_param['willisdiarize_endpoint'] = kwargs.get('willisdiarize_endpoint', '')
    input_param['willisdiarize_parallel'] = kwargs.get('willisdiarize_parallel', 1)

    return input_param

def speech_transcription_aws(s3_uri, **kwargs):
    """
    ------------------------------------------------------------------------------------------------------

    Speech transcription function that transcribes an audio file using Amazon Transcribe.

    Parameters:
    ...........
    s3_uri : str
        The S3 uri for the recording to be transcribed.
    kwargs: Object
        language : str, optional
            The language of the audio file (e.g. 'en-US', 'en-IN'). Default is 'en-US'.
        region : str, optional
            The AWS region to use (e.g. 'us-east-1'). Only applicable if model is 'aws'. Default is 'us-east-1'.
        job_name : str, optional
            The name of the transcription job. Only applicable if model is 'aws'. Default is 'transcribe_job_01'.
        access_key : str, optional
            AWS access key
        secret_key : str, optional
            AWS secret key
        speaker_labels : boolean, optional
            Show speaker labels
        max_speakers : int, optional
            Max number of speakers
        context : str, optional
            scale to use for slicing the separated audio files, if any.
        context_model : str, optional
            model to use for speaker identification, if context is provided.
        willisdiarize_endpoint : str, optional
            The SageMaker endpoint for the Willisdiarize API.
        willisdiarize_parallel : int, optional
            Whether to use parallel processing for Willisdiarize API.
            
    Returns:
    ...........
    json_response : JSON Object
        A transcription response object in JSON format
    transcript : str
        The transcript of the recording.
    willisdiarize_status : bool
        The status of the Willisdiarize API.

    ------------------------------------------------------------------------------------------------------
    """
    input_param = read_kwargs(kwargs)
    measures = get_config()
    json_response, transcript = tutil.transcribe_audio(s3_uri, input_param)
    willisdiarize_status = False

    present_labels = [x['speaker_label'] for x in json_response['results']['items'] if 'speaker_label' in x]
    if len(set(present_labels)) != 2:
        if input_param['speaker_labels'] == True and input_param['max_speakers'] == 1:
            for item in json_response['results']['items']:
                item['speaker_label'] = 'speaker_0'
        return json_response, transcript, willisdiarize_status

    if input_param['language'].lower()[:2] == 'en' and input_param['willisdiarize_endpoint'].lower() not in ['', 'none']:
        json_response, willisdiarize_status = diarization_correction_aws(
            json_response, input_param['willisdiarize_endpoint'],
            parallel_processing=input_param['willisdiarize_parallel'],
            region=input_param['region'], access_key=input_param['access_key'],
            secret_key=input_param['secret_key']
        )

    if input_param['speaker_labels'] == True and input_param['context'].lower() in measures['scale'].split(',') and input_param['context_model'] in measures['embedding_models']:
        content_dict = tutil.extract_content(json_response)
        json_response = tutil.get_clinical_labels(input_param['context'], measures, content_dict, json_response, input_param['context_model'])

    return json_response, transcript, willisdiarize_status
