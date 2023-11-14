# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import numpy as np
import pandas as pd
import os
import json
import logging

from pydub import AudioSegment
from openwillis.measures.audio.util import util as ut
from openwillis.measures.audio.util import transcribe_util as tutil

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
    input_param: dict
        A dictionary containing input parameters with their corresponding values.

    ------------------------------------------------------------------------------------------------------
    """
    input_param = {}
    input_param['model'] = kwargs.get('model', 'tiny')
    input_param['language'] = kwargs.get('language', 'en')
    
    input_param['context'] = kwargs.get('context', '')
    input_param['max_speakers'] = kwargs.get('max_speakers', None)
    input_param['min_speakers'] = kwargs.get('min_speakers', None)

    input_param['hf_token'] = kwargs.get('hf_token', '')
    input_param['del_model'] = kwargs.get('del_model', False) #Temp filter
    input_param['infra_model'] = kwargs.get('infra_model', [True, None, None]) #Temp filter
    
    return input_param

def run_whisperx(filepath, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribe audio data using the WhisperX model.

    Parameters:
    ...........
    filepath : str
        The path to the audio file to be transcribed.
    input_param : dict
        A dictionary containing input parameters

    Returns:
    ...........
    json_response : JSON Object
        A transcription response object in JSON format
    transcript : str
        The transcript of the recording.

    ------------------------------------------------------------------------------------------------------
    """
    json_response = json.dumps({})
    transcript = ''
    
    if os.path.exists(filepath)== False or input_param['hf_token'] == '':
        return json_response, transcript
    
    from openwillis.measures.audio.util import whisperx_util as wutil #import in-case of model=whisperx
    json_response, transcript = wutil.get_whisperx_diariazation(filepath, input_param)
    
    if str(json_response) != '{}':
        json_response = tutil.filter_labels_whisper(json_response)
    return json_response, transcript
    

def speech_transcription_whisper(filepath, **kwargs):
    """
    ------------------------------------------------------------------------------------------------------

    Speech transcription function that transcribes an audio file using whisperx.

    Parameters:
    ...........
    filepath : str
        The path to the audio file to be transcribed.
    model : str, optional
        The transcription model to use ('vosk'). Default is 'vosk'.
    language : str, optional
        The language of the audio file (e.g. 'en-us', 'es', 'fr'). Default is 'en-us'.
    transcribe_interval : list, optional
        A list of tuples representing the start and end times (in seconds) of segments of the audio file to be transcribed.
        Only applicable if model is 'vosk'. Default is an empty list.

    Returns:
    ...........
    json_response : JSON Object
        A transcription response object in JSON format
    transcript : str
        The transcript of the recording.

    ------------------------------------------------------------------------------------------------------
    """
    measures = get_config()
    input_param = read_kwargs(kwargs)
    
    json_response, transcript = run_whisperx(filepath, input_param)
    if input_param['context'].lower() in measures['scale'].split(','):
        
        content_dict = tutil.get_whisperx_content(json_response)
        json_response = tutil.get_whisperx_clinical_labels(input_param['context'], measures, content_dict, json_response)
    return json_response, transcript
