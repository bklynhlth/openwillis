# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import numpy as np
import pandas as pd
import os
import wave
import json

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from openwillis.features.speech import util as ut
from openwillis.features.speech import aws_transcribe as aws

import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def run_vosk(filepath, language='en-us', transcribe_interval = []):
    """
    -----------------------------------------------------------------------------------------

    Speech transcribe

    Args:
        path: audio path

    Returns:
        results: Speech transcription (list of JSON dictionaries)

    -----------------------------------------------------------------------------------------
    """
    json_response = '{}'
    transcript = mono_filepath = ''

    try:
        if os.path.exists(filepath):

            measures = get_config()
            mono_filepath = stereo_to_mono(filepath, transcribe_interval)
            results = get_vosk(mono_filepath, language)

            ut.remove_dir(os.path.dirname(mono_filepath)) #Clean temp directory
            json_response, transcript = filter_speech(measures, results)

        else:
            logger.info('Audio file not available.')

    except Exception as e:
        ut.remove_dir(os.path.dirname(mono_filepath))#Clean temp directory
        logger.error('Error in speech Transcription: {}'.format(e))

    finally:
        return json_response, transcript

def filter_audio(filepath, t_interval):
    """
    Speaker annotation using time interval
    """
    sound = AudioSegment.from_wav(filepath)

    if len(t_interval)==2:
        sound = sound[int(t_interval[0])*1000 : int(t_interval[1])*1000]

    elif len(t_interval)==1:
        sound = sound[int(t_interval[0])*1000:]

    sound = sound.set_channels(1)
    return sound

def stereo_to_mono(filepath, t_interval):
    sound = filter_audio(filepath, t_interval)

    filename, _ = os.path.splitext(os.path.basename(filepath))
    dir_name = os.path.join(os.path.dirname(filepath), 'temp_mono_' + filename)

    ut.make_dir(dir_name)
    mono_filepath = os.path.join(dir_name, filename + '.wav')
    sound.export(mono_filepath, format="wav")
    return mono_filepath

def get_vosk(audio_path, lang):
    """
    Recognize speech using vosk model
    """
    model = Model(lang=lang)
    wf = wave.open(audio_path, "rb")

    recog = KaldiRecognizer(model, wf.getframerate())
    recog.SetWords(True)

    results = []
    while True:

        data = wf.readframes(4000) #Future work
        if len(data) == 0:
            break

        if recog.AcceptWaveform(data):
            partial_result = json.loads(recog.Result())
            results.append(partial_result)

    partial_result = json.loads(recog.FinalResult())
    results.append(partial_result)
    return results

def filter_speech(measures, results):
    result_key = []
    text_key = []
    transcript_dict = {}

    for res in results:
        dict_keys = res.keys()

        if 'result' in dict_keys and 'text' in dict_keys:
            result_key.extend(res['result'])
            text_key.append(res['text'])

    transcript_dict['result'] = result_key
    transcript_dict['text'] = ' '.join(text_key)
    return result_key, ' '.join(text_key)


def get_config():
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures

def speech_transcription(filepath, **kwargs):
    """
    -----------------------------------------------------------------------------------------

    Speech transcription function that transcribes an audio file using either Amazon Transcribe or Vosk.

    Args:
        filepath (str): The path to the audio file to be transcribed.
        kwargs (dict): Optional keyword arguments that can be used to configure the transcription.
            model (str): The transcription model to use ('aws' or 'vosk'). Default is 'vosk'.
            language (str): The language of the audio file (e.g. 'en-us', 'es', 'fr'). Default is 'en-us'.
            region (str): The AWS region to use (e.g. 'us-east-1'). Only applicable if model is 'aws'. Default is 'us-east-1'.
            job_name (str): The name of the transcription job. Only applicable if model is 'aws'. Default is 'trns_job'.
            transcribe_interval (list): A list of tuples representing the start and end times (in seconds) of segments
            of the audio file to be transcribed. Only applicable if model is 'vosk'. Default is an empty list.

    Returns:
        A tuple of two values: the JSON response from the transcription service and the transcript of the audio file.

    -----------------------------------------------------------------------------------------
    """
    model = kwargs.get('model', 'vosk')
    language = kwargs.get('language', 'en-us')

    region = kwargs.get('region', 'us-east-1')
    job_name = kwargs.get('job_name', 'transcribe_job_01')
    transcribe_interval = kwargs.get('transcribe_interval', [])

    if model.lower() == 'aws':
        json_response, transcript = aws.transcribe_audio(filepath, region, job_name)

    else:
        json_response, transcript = run_vosk(filepath, language, transcribe_interval)

    return json_response, transcript
