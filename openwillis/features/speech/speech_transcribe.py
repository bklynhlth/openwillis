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

import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def speech_transcription(filepath, language='en-us', transcribe_interval = []):
    """
    -----------------------------------------------------------------------------------------
    
    Speech transcribe 
    
    Args:
        path: audio path 
        
    Returns:
        results: Speech transcription (list of JSON dictionaries)
        
    -----------------------------------------------------------------------------------------
    """
    try:
        mono_filepath = ''
        if os.path.exists(filepath): 
        
            measures = get_config()
            mono_filepath = stereo_to_mono(filepath, transcribe_interval)
            results = get_vosk(mono_filepath, language)
            
            ut.remove_dir(os.path.dirname(mono_filepath)) #Clean temp directory
            json_conf, transcript = filter_speech(measures, results)
            return json_conf, transcript
        
        else:
            logger.info('Audio file not available.')
            
    except Exception as e:
        ut.remove_dir(os.path.dirname(mono_filepath))#Clean temp directory
        logger.error('Error in speech Transcription')
        
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
    