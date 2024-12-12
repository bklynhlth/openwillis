# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

import whisperx
import gc 
import torch
import faster_whisper
from whisperx.vad import load_vad_model
from whisperx.asr import FasterWhisperPipeline, WhisperModel
from typing import Optional

import os
import json
import logging

from ..commons import get_config

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def delete_model(model):
    """
    ------------------------------------------------------------------------------------------------------

    delete model if low on GPU resources
    Parameters:
    ...........
    model : object
        loaded model object
    
    ------------------------------------------------------------------------------------------------------
    """
    gc.collect()
    torch.cuda.empty_cache()
    del model

def get_diarization(audio, align_json, device, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    Assign speaker labels
    Parameters:
    ...........
    audio : object
        audio signal object
    align_json: json
        aligned whisper transcribed output
    device : str
        device type
    input_param : dict
        A dictionary containing input parameters
    
    Returns:
    ...........
    json_response : JSON Object
        A transcription response object in JSON format

    ------------------------------------------------------------------------------------------------------
    """
    # Assign speaker labels
    if input_param['infra_model'][0]:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=input_param['hf_token'], device=device)
    else:
        diarize_model = input_param['infra_model'][2]

    if input_param['min_speakers'] == None and input_param['max_speakers'] == None:
        diarize_segments = diarize_model(audio)
    
    elif input_param['min_speakers'] == None and input_param['max_speakers'] != None:
        diarize_segments = diarize_model(audio, max_speakers = input_param['max_speakers'])
    
    elif input_param['min_speakers'] != None and input_param['max_speakers'] == None:
        diarize_segments = diarize_model(audio, min_speakers= input_param['min_speakers'])
        
    else:
        diarize_segments = diarize_model(audio, min_speakers=input_param['min_speakers'], max_speakers=input_param['max_speakers'])
        
    json_response = whisperx.assign_word_speakers(diarize_segments, align_json)
    return json_response

def get_transcribe_summary(json_response):
    """
    ------------------------------------------------------------------------------------------------------

    Assign speaker labels
    Parameters:
    ...........
    json_response: json
        whisper transcribed output
    
    Returns:
    ...........
    summary : str
        whisper transcribed test summary

    ------------------------------------------------------------------------------------------------------
    """
    summary = ""
    
    if 'segments' in json_response:
        summary = "".join([item['text'] for item in json_response["segments"] if item.get('text', '')])
    return summary

def transcribe_whisper(filepath, model, device, compute_type, batch_size, infra_model, language):
    """
    ------------------------------------------------------------------------------------------------------
   
    Transcribe with whisper (batched)
    
    Parameters:
    ...........
    filepath : str
        The path to the audio file to be transcribed.
    model: str
        name of the pretrained model
    device: str
        cpu vs gpu
    compute_type: str
        computation format
    batch_size: int
        batch size
    infra_model:list
        whisper model artifacts (this is optional param: to optimize willisInfra) 
    language: str
        language code

        
    ------------------------------------------------------------------------------------------------------
    """
    if infra_model[0]:
        model_whisp = load_model(model, device, compute_type=compute_type)
    
    else:
        model_whisp = infra_model[1] #passing param from willismeansure
    audio = whisperx.load_audio(filepath)

    transcribe_json = model_whisp.transcribe(audio, batch_size=batch_size, language=language)
    return transcribe_json, audio

def get_whisperx_diariazation(filepath, input_param):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribe an audio file using Whisperx.

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
    device = 'cpu'
    compute_type = input_param['compute_type']
    
    json_response = json.dumps({})
    transcript = ''
    
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            if not compute_type.startswith('float'):
                compute_type = "float16"

        transcribe_json, audio = transcribe_whisper(filepath, input_param['model'], device, compute_type, input_param['batch_size'], input_param['infra_model'], input_param['language'])
    
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=input_param['language'], device=device)
        align_json = whisperx.align(transcribe_json["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
        if input_param['del_model']:
            delete_model(model_a)
            
        json_response = get_diarization(audio, align_json, device, input_param)    
        transcript = get_transcribe_summary(json_response)
    
    except Exception as e:
        logger.info(f'Error in speech Transcription: {e} & File: {filepath}')
    return json_response, transcript

# Modified from WhisperX package to support faster-whisper==1.1.0
def load_model(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               vad_model_fp=None,
               vad_options=None,
               model : Optional[WhisperModel] = None,
               task="transcribe",
               download_root=None,
               threads=4):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        vad_model_fp: str - File path to the VAD model to use
        model: Optional[WhisperModel] - The WhisperModel instance to use.
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  get_config(os.path.dirname(os.path.abspath(__file__)), 'whisper_asr.json')
    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model_fp is not None:
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options, model_fp=vad_model_fp)
    else:
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
