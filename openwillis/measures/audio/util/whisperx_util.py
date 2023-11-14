# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

import whisperx
import gc 
import torch

import json
import logging

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
    batch_size: str
        batch size
    infra_model:list
        whisper model artifacts (this is optional param: to optimize willisInfra) 
    language: str
        language code

        
    ------------------------------------------------------------------------------------------------------
    """
    if infra_model[0]:
        model_whisp = whisperx.load_model(model, device, compute_type=compute_type)
    
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
    compute_type = "int16"
    batch_size = 16
    
    json_response = json.dumps({})
    transcript = ''
    
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            compute_type = "float16"
    
        transcribe_json, audio = transcribe_whisper(filepath, input_param['model'], device, compute_type, batch_size, input_param['infra_model'], input_param['language'])
    
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=input_param['language'], device=device)
        align_json = whisperx.align(transcribe_json["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
        if input_param['del_model']:
            delete_model(model_a)
            
        json_response = get_diarization(audio, align_json, device, input_param)    
        transcript = get_transcribe_summary(json_response)
    
    except Exception as e:
        logger.error(f'Error in speech Transcription: {e} & File: {filepath}')
    return json_response, transcript
