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

def get_diarization(audio, align_json, HF_TOKEN, device, num_speakers):
    """
    ------------------------------------------------------------------------------------------------------

    Assign speaker labels
    Parameters:
    ...........
    audio : object
        audio signal object
    align_json: json
        aligned whisper transcribed output
    HF_TOKEN : str
        The Hugging Face token for model authentication.
    device : str
        device type
    num_speakers: int
        Number of speaker
    
    Returns:
    ...........
    json_response : JSON Object
        A transcription response object in JSON format

    ------------------------------------------------------------------------------------------------------
    """
    # Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    if num_speakers == None:
        diarize_segments = diarize_model(audio)
    
    else:
        diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
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

def get_whisperx_diariazation(filepath, HF_TOKEN, del_model, num_speakers):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribe an audio file using Whisperx.

    Parameters:
    ...........
    filepath : str
        The path to the audio file to be transcribed.
    HF_TOKEN : str
        The Hugging Face token for model authentication.
    del_model: boolean
        Boolean indicator to delete model if low on GPU resources 
    num_speakers: int
        Number of speaker

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
    
    model = 'large-v2'
    batch_size = 16
    
    json_response = '{}'
    transcript = ''
    
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            compute_type = "float16"

        #Transcribe with original whisper (batched)
        model = whisperx.load_model(model, device, compute_type=compute_type)
        
        audio = whisperx.load_audio(filepath)
        transcribe_json = model.transcribe(audio, batch_size=batch_size)

        if del_model:
            delete_model(model)

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=transcribe_json["language"], device=device)
        align_json = whisperx.align(transcribe_json["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        if del_model:
            delete_model(model_a)
            
        json_response = get_diarization(audio, align_json, HF_TOKEN, device, num_speakers)    
        transcript = get_transcribe_summary(json_response)
    
    except Exception as e:
        logger.error(f'Error in speech Transcription: {e} & File: {filepath}')
    return json_response, transcript
