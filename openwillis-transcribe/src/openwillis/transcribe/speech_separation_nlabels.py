# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

from pyannote.audio import Pipeline
from .util import separation_util as sutil
from pydub import AudioSegment
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()
logging.getLogger('torch.distributed.nn.jit.instantiator').setLevel(logging.WARNING)

def get_config():
    """
    ------------------------------------------------------------------------------------------------------

    This function loads and returns a dictionary containing the configuration settings for speech separation
    from a JSON file.

    Parameters:
    ...........
    None

    Returns:
    ...........
    measures : dict
        A dictionary containing the configuration settings for speech separation.

    ------------------------------------------------------------------------------------------------------
    """
    #Loading json config
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/speech.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures

def run_pyannote(file_path, hf_token):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the provided audio file using the 'pyannote/speaker-diarization' speech diarization
    model, and returns a pandas dataframe containing the speaker diarization information.

    Parameters:
    ...........
    file_path : str
        Path to the input audio file.
    hf_token : str
        Access token for HuggingFace to access pre-trained models.

    Returns:
    ...........
    diart_df : pandas.DataFrame
        A pandas dataframe containing the speaker diarization information.

    ------------------------------------------------------------------------------------------------------
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=hf_token)
    diart = pipeline(file_path, num_speakers=2)
    
    diart_df = sutil.get_diart_interval(diart)
    diart_df = diart_df.sort_values(by=['start_time', 'end_time']).reset_index(drop=True)
    return diart_df

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
    input_param['hf_token'] = kwargs.get('hf_token', '')
    
    input_param['transcript_json'] = kwargs.get('transcript_json', json.dumps({}))
    input_param['context'] = kwargs.get('context', '')
    input_param['context_model'] = kwargs.get('context_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    input_param['volume_normalization'] = kwargs.get('volume_normalization', None)

    return input_param

def get_pyannote(input_param, filepath):
    """
    ------------------------------------------------------------------------------------------------------

    Retrieves speaker identification information using the local Diarization model.

    Parameters:
    ...........
    input_param : dict
        A dictionary containing input parameters
    filepath : str
        The file path.

    Returns:
    ...........
    speaker_df :pandas.DataFrame
        The speaker identification dataframe.
    speaker_count :int
        The number of identified speakers.

    ------------------------------------------------------------------------------------------------------
    """
    
    diart_df = run_pyannote(filepath, input_param['hf_token'])
    transcribe_df = pd.DataFrame(input_param['transcript_json'])

    speaker_df, speaker_count = sutil.get_speaker_identification(diart_df, transcribe_df)
    return speaker_df, speaker_count

def speaker_separation_nolabels(filepath, **kwargs):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs speaker separation using speech diarization techniques on the provided audio file.

    Parameters:
    ...........
    filepath : str
        Path to the input audio file.
    transcript_json : json
        Speech transcription json response.
    hf_token : str
        Access token for HuggingFace to access pre-trained models.
    context : str, optional
        scale to use for slicing the separated audio files, if any.
    context_model : str, optional
        model to use for speaker identification, if context is provided.
    volume_normalization : int
        The volume normalization level. Default is None.

    Returns:
    ...........
    signal_label : pandas.DataFrame
        A pandas dataframe containing the speaker diarization information.

    ------------------------------------------------------------------------------------------------------
    """
    signal_label = {}
    input_param = read_kwargs(kwargs)

    measures = get_config()

    try:
        if not os.path.exists(filepath) or 'transcript_json' not in kwargs:
            return signal_label
        
        if input_param['context_model'] not in measures['embedding_models']:
            raise ValueError(f"Model {input_param['context_model']} not found in the config file")

        speaker_df, speaker_count = get_pyannote(input_param, filepath)
        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")

        if len(speaker_df)>0 and speaker_count>1:
            combined_df = sutil.combine_turns(speaker_df)
            signal_label = sutil.generate_audio_signal(combined_df, audio_signal, input_param['context'], input_param['context_model'], measures)

            if input_param['volume_normalization']:
                if type(input_param['volume_normalization']) != int or input_param['volume_normalization'] < -60 or input_param['volume_normalization'] > 0:
                    logger.info('Volume normalization value should be an integer between -60 and 0')
                    return signal_label
                signal_label = sutil.adjust_volume(filepath, signal_label, input_param['volume_normalization'])

    except Exception as e:
        logger.info(f'Error in diard processing: {e} & File: {filepath}')

    return signal_label
