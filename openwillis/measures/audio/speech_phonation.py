# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import tempfile
import os
import logging

from openwillis.measures.audio.util import separation_util as sutil
from openwillis.measures.audio.util import phonation_util as putil
from openwillis.measures.audio.acoustic import vocal_acoustics
from openwillis.measures.commons.common import to_audio
from pydub import AudioSegment
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def is_amazon_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------
    This function checks if the json response object is from Amazon Transcribe.
    Parameters:
    ...........
    json_conf: dict
        JSON response object.
    Returns:
    ...........
    bool: True if the json response object
     is from Amazon Transcribe, False otherwise.
    ------------------------------------------------------------------------------------------------------
    """
    return "jobName" in json_conf and "results" in json_conf

def is_whisper_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks if the json response object is from Whisper Transcribe.

    Parameters:
    ...........
    json_conf: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object
     is from Whisper Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    if "segments" in json_conf:
        if len(json_conf["segments"])>0:

            if "words" in json_conf["segments"][0]:
                return True
    return False

def phonation_extraction(filepath, transcript_json, speaker_label=''):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs phonation extraction from the audio signal based on the transcripted information.

    Parameters:
    ...........
    filepath : str
        Path to the input audio file.
    transcript_json : json
        Speech transcription json response.
    speaker_label : str
        Speaker label.

    Returns:
    ...........
    phonation_dict : pandas.DataFrame
        A pandas dataframe containing the phonation information.

    ------------------------------------------------------------------------------------------------------
    """
    phonation_dict = {}

    try:
        if not os.path.exists(filepath):
            return phonation_dict

        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        if is_whisper_transcribe(transcript_json):
            speaker_df, _ = sutil.whisperx_to_dataframe(transcript_json)
        elif is_amazon_transcribe(transcript_json):
            speaker_df, _ = sutil.transcribe_response_to_dataframe(transcript_json)
        else:
            speaker_df = sutil.vosk_to_dataframe(transcript_json)

        if speaker_label:
            speaker_df = speaker_df[speaker_df['speaker_label']==speaker_label]
            if len(speaker_df)==0:
                raise Exception(f'Speaker label {speaker_label} not found in the transcript')
        else:
            if speaker_df['speaker_label'].unique().shape[0]>1:
                raise Exception('Multiple speakers found in the transcript. Please provide a speaker label')

        phonation_df = putil.extract_phonation(speaker_df)

        if len(phonation_df)>0:
            phonation_dict = putil.segment_phonations(audio_signal, phonation_df)

    except Exception as e:
        logger.error(f'Error phonation extraction: {e} & File: {filepath}')

    return phonation_dict

def volume_normalization(audio_signal, target_dBFS):
    """
    ------------------------------------------------------------------------------------------------------
    
    Normalizes the volume of the audio signal to the target dBFS.
    
    Parameters:
    ...........
    audio_signal : pydub.AudioSegment
        input audio signal
    target_dBFS : float
        target dBFS
        
    Returns:
    ...........
    pydub.AudioSegment
        normalized audio signal

    ------------------------------------------------------------------------------------------------------
    """

    headroom = -audio_signal.max_dBFS
    gain_adjustment = target_dBFS - audio_signal.dBFS

    if gain_adjustment > headroom:
        gain_adjustment = headroom

    audio_signal = audio_signal.apply_gain(gain_adjustment)
    return audio_signal

def clean_acoustic_df(df, file, duration):
    """
    ------------------------------------------------------------------------------------------------------

    This function cleans the acoustic dataframe.

    Parameters:
    ...........
    df : pandas.DataFrame
        input dataframe
    file : str
        file name
    duration : float
        duration of the audio signal in seconds

    Returns:
    ...........
    pandas.DataFrame
        cleaned dataframe

    ------------------------------------------------------------------------------------------------------
    """
    df['phonation_type'] = file.split('_')[-1][0]
    df = df[['phonation_type'] + [col for col in df.columns if col != 'phonation_type']]
    df['duration'] = duration

    # remove unused columns
    ## pause related measures
    for col in ['spir', 'dur_med', 'dur_mad']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df

def phonations_acoustics(audio_path, transcript_json, speaker_label=''):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts phonation acoustics from the audio signal based on the transcripted information,
     then computes advanced vocal acoustics measures.

    Parameters:
    ...........
    audio_path : str
        Path to the input audio file.
    transcript_json : json
        Speech transcription json response.
    speaker_label : str
        Speaker label.

    Returns:
    ...........
    phonations_df : pandas.DataFrame
        A pandas dataframe containing the phonation-level measures.
    summ_df : pandas.DataFrame
        A pandas dataframe containing the summarized phonation acoustics measures.

    ------------------------------------------------------------------------------------------------------
    """

    phonations_df, summ_df = pd.DataFrame(), pd.DataFrame()

    try:
        if not os.path.exists(audio_path) or not transcript_json:
            raise Exception('Audio file or transcript json not found')

        # extract phonation segments
        phonation_dict = phonation_extraction(audio_path, transcript_json, speaker_label)
        if not phonation_dict:
            return phonations_df, summ_df

        # save phonation segments
        temp_dir = tempfile.mkdtemp()
        to_audio(audio_path, phonation_dict, temp_dir)

        for file in os.listdir(temp_dir):
            # standardize volume level
            audio_signal = AudioSegment.from_file(file = os.path.join(temp_dir, file), format = "wav")
            audio_signal = volume_normalization(audio_signal, -20)
            audio_signal.export(file, format="wav")

            # compute advanced vocal acoustics measures
            _, df = vocal_acoustics(os.path.join(temp_dir, file), option='advanced')
            df = clean_acoustic_df(df, file, audio_signal.duration_seconds)

            phonations_df = pd.concat([phonations_df, df])

        # summarize the phonation acoustics into dfs
        summ_df = phonations_df.groupby('phonation_type').mean().reset_index()
    
    except Exception as e:
        logger.error(f'Error in phonation acoustic calculation- file: {audio_path} & Error: {e}')

    return phonations_df, summ_df
