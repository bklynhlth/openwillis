# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import tempfile
import os
import shutil
import json
import logging

from .util import phonation_util as putil
from .acoustic import vocal_acoustics
from .commons import (
    to_audio, volume_normalization, whisperx_to_dataframe,
    transcribe_response_to_dataframe, vosk_to_dataframe
)
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
        audio_signal = AudioSegment.from_file(file = filepath, format = "wav")
        if is_whisper_transcribe(transcript_json):
            speaker_df, _ = whisperx_to_dataframe(transcript_json)
        elif is_amazon_transcribe(transcript_json):
            speaker_df, _ = transcribe_response_to_dataframe(transcript_json)
        else:
            speaker_df = vosk_to_dataframe(transcript_json)

        if speaker_label:
            speaker_df = speaker_df[speaker_df['speaker_label']==speaker_label]
            if speaker_df.empty:
                raise ValueError(f"Speaker label {speaker_label} not found in the transcript.")
        elif speaker_df['speaker_label'].nunique() > 1:
            raise ValueError("Multiple speakers found in the transcript. Please provide a speaker label.")

        phonation_df = putil.extract_phonation(speaker_df)
        if not phonation_df.empty:
            phonation_dict = putil.segment_phonations(audio_signal, phonation_df)

    except Exception as e:
        logger.info(f'Error phonation extraction: {e} & File: {filepath}')

    return phonation_dict

def clean_acoustic_df(df, file, duration, measures):
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
    measures : dict
        column names configuration

    Returns:
    ...........
    pandas.DataFrame
        cleaned dataframe

    ------------------------------------------------------------------------------------------------------
    """
    df[measures['phonation_type']] = file.split('_')[-1][0]
    df = df[[measures['phonation_type']] + [col for col in df.columns if col != measures['phonation_type']]].copy()
    df.loc[:, measures['duration']] = duration

    # remove unused columns - pause related measures
    cols_to_drop = [measures['spir'], measures['pause_meddur'], measures['pause_maddur'], measures['silence_ratio']]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    return df

def phonation_acoustics(audio_path, transcript_json, speaker_label=''):
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
    measure_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), './config/acoustic.json'))
    file = open(measure_path)
    measures = json.load(file)
    temp_dir = tempfile.mkdtemp()

    try:
        if audio_path.split('.')[-1] != 'wav':
            raise Exception('Audio file format not supported. Please provide a .wav file.')

        # extract phonation segments
        phonation_dict = phonation_extraction(audio_path, transcript_json, speaker_label)
        if not phonation_dict:
            logger.info(f'No phonation segments found in the audio file: {audio_path}')
            return phonations_df, summ_df
        logger.info(f'Found {len(phonation_dict)} phonation segments in the audio file: {audio_path}')
        # save phonation segments
        to_audio(audio_path, phonation_dict, temp_dir)

        for file in os.listdir(temp_dir):
            try:
                # standardize volume level
                audio_signal = AudioSegment.from_file(file = os.path.join(temp_dir, file), format = "wav")
                audio_signal = volume_normalization(audio_signal, -20)
                audio_signal.export(os.path.join(temp_dir, file), format="wav")

                # compute advanced vocal acoustics measures
                _, df = vocal_acoustics(os.path.join(temp_dir, file), option='advanced')
                df = clean_acoustic_df(df, file, audio_signal.duration_seconds, measures)

                phonations_df = pd.concat([phonations_df, df])
            except Exception as e:
                logger.info(f'Error in phonation acoustics calculation for single phonation: {file} & Error: {e}')

        # summarize the phonation acoustics into dfs
        logger.info(f'Phonation acoustics calculation completed for the audio file: {audio_path}')
        summ_df = phonations_df.groupby(measures['phonation_type']).mean().reset_index()
    
    except Exception as e:
        logger.info(f'Error in phonation acoustic calculation- file: {audio_path} & Error: {e}')
    finally:
        # clear the temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        return phonations_df, summ_df
