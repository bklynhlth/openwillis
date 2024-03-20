# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

import numpy as np
import pandas as pd

from openwillis.measures.audio.util import acoustic_util as autil

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()


def get_summary(sound, framewise, sig_df, df_silence, measures):
    """
    ------------------------------------------------------------------------------------------------------
    Calculates the summary statistics for a given audio file.

    Parameters:
    ...........
    sound : Praat sound object
        the audio file to analyze.
    framewise : pandas dataframe
        a dataframe containing the fundamental frequency, loudness, HNR, and formant frequency values for
        each frame in the audio file.
    sig_df : pandas dataframe
        a dataframe containing the jitter, shimmer, and GNE values for the audio file.
    df_silence :pandas dataframe
        a dataframe containing the silence intervals in the audio file.
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    df_concat : pandas dataframe
        a dataframe containing all the summary statistics for the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    df_list = []
    col_list = list(framewise.columns)

    for col in col_list:
        com_summ = autil.common_summary(framewise[col], col)
        df_list.append(com_summ)

    summ_silence = autil.silence_summary(sound, df_silence, measures)
    voice_pct = autil.voice_frame(sound, measures)

    df_concat = pd.concat(df_list+ [sig_df, summ_silence, voice_pct], axis=1)
    return df_concat

def get_voiced_segments(df_silence, min_duration, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts the frames containing voice using the silence window values.

    Parameters:
    ........... 
    df_silence : pandas dataframe
        dataframe containing the silence window values
    min_duration : int
        minimum duration of the voiced segment (in ms)
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    speech_indices : list
        list containing the indices of the voiced segments

    ------------------------------------------------------------------------------------------------------
    """
    if len(df_silence) == 0:
        return None
    elif len(df_silence) == 1:
        return np.arange(0, df_silence[measures['silence_start']][0] * 100)

    speech_durations = df_silence[measures['silence_start']] - df_silence[measures['silence_end']].shift(1)
    # first speech duration is the first silence start time - 0
    speech_durations[0] = df_silence[measures['silence_start']][0]
    # convert to ms
    speech_durations *= 1000
    # get indices of speech_durations > 100
    speech_indices = speech_durations[speech_durations > min_duration].index.tolist()

    speech_indices_expanded = np.array([])
    for idx in speech_indices:
        # multiply by 100 to get frame number
        if idx == 0:
            speech_start = 0
            speech_end = np.floor(df_silence[measures['silence_start']][idx] * 100)
        else:
            speech_start = np.ceil(df_silence[measures['silence_end']][idx-1] * 100)
            speech_end = np.floor(df_silence[measures['silence_start']][idx] * 100)

        speech_indices_expanded = np.append(speech_indices_expanded, np.arange(speech_start, speech_end))

    speech_indices_expanded = speech_indices_expanded.astype(int)
    return speech_indices_expanded


def vocal_acoustics(audio_path):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the vocal acoustic variables of an audio file.

    Parameters:
    ...........
    audio_path : str
        path to the audio file

    Returns:
    ...........
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values
    df_silence : pandas dataframe
        dataframe containing the silence window values
    df_summary : pandas dataframe
        dataframe containing the summary of all acoustic variables

    ------------------------------------------------------------------------------------------------------
    """
    try:
        sound, measures = autil.read_audio(audio_path)
        df_pitch = autil.pitchfreq(sound, measures, 75, 500)
        df_loudness = autil.loudness(sound, measures)

        df_jitter = autil.jitter(sound, measures)
        df_shimmer = autil.shimmer(sound, measures)

        df_hnr = autil.harmonic_ratio(sound, measures)
        df_gne = autil.glottal_ratio(sound, measures)
        df_formant = autil.formfreq(sound, measures)
        df_silence = autil.get_voice_silence(audio_path, 500, measures)

        framewise = pd.concat([df_pitch, df_formant, df_loudness, df_hnr], axis=1)
        sig_df = pd.concat([df_jitter, df_shimmer, df_gne], axis=1)

        df_summary = get_summary(sound, framewise, sig_df, df_silence, measures)
        return framewise, df_silence, df_summary

    except Exception as e:
        logger.error(f'Error in acoustic calculation- file: {audio_path} & Error: {e}')
