# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import os
import logging

import numpy as np
import pandas as pd
from parselmouth.praat import run_file

from openwillis.measures.audio.util import acoustic_util as autil
from openwillis.measures.audio.util import disvoice_util as dutil

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
        a dataframe containing the jitter, shimmer, GNE values and Cepstral features for the audio file.
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

    df_relative = calculate_relative_stds(framewise, df_silence, measures)

    df_concat = pd.concat(df_list+ [sig_df, summ_silence, voice_pct, df_relative], axis=1)
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

def calculate_relative_stds(framewise, df_silence, measures):
    """
    ------------------------------------------------------------------------------------------------------
    
    Calculates the relative standard deviation of F0 and loudness for the voiced segments.

    Parameters:
    ...........
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values
    df_silence : pandas dataframe
        dataframe containing the silence window values
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    df_relative : pandas dataframe
        dataframe containing the relative standard deviation of F0 and loudness for the voiced segments

    ------------------------------------------------------------------------------------------------------
    """

    speech_indices = get_voiced_segments(df_silence, 100, measures)
    if speech_indices is None or len(speech_indices) == 0:
        speech_indices = np.arange(0, len(framewise))
    elif speech_indices[-1] < len(framewise):
        speech_indices = np.append(speech_indices, np.arange(speech_indices[-1], len(framewise)))

    f0 = framewise[measures['fundfreq']][speech_indices]
    loudness = framewise[measures['loudness']][speech_indices]

    relF0sd = np.std(f0) / np.mean(f0)
    relSE0SD = np.std(loudness) / np.mean(loudness)

    df_relative = pd.DataFrame([[relF0sd, relSE0SD]], columns=[measures['relF0sd'], measures['relSE0SD']])
    return df_relative

def calculate_glottal(audio_path):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the glottal features of an audio file.

    Parameters:
    ...........
    audio_path : str
        path to the audio file
    
    Returns:
    ...........
    glottal_features : list
        list containing the glottal features
         [mean_hrf, std_hrf, mean_naq, std_naq, mean_qoq, std_qoq]

    ------------------------------------------------------------------------------------------------------
    """

    glottal_features = dutil.extract_features_file(audio_path)

    return glottal_features

def calculate_tremor(audio_path):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the tremor features of an audio file.

    Parameters:
    ...........
    audio_path : str
        path to the audio file

    Returns:
    ...........
    tremor_features : list
        list containing the tremor features
        [FCoM, FTrC, FMon, FTrF, FTrI, FTrP, FTrCIP, FTrPS, FCoHNR, ACoM, ATrC, AMoN, ATrF, ATrI, ATrP, ATrCIP, ATrPS, ACoHNR]

    ------------------------------------------------------------------------------------------------------
    """
    tremor_dir = os.path.dirname(os.path.abspath(__file__))
    tremor_dir = os.path.join(tremor_dir, "util/praat_tremor")

    tremor_var = run_file(
        f"{tremor_dir}/vocal_tremor.praat",
        "4", audio_path, "0.015", "60", "350", "0.03", "0.3",
        "0.01", "0.35", "0.14", "2", "1.5", "15", "0.01", "0.15",
        "0.01", "0.01", "2", capture_output=True
    )

    # retrieve the tremor features
    tremor_features = tremor_var[1].replace('\n', '').split('\t')
    tremor_features2 = []
    for x in tremor_features[1:]:
        if x != '--undefined--':
            tremor_features2.append(float(x))
        else:
            tremor_features2.append(np.NaN)

    return tremor_features2

def get_advanced_summary(df_summary, audio_path, option, measures):
    """
    ------------------------------------------------------------------------------------------------------
    
    Calculates the summary statistics for a sustained vowel.

    Parameters:
    ...........
    df_summary : pandas dataframe
        dataframe containing the summary statistics for the audio file
    audio_path : str
        path to the audio file
    option : str
        whether to calculate the advanced vocal acoustic variables
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    df_summary : pandas dataframe
        dataframe containing the summary statistics for the audio file

    ------------------------------------------------------------------------------------------------------
    """

    glottal_cols = [
        measures["mean_hrf"], measures["std_hrf"],
        measures["mean_naq"], measures["std_naq"],
        measures["mean_qoq"], measures["std_qoq"]
    ]
    tremor_cols = [
        measures["FCoM"], measures["FTrC"], measures["FMon"],
        measures["FTrF"], measures["FTrI"], measures["FTrP"],
        measures["FTrCIP"], measures["FTrPS"], measures["FCoHNR"],
        measures["ACoM"], measures["ATrC"], measures["AMoN"],
        measures["ATrF"], measures["ATrI"], measures["ATrP"],
        measures["ATrCIP"], measures["ATrPS"], measures["ACoHNR"]
    ]

    if option == 'simple':
        tremor_summ = pd.DataFrame([[np.NaN] * 18], columns=tremor_cols)
        glottal_summ = pd.DataFrame([[np.NaN] * 6], columns=glottal_cols)
    elif option == 'tremor':
        tremor_features = calculate_tremor(audio_path)
        tremor_summ = pd.DataFrame([tremor_features], columns=tremor_cols)
        glottal_summ = pd.DataFrame([[np.NaN] * 6], columns=glottal_cols)
    else:
        tremor_features = calculate_tremor(audio_path)
        tremor_summ = pd.DataFrame([tremor_features], columns=tremor_cols)

        glottal_features = calculate_glottal(audio_path)
        glottal_summ = pd.DataFrame([glottal_features], columns=glottal_cols)

    df_summary = pd.concat([df_summary, glottal_summ, tremor_summ], axis=1)
    return df_summary

def vocal_acoustics(audio_path, option='simple'):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the vocal acoustic variables of an audio file.

    Parameters:
    ...........
    audio_path : str
        path to the audio file
    option : str
        whether to calculate the advanced vocal acoustic variables
        can be either 'simple', 'advanced' or 'tremor'

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
        if option not in ['simple', 'advanced', 'tremor']:
            raise ValueError("Option should be either 'simple', 'advanced' or 'tremor'")

        sound, measures = autil.read_audio(audio_path)
        df_pitch = autil.pitchfreq(sound, measures, 75, 500)
        df_loudness = autil.loudness(sound, measures)

        df_jitter = autil.jitter(sound, measures)
        df_shimmer = autil.shimmer(sound, measures)

        df_hnr = autil.harmonic_ratio(sound, measures)
        df_gne = autil.glottal_ratio(sound, measures)
        df_formant = autil.formfreq(sound, measures)
        df_silence = autil.get_voice_silence(audio_path, 500, measures)
        df_cepstral = autil.get_cepstral_features(audio_path, measures)

        framewise = pd.concat([df_pitch, df_formant, df_loudness, df_hnr], axis=1)
        sig_df = pd.concat([df_jitter, df_shimmer, df_gne, df_cepstral], axis=1)

        df_summary = get_summary(sound, framewise, sig_df, df_silence, measures)
        df_summary2 = get_advanced_summary(df_summary, audio_path, option, measures)
        return framewise, df_summary2

    except Exception as e:
        logger.error(f'Error in acoustic calculation- file: {audio_path} & Error: {e}')
