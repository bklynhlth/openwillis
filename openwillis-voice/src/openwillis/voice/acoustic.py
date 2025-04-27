# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import os
import logging

import scipy.stats
import numpy as np
import pandas as pd
from parselmouth.praat import run_file
from parselmouth import Sound

from .util import acoustic_util as autil
from .util import coordination_util as cutil

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()


def get_empty_dataframes(measures):
    """
    ------------------------------------------------------------------------------------------------------
    
    Creates empty dataframes for the framewise and summary statistics.

    Parameters:
    ...........
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    framewise : pandas dataframe
        empty dataframe containing pitch, loudness, HNR, and formant frequency values
    turns : pandas dataframe
        empty dataframe containing the summary statistics for each turn
    summary : pandas dataframe
        empty dataframe containing the summary statistics for the audio file

    ------------------------------------------------------------------------------------------------------
    """
    framewise = pd.DataFrame(columns=[
        measures['fundfreq'], measures['form1freq'], measures['form2freq'], measures['form3freq'], measures['form4freq'],
        measures['loudness'], measures['hnratio']
    ])
    summary = pd.DataFrame(columns=[
        measures['fundfreq'] + "_mean", measures['fundfreq'] + "_stddev", measures['fundfreq'] + "_range",
        measures['form1freq'] + "_mean", measures['form1freq'] + "_stddev", measures['form1freq'] + "_range",
        measures['form2freq'] + "_mean", measures['form2freq'] + "_stddev", measures['form2freq'] + "_range",
        measures['form3freq'] + "_mean", measures['form3freq'] + "_stddev", measures['form3freq'] + "_range",
        measures['form4freq'] + "_mean", measures['form4freq'] + "_stddev", measures['form4freq'] + "_range",
        measures['loudness'] + "_mean", measures['loudness'] + "_stddev", measures['loudness'] + "_range",
        measures['hnratio'] + "_mean", measures['hnratio'] + "_stddev", measures['hnratio'] + "_range",
        measures["jitter"], measures["jitterabs"], measures["jitterrap"], measures["jitterppq5"], measures["jitterddp"],
        measures["shimmer"], measures["shimmerdb"], measures["shimmerapq3"], measures["shimmerapq5"], measures["shimmerapq11"], measures["shimmerdda"],
        measures["gneratio"], measures["silence_ratio"], measures["relF0sd"], measures["relSE0SD"],
        measures["f0_kurtosis"], measures["f0_skewness"], measures["spir"], measures["pause_meddur"], measures["pause_maddur"],
        measures["mean_hrf"], measures["std_hrf"], measures["mean_naq"], measures["std_naq"], measures["mean_qoq"], measures["std_qoq"],
        measures["FCoM"], measures["FTrC"], measures["FMon"], measures["FTrF"], measures["FTrI"], measures["FTrP"], measures["FTrCIP"],
        measures["FTrPS"], measures["FCoHNR"], measures["ACoM"], measures["ATrC"], measures["AMoN"], measures["ATrF"], measures["ATrI"],
        measures["ATrP"], measures["ATrCIP"], measures["ATrPS"], measures["ACoHNR"],
        measures["mfcc1_mean"], measures["mfcc2_mean"], measures["mfcc3_mean"], measures["mfcc4_mean"], measures["mfcc5_mean"], measures["mfcc6_mean"],
        measures["mfcc7_mean"], measures["mfcc8_mean"], measures["mfcc9_mean"], measures["mfcc10_mean"], measures["mfcc11_mean"], measures["mfcc12_mean"],
        measures["mfcc13_mean"], measures["mfcc14_mean"], measures["mfcc1_var"], measures["mfcc2_var"], measures["mfcc3_var"], measures["mfcc4_var"],
        measures["mfcc5_var"], measures["mfcc6_var"], measures["mfcc7_var"], measures["mfcc8_var"], measures["mfcc9_var"], measures["mfcc10_var"],
        measures["mfcc11_var"], measures["mfcc12_var"], measures["mfcc13_var"], measures["mfcc14_var"], measures["cpp_mean"], measures["cpp_var"],
        measures["cpp_kurtosis"], measures["cpp_skewness"], measures["spectral_variability"], measures["lhr_mean"], measures["lhr_var"],
        measures["lhr_kurtosis"], measures["lhr_skewness"], measures["ACF1"], measures["ACF2"], measures["ACF3"]
    ])
    turns = summary.copy()

    return framewise, turns, summary

def get_framewise_summary(framewise):
    """
    ------------------------------------------------------------------------------------------------------
    Calculates the summary statistics for the framewise measures.

    Parameters:
    ...........
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values

    Returns:
    ...........
    summary : pandas dataframe
        dataframe containing the summary statistics for the framewise measures

    ------------------------------------------------------------------------------------------------------
    """
    summary = pd.DataFrame()

    for col in framewise.columns:
        com_summ = autil.common_summary(framewise[col], col)
        summary = pd.concat([summary, com_summ], axis=1)

    return summary

def get_summary(sound, framewise, framewise_voiced, df_silence, measures):
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
    framewise_voiced : pandas dataframe
        if voiced_segments=True, this dataframe contains the fundamental frequency, loudness, HNR, and formant frequency values for
        each voiced frame in the audio file else it is the same as framewise.
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
    summ_silence = autil.silence_summary(sound, df_silence, measures)
    voice_pct = autil.voice_frame(sound, measures)

    df_relative = calculate_relative_stds(framewise, df_silence, measures)
    df_f0_stats = calculate_f0_stats(framewise_voiced, measures)

    df_concat = pd.concat([summ_silence, voice_pct, df_relative, df_f0_stats], axis=1)
    return df_concat

def get_voiced_segments(df_silence, framewise, min_duration, measures):
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
        return np.arange(0, len(framewise))

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

    if len(speech_indices_expanded) == 0 or speech_indices_expanded is None:
        speech_indices_expanded = np.arange(0, len(framewise))
    elif np.floor(df_silence[measures['silence_end']][idx] * 100) < len(framewise):
        speech_dur = 1000 * (len(framewise)/100 - df_silence[measures['silence_end']][idx])
        if speech_dur > min_duration:
            speech_indices_expanded = np.append(speech_indices_expanded, np.arange(np.floor(df_silence[measures['silence_end']][idx] * 100), len(framewise)))

    speech_indices_expanded = speech_indices_expanded.astype(int)
    return speech_indices_expanded

def calculate_f0_stats(framewise, measures):
    """
    ------------------------------------------------------------------------------------------------------
    
    Calculates the F0 kurtosis and skewness.

    Parameters:
    ...........
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values
    measures : dict
        a dictionary containing the measures names for the calculated statistics.

    Returns:
    ...........
    df_f0_stats : pandas dataframe
        dataframe containing the F0 kurtosis and skewness

    ------------------------------------------------------------------------------------------------------
    """
    f0 = framewise[measures['fundfreq']]
    f0 = f0[~np.isnan(f0)]
    f0_kurtosis = scipy.stats.kurtosis(f0)
    f0_skewness = scipy.stats.skew(f0)

    df_f0_stats = pd.DataFrame([[f0_kurtosis, f0_skewness]], columns=[measures['f0_kurtosis'], measures['f0_skewness']])
    return df_f0_stats

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

    speech_indices = get_voiced_segments(df_silence, framewise, 100, measures)

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
    from .util import disvoice_util as dutil

    glottal_features = dutil.extract_features_file(audio_path)

    return glottal_features

def calculate_tremor(audio_path, duration_seconds):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the tremor features of an audio file.

    Parameters:
    ...........
    audio_path : str
        path to the audio file
    duration_seconds : float
        duration of the audio file

    Returns:
    ...........
    tremor_features : list
        list containing the tremor features
        [FCoM, FTrC, FMon, FTrF, FTrI, FTrP, FTrCIP, FTrPS, FCoHNR, ACoM, ATrC, AMoN, ATrF, ATrI, ATrP, ATrCIP, ATrPS, ACoHNR]

    ------------------------------------------------------------------------------------------------------
    """
    if duration_seconds < 3:
        return [np.NaN] * 18

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

def get_advanced_summary(audio_path, option, duration_seconds, measures):
    """
    ------------------------------------------------------------------------------------------------------
    
    Calculates the summary statistics for a sustained vowel.

    Parameters:
    ...........
    audio_path : str
        path to the audio file
    option : str
        whether to calculate the advanced vocal acoustic variables
        can be either 'simple', 'advanced' or 'tremor'
    duration_seconds : float
        duration of the audio file
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
        tremor_features = calculate_tremor(audio_path, duration_seconds)
        tremor_summ = pd.DataFrame([tremor_features], columns=tremor_cols)
        glottal_summ = pd.DataFrame([[np.NaN] * 6], columns=glottal_cols)
    else:
        tremor_features = calculate_tremor(audio_path, duration_seconds)
        tremor_summ = pd.DataFrame([tremor_features], columns=tremor_cols)

        glottal_features = calculate_glottal(audio_path)
        glottal_summ = pd.DataFrame([glottal_features], columns=glottal_cols)

    df_summary = pd.concat([glottal_summ, tremor_summ], axis=1)
    return df_summary

def process_audio(audio_path, framewise, summary, measures, voiced_segments=True, option='simple'):
    """
    ------------------------------------------------------------------------------------------------------

    Processes a single audio file and returns the framewise and summary statistics.

    Parameters:
    ...........
    audio_path : str
        path to the audio file
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values
    summary : pandas dataframe
        dataframe containing the summary statistics for the audio file
    measures : dict
        a dictionary containing the measures names for the calculated statistics.
    voiced_segments : bool
        whether to summarize framewise measures on voiced segments over 100ms
    option : str
        whether to calculate the advanced vocal acoustic variables

    Returns:
    ...........
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values
    summary : pandas dataframe
        dataframe containing the summary of all acoustic variables

    ------------------------------------------------------------------------------------------------------
    """
    try:
        sound = Sound(audio_path)
        duration_seconds = sound.get_total_duration()

        df_pitch = autil.pitchfreq(sound, measures, 75, 500)
        framewise[df_pitch.columns] = df_pitch

        df_formant = autil.formfreq(sound, measures)
        framewise[df_formant.columns] = df_formant

        df_loudness = autil.loudness(sound, measures)
        framewise[df_loudness.columns] = df_loudness

        df_hnr = autil.harmonic_ratio(sound, measures)
        framewise[df_hnr.columns] = df_hnr

        df_silence = autil.get_voice_silence(audio_path, 500, measures)
        framewise_voiced = framewise
        if voiced_segments:
            speech_indices = get_voiced_segments(df_silence, framewise, 100, measures)
            framewise_voiced = framewise.iloc[speech_indices]

        framewise_summary = get_framewise_summary(framewise_voiced)
        summary[framewise_summary.columns] = framewise_summary

        df_jitter = autil.jitter(sound, measures)
        summary[df_jitter.columns] = df_jitter

        df_shimmer = autil.shimmer(sound, measures)
        summary[df_shimmer.columns] = df_shimmer

        df_gne = autil.glottal_ratio(sound, measures)
        summary[df_gne.columns] = df_gne

        df_cepstral = autil.get_cepstral_features(audio_path, measures)
        summary[df_cepstral.columns] = df_cepstral

        df_lhr = autil.calculate_lhr(sound, measures)
        summary[df_lhr.columns] = df_lhr

        df_coordination = cutil.calculate_articulation_coordination(audio_path, measures)
        summary[df_coordination.columns] = df_coordination

        summary_measures = get_summary(sound, framewise, framewise_voiced, df_silence, measures)
        summary[summary_measures.columns] = summary_measures

        advanced_summary_measures = get_advanced_summary(audio_path, option, duration_seconds, measures)
        summary[advanced_summary_measures.columns] = advanced_summary_measures

    except Exception as e:
        logger.info(f'Error in acoustic calculation- file: {audio_path} & Error: {e}')
    finally:
        return framewise, summary


def vocal_acoustics(audio_path, voiced_segments = True, option='simple'):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the vocal acoustic variables of an audio file.

    Parameters:
    ...........
    audio_path : str | list
        path to the audio file
    voiced_segments : bool
        whether to summarize framewise measures on voiced segments over 100ms
    option : str
        whether to calculate the advanced vocal acoustic variables
        can be either 'simple', 'advanced' or 'tremor'

    Returns:
    ...........
    framewise : pandas dataframe
        dataframe containing pitch, loudness, HNR, and formant frequency values
    df_turns : pandas dataframe
        dataframe containing the summary of all acoustic variables for each turn
    df_summary : pandas dataframe
        dataframe containing the summary of all acoustic variables

    ------------------------------------------------------------------------------------------------------
    """
    measures = autil.get_measures()
    framewise, df_turns, df_summary = get_empty_dataframes(measures)

    try:
        if option not in ['simple', 'advanced', 'tremor']:
            raise ValueError("Option should be either 'simple', 'advanced' or 'tremor'")

        if isinstance(audio_path, str):
            framewise, df_summary = process_audio(audio_path, framewise, df_summary, measures, voiced_segments, option)
        elif isinstance(audio_path, list):
            for audio in audio_path:
                framewise_audio, _, df_summary_audio = get_empty_dataframes(measures)
                framewise_audio, df_summary_audio = process_audio(audio, framewise_audio, df_summary_audio, measures, voiced_segments, option)
                framewise_audio['file'] = audio
                df_summary_audio['file'] = audio
                if framewise.empty:
                    framewise = framewise_audio
                    df_turns = df_summary_audio
                else:
                    framewise = pd.concat([framewise, framewise_audio], axis=0)
                    df_turns = pd.concat([df_turns, df_summary_audio], axis=0)

            df_summary = df_turns.drop(columns=['file'])
            df_summary = df_summary.mean(axis=0).to_frame().T
        else:
            raise ValueError("Audio path should be a string or a list of strings")
    except Exception as e:
        logger.info(f'Error in acoustic calculation- file: {audio_path} & Error: {e}')
    finally:
        return framewise, df_turns, df_summary
