# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

import string

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

PHONATIONS = {
    'a': ['ah', 'uh', 'huh'],
    'e': ['eh'],
    'i': [],
    'o': ['oh'],
    'u': [],
    'm': ['mm', 'hmm', 'hm', 'um']
}


def extract_phonation(speaker_df):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts phonation samples for each speaker.

    Parameters:
    ----------
    speaker_df : pandas DataFrame
        The speaker data.

    Returns:
    -------
    phonation_df : pandas DataFrame
        The phonation data.

    ------------------------------------------------------------------------------------------------------
    """
    speaker_df_clean = speaker_df.copy()
    speaker_df_clean['content'] = speaker_df_clean['content'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())

    phonation_df = pd.DataFrame()
    for phonation in PHONATIONS:
        phonation_list = PHONATIONS[phonation]
        phonation_df_temp = speaker_df_clean[speaker_df_clean['content'].isin(phonation_list)].copy()
        if phonation_df_temp.empty:
            continue

        phonation_df_temp.loc[:, 'phonation'] = phonation
        phonation_df = pd.concat([phonation_df, phonation_df_temp])

    if phonation_df.empty:
        return phonation_df

    phonation_df = phonation_df[phonation_df['end_time'].astype(float) - phonation_df['start_time'].astype(float) > 0.7]

    return phonation_df

def segment_phonations(audio_signal, phonation_df):
    """
    ------------------------------------------------------------------------------------------------------

    Segments phonations from an audio signal based on a DataFrame.

    Parameters:
    ----------
    audio_signal : AudioSignal
        The audio signal.
    phonation_df : pandas DataFrame
        The DataFrame containing phonation information.

    Returns:
    -------
    phonation_dict : dict
        A dictionary with key as the speaker label and value as a numpy array representing the phonation segment.

    ------------------------------------------------------------------------------------------------------
    """
    phonation_df['start_time'] = phonation_df['start_time'].astype(float) * 1000
    phonation_df['end_time'] = phonation_df['end_time'].astype(float) * 1000

    phonation_dict = {}
    phonation_counts = {spk + ph: 0 for spk in phonation_df['speaker_label'].unique() for ph in phonation_df['phonation'].unique()}

    for _, row in phonation_df.iterrows():
        try:
            start_time = row['start_time']
            end_time = row['end_time']

            speaker_label = row['speaker_label']

            speaker_audio = audio_signal[start_time:end_time]
            # strip silence more than 100ms in the beginning and end
            speaker_audio = speaker_audio.strip_silence(silence_len=100, silence_thresh=-30)

            if len(speaker_audio) < 500:
                continue

            speaker_array = np.array(speaker_audio.get_array_of_samples())

            phonation_dict[f"{speaker_label}_{row['phonation']}{phonation_counts[speaker_label+row['phonation']]}"] = speaker_array
            phonation_counts[speaker_label + row['phonation']] += 1
        except Exception as e:
            logger.info(f'Error segmenting phonation: {e}')

    return phonation_dict
