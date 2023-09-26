# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import os
import shutil
import piso

import pandas as pd
import numpy as np
import logging

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openwillis.measures.audio import speech_transcribe as stranscribe
from openwillis.measures.audio.util import util as ut

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def filter_rttm_line(line):
    """
    ------------------------------------------------------------------------------------------------------

    Processes a line of an RTTM file and returns the relevant fields.

    Parameters:
    ...........
    line : (bytes)
        The line to process.

    Returns:
    ...........
    tuple: A tuple containing the turn onset time, turn duration, speaker ID, and file ID.
    ------------------------------------------------------------------------------------------------------
    """
    line = line.decode('utf-8').strip()
    fields = line.split()

    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)

    file_id = fields[1]
    speaker_id = fields[7]

    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError:
        raise IOError('Turn duration not FLOAT. LINE: "%s"' % line)

    if dur <= 0:
        raise IOError('Turn duration <= 0 seconds. LINE: "%s"' % line)

    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError:
        raise IOError('Turn onset not FLOAT. LINE: "%s"' % line)

    if onset < 0:
        raise IOError('Turn onset < 0 seconds. LINE: "%s"' % line)

    return onset, dur, speaker_id, file_id

def load_rttm(rttmf):
    """
    ------------------------------------------------------------------------------------------------------

    Loads an RTTM file and extracts the turn information.

    Parameters:
    ...........
    rttmf : str
        The path to the RTTM file.

    Returns:
    ...........
    turns: list
        A list of tuples containing the turn onset time, turn duration, speaker ID, and file ID.

    ------------------------------------------------------------------------------------------------------
    """
    with open(rttmf, 'rb') as f:
        turns = []

        speaker_ids = set()
        file_ids = set()

        for line in f:
            if line.startswith(b'SPKR-INFO'):
                continue

            turn = filter_rttm_line(line)
            turns.append(turn)
    return turns

def clean_prexisting(temp_dir, temp_rttm):
    """
    ------------------------------------------------------------------------------------------------------

    Deletes any existing temporary directories and RTTM files.

    Parameters:
    ...........
    temp_dir : str
        The path to the temporary directory.
    temp_rttm : str
        The path to the temporary RTTM file.

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    #Clean prexisting dir
    ut.clean_dir(temp_dir)
    ut.clean_dir(temp_rttm)

def make_temp_dir(out_dir, temp_dir, temp_rttm):
    """
    ------------------------------------------------------------------------------------------------------

    Creates a temporary directory and RTTM file for a given audio file.

    Parameters:
    ...........
    out_dir : str
        The path to the output directory
    temp_dir : str
        The path to the temporary directory
    temp_rttm : str
        The path to the temporary RTTM file

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    #Make dir
    ut.make_dir(out_dir)
    ut.make_dir(temp_dir)
    ut.make_dir(temp_rttm)

def temp_process(out_dir, file_path, audio_path):
    """
    ------------------------------------------------------------------------------------------------------

    Creates a temporary directory for an audio file and copies the audio file into it.

    Parameters:
    ...........
    out_dir : str
        The path to the output directory.
    file_path : str
        The name of the audio file.
    audio_path : str
        The path to the audio file.

    Returns:
    ...........
    temp_dir : str
        paths to the temporary directory
    temp_rttm: obj
        RTTM file object.

    ------------------------------------------------------------------------------------------------------
    """
    temp_dir = os.path.join(out_dir, file_path + '_temp')
    temp_rttm = os.path.join(out_dir, file_path + '_rttm')

    clean_prexisting(temp_dir, temp_rttm)#clean dir
    make_temp_dir(out_dir, temp_dir, temp_rttm)#Make dir

    shutil.copy(audio_path, temp_dir)
    return temp_dir, temp_rttm

def overalp_index(df):
    """
    ------------------------------------------------------------------------------------------------------

    Identifies overlapping turns in a DataFrame and removes any that overlap by more than

    Parameters:
    ...........
    df : pandas dataframe
        The DataFrame to process.

    Returns:
    ...........
    df_combine : pandas dataframe
        The updated DataFrame.

    ------------------------------------------------------------------------------------------------------
    """
    df_combine = df.copy()

    if len(df)>1:
        com_interval = pd.IntervalIndex.from_arrays(df["start_time"], df["end_time"])

        df["isOverlap"] = piso.adjacency_matrix(com_interval).any(axis=1).astype(int).values
        df_n_overlap = df[df['isOverlap']==0]
        df_overlap = df[(df['isOverlap']==1) & (df['interval']>.5)]

        df_combine = pd.concat([df_n_overlap, df_overlap]).reset_index(drop=True)
    return df_combine

def read_rttm(temp_dir, file_path):
    """
    ------------------------------------------------------------------------------------------------------

    Reads the turn information from a temporary RTTM file.

    Parameters:
    ...........
    temp_dir : str
        The path to the temporary directory.
    file_path : str
        The name of the audio file.

    Returns:
    ...........
    rttm_df : pandas dataframe
        A DataFrame containing the turn information.

    ------------------------------------------------------------------------------------------------------
    """
    rttm_df = pd.DataFrame()
    rttm_file = os.path.join(temp_dir, file_path + '.rttm')

    if os.path.exists(rttm_file):
        rttm_info = load_rttm(rttm_file)

        rttm_df = pd.DataFrame(rttm_info, columns=['start_time', 'interval', 'speaker', 'filename'])
        rttm_df['end_time'] = rttm_df['start_time'] + rttm_df['interval']
        rttm_df = rttm_df.drop(columns=['filename'])

        rttm_df = overalp_index(rttm_df)
    return rttm_df

def get_similarity_prob(sentence_embeddings):
    """
    ------------------------------------------------------------------------------------------------------

    This function takes in a list of sentence embeddings and computes the cosine similarity between them,
    and returns the similarity score as a float.

    Parameters:
    ...........
    sentence_embeddings : list
        a list of sentence embeddings as numpy arrays

    Returns:
    ...........
    prob : float
        a float value representing the cosine similarity between the two input sentence embeddings

    ------------------------------------------------------------------------------------------------------
    """
    pscore = cosine_similarity([sentence_embeddings[0]],[sentence_embeddings[1]])
    prob = pscore[0][0]
    return prob

def match_transcript(sigma_string, speech):
    """
    ------------------------------------------------------------------------------------------------------

    The function uses a pre-trained BERT-based sentence transformer model to compute the similarity between
    the speech and a list of pre-defined PANSS or MADRS script sentences. It
    returns the average similarity score of the top 5 matches.

    Parameters:
    ...........
    sigma_string : str
        a string of sigma script
    speech : str
        a string containing the speech to be matched with the PANSS or MADRS script sentences

    Returns:
    ...........
    match_score : float
        a float value representing the average similarity score of the top 5 matches

    ------------------------------------------------------------------------------------------------------
    """
    prob_list = []

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    sigma_script = sigma_string.split(',')

    for script in sigma_script:
        sen_list = [script, speech]

        sentence_embeddings = model.encode(sen_list)
        prob = get_similarity_prob(sentence_embeddings)
        prob_list.append(prob)

    prob_list.sort(reverse=True)
    match_score = np.mean(prob_list[:5]) #top 5 probability score
    return match_score

def prepare_diart_interval(start_time, end_time, speaker_list):
    """
    ------------------------------------------------------------------------------------------------------

    The function checks for overlapping intervals and merges them if necessary.

    Parameters:
    ...........
    start_time : list
        a list of start times as floats
    end_time : list
        a list of end times as floats
    speaker_list : list
        a list of speaker names as strings

    Returns:
    ...........
    df : pandas dataframe
        a dataframe containing the start time, end time, interval, and speaker columns with overlapping
        intervals merged if necessary

    ------------------------------------------------------------------------------------------------------
    """
    df = pd.DataFrame(start_time, columns=['start_time'])

    df['end_time'] = end_time
    df['interval'] =df['end_time'] - df['start_time']
    df['speaker'] = speaker_list

    df = overalp_index(df)
    return df

def get_diart_interval(diarization):
    """
    ------------------------------------------------------------------------------------------------------

    This function takes in a pyannote.core.SegmentManification object representing the diarization results
    and returns a pandas dataframe with columns for start time, end time, interval, and speaker.

    Parameters:
    ...........
    diarization : pyannote.core.SegmentManification
        a diarization object representing the diarization results

    Returns:
    ...........
    df : pandas dataframe
        a dataframe containing the start time, end time, interval, and speaker columns

    ------------------------------------------------------------------------------------------------------
    """
    start_time = []
    end_time = []
    speaker_list = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        try:

            speaker_id = str(speaker).split('_')[1]
            speaker_id = int(speaker_id)
            start_time.append(turn.start)

            end_time.append(turn.end)
            speaker_list.append('speaker'+ str(speaker_id))

        except Exception as e:
            logger.error(f'Error in pyannote filtering: {e}')

    df = prepare_diart_interval(start_time, end_time, speaker_list)
    return df

def get_patient_rater_label(df, measures, scale, signal):
    """
    ------------------------------------------------------------------------------------------------------

    This function takes in a pandas dataframe 'df' containing diarization results and returns a dictionary
    signal_label with the labels assigned to the speakers based on the comparison of their scores.

    Parameters:
    ----------
    df : pandas DataFrame
        A dataframe containing the diarization results
    measures : dict
        A dictionary with config values.
    scale : str
        A clinical scale.
    signal : list
        A list of audio signals.

    Returns:
    -------
    signal_label : dict
        A dictionary with the assigned labels for the speakers.

    ------------------------------------------------------------------------------------------------------
    """
    signal_label = {}
    spk1_txt = ' '.join(df[df['speaker_label'] == 'speaker0'].reset_index(drop=True)['content'])
    spk2_txt = ' '.join(df[df['speaker_label'] == 'speaker1'].reset_index(drop=True)['content'])

    if scale.lower() not in measures['scale'].split(","):
        return signal

    elif spk1_txt == '' and spk2_txt == '': #Check empty text
        return signal
    
    score_string = scale.lower()+'_string'
    spk1_score = match_transcript(measures[score_string], spk1_txt)
    
    spk2_score = match_transcript(measures[score_string], spk2_txt)
    signal_label = {'clinician': signal['speaker1'], 'participant':signal['speaker0']}

    if spk1_score > spk2_score:
        signal_label = {'clinician': signal['speaker0'], 'participant':signal['speaker1']}
    return signal_label

def transcribe_response_to_dataframe(response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes a response from a speech-to-text service into a pandas DataFrame.

    Parameters:
    ----------
    response : dict
        The response object containing the transcribed data.

    Returns:
    -------
    df : pandas DataFrame
        The transcribed data in a DataFrame.
    speakers: int
        The number of speakers detected in the transcription.

    ------------------------------------------------------------------------------------------------------
    """
    speakers = 0
    df = pd.DataFrame()

    if 'results' in response:
        if 'speaker_labels' in response['results']:

            if 'speakers' in response['results']['speaker_labels']:
                speakers = response['results']['speaker_labels']["speakers"]

            if 'items' in response['results']:
                items = response['results']["items"]
                df = pd.DataFrame(items)

                df["confidence"] = df["alternatives"].apply(lambda x: x[0]["confidence"])
                df["content"] = df["alternatives"].apply(lambda x: x[0]["content"])
                df["confidence"] = df["confidence"].astype(float)

                df = df[df["confidence"] > 0].reset_index(drop=True)
                df = df[["start_time", "end_time", "confidence", "speaker_label", "content"]]
    return df, speakers

def get_segment_signal(audio_signal, df):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts speaker-specific segments from an audio signal based on a dataframe.

    Parameters:
    ----------
    audio_signal : AudioSignal
        The audio signal.
    df : pandas DataFrame
        The dataframe containing speaker information.

    Returns:
    -------
    signal_dict: dict
        A dictionary with key as labled speaker and value as a List of numpy arrays representing the audio segments
        of that speaker.

    ------------------------------------------------------------------------------------------------------
    """
    signal_dict = {}

    for index, row in df.iterrows():
        start_time = row['start_time']

        end_time = row['end_time']
        speaker_label = row['speaker_label']

        speaker_audio = audio_signal[start_time:end_time]
        speaker_array = np.array(speaker_audio.get_array_of_samples())

        if speaker_label in ['speaker0', 'speaker1', 'clinician', 'participant']:
            signal_dict.setdefault(speaker_label, []).append(speaker_array)

    return signal_dict

def generate_audio_signal(df, audio_signal, scale, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Generates a labeled audio signal based on the given DataFrame, audio signal, scale, and measures.

    Parameters:
    ----------
    df : pandas DataFrame
        The DataFrame containing the start and end times of segments.
    audio_signal : AudioSignal
        The original audio signal.
    scale : str
        A clinical scale
    measures : dict
        A config dictionary.

    Returns:
    -------
    signal_label: dict
        A dictionary with the assigned labels for the speakers and audio signals.

    ------------------------------------------------------------------------------------------------------
    """
    df['start_time'] = df['start_time'].astype(float) *1000
    df['end_time'] = df['end_time'].astype(float)*1000

    signal_dict = get_segment_signal(audio_signal, df)
    signal_dict = {key: np.concatenate(value) for key, value in signal_dict.items()}

    signal_label = get_patient_rater_label(df, measures, scale, signal_dict)
    return signal_label

def get_speaker_identification(df1, df2):
    """
    ------------------------------------------------------------------------------------------------------

    Identifies speakers from two dataframes based on time intervals.

    Parameters:
    ----------
    df1 : pandas DataFrame
        First dataframe containing speaker information.
    df2 : pandas.DataFrame
        Second dataframe containing time interval information.

    Returns:
    -------
    grouped_df : pandas.DataFrame
        A dataframe with aggregated speaker information.
    speaker_count : int
        The number of unique speaker labels identified.

    ------------------------------------------------------------------------------------------------------
    """
    # Merge the dataframes based on overlapping time intervals
    merged_df = pd.merge(df1, df2, how='cross')

    # Filter the merged dataframe based on overlapping intervals
    filtered_df = merged_df[ (merged_df['start'] >= merged_df['start_time']) &
                            (merged_df['end'] <= merged_df['end_time'])]

    # Groupby and aggregate word and conf
    grouped_df = filtered_df.groupby(['start_time', 'end_time', 'speaker']).agg({'word': ' '.join, 'conf': 'mean'}).reset_index()
    grouped_df = grouped_df.rename(columns={'conf': 'confidence', 'speaker': 'speaker_label', 'word': 'content'})

    grouped_df = grouped_df[["start_time", "end_time", "confidence", "speaker_label", "content"]]
    speaker_count = grouped_df['speaker_label'].nunique()
    return grouped_df, speaker_count
