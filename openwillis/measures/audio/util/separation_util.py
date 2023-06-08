# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import os
import shutil
import piso

import pandas as pd
import numpy as np
from pydub import AudioSegment
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

def concat_audio(df_driad, audio_path):
    """
    ------------------------------------------------------------------------------------------------------

    Concatenates the audio segments corresponding to a set of turns.

    Parameters:
    ...........
    df_driad : pandas dataframe
        A DataFrame containing the turn information.
    audio_path : str
        The path to the audio file.

    Returns:
    ...........
    concat_audio : sound object
        The concatenated audio segment.

    ------------------------------------------------------------------------------------------------------
    """
    aud_list = []

    for index, row in df_driad.iterrows():
        try:

            sound = AudioSegment.from_wav(audio_path)
            st_index = row['start_time']*1000
            end_index = row['end_time']*1000

            split_aud = sound[st_index:end_index+1]
            aud_list.append(split_aud)

        except Exception as e:
            logger.error(f'Error in audio concationation: {e}')

    concat_audio = sum(aud_list)
    return concat_audio

def diart_speaker(df, speaker_list, audio_path, out_dir):
    """
    ------------------------------------------------------------------------------------------------------

    The function extracts the audio segments of the specified speakers from the audio file and saves them in
    the specified output directory. The function returns a list of filenames of the saved speaker audio segments.

    Parameters:
    ...........
    df : pandas dataframe
        a dataframe containing the diarization results with columns for start time, end time, and speaker
    speaker_list : list
        a list of strings containing the names of the speakers to extract audio segments for
    audio_path : str
        a string containing the file path of the audio file to extract audio segments from
    out_dir : str
        a string containing the file path of the output directory to save the extracted audio segments

    Returns:
    ...........
    speaker_audio : list
        a list of strings containing the filenames of the saved audio segments for the specified speakers

    ------------------------------------------------------------------------------------------------------
    """
    speaker_audio = []
    for speaker in speaker_list:
        try:

            file_name, _ = os.path.splitext(os.path.basename(audio_path))
            speaker_df = df[df['speaker']==speaker].reset_index(drop=True)

            if len(speaker_df)>0:
                speaker_segment = concat_audio(speaker_df, audio_path)

                out_file = file_name + '_' +speaker +'.wav'
                speaker_segment.export(os.path.join(out_dir, out_file), format="wav")
                speaker_audio.append(out_file)

        except Exception as e:
            logger.error(f'Error in diart seperation: {e}')

    return speaker_audio

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

def match_transcript(measures, speech):
    """
    ------------------------------------------------------------------------------------------------------

    The function uses a pre-trained BERT-based sentence transformer model to compute the similarity between
    the speech and a list of pre-defined PANSS (Positive and Negative Syndrome Scale) script sentences. It
    returns the average similarity score of the top 5 matches.

    Parameters:
    ...........
    measures : dict
        a dictionary of measures containing the PANSS script sentences
    speech : str
        a string containing the speech to be matched with the PANSS script sentences

    Returns:
    ...........
    match_score : float
        a float value representing the average similarity score of the top 5 matches

    ------------------------------------------------------------------------------------------------------
    """
    prob_list = []

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    panss_script = measures['panss_string'][1:-1].split(',')#hardcode for PANSS

    for script in panss_script:
        sen_list = [script, speech]

        sentence_embeddings = model.encode(sen_list)
        prob = get_similarity_prob(sentence_embeddings)
        prob_list.append(prob)

    prob_list.sort(reverse=True)
    match_score = np.mean(prob_list[:5]) #top 5 probability score
    return match_score

def rename_speech(match_list, speaker_audio, out_dir):
    """
    ------------------------------------------------------------------------------------------------------

    The function renames the audio segments based on the match scores such that the speaker with the highest
    match score is renamed as 'rater' and the other speaker is renamed as 'patient'.

    Parameters:
    ...........
    match_list : list
        a list of float values representing the match scores for the saved speaker audio segments
    speaker_audio : list
        a list of strings containing the filenames of the saved audio segments for the two speakers
    out_dir : str
        a string containing the file path of the output directory where the renamed audio segments are saved

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    if len(match_list)==2:

        rater_index = np.argmax(match_list)
        patient_index = np.argmin(match_list)

        rater_filename = speaker_audio[rater_index].replace('speaker0', 'rater').replace('speaker1', 'rater')
        patient_filename = speaker_audio[patient_index].replace('speaker0', 'patient').replace('speaker1', 'patient')

        #Add threshold in future
        os.rename(os.path.join(out_dir, speaker_audio[rater_index]), os.path.join(out_dir, rater_filename))
        os.rename(os.path.join(out_dir, speaker_audio[patient_index]), os.path.join(out_dir, patient_filename))

def annote_speaker(out_dir, measures, speaker_audio):
    """
    ------------------------------------------------------------------------------------------------------

    The function extracts speech from the audio segments, matches them with the PANSS script sentences, and
    renames the audio segments based on the match scores.

    Parameters:
    ...........
    out_dir : str
        a string containing the file path of the output directory where the audio segments are saved
    measures : dict
        a dictionary of measures containing the PANSS script sentences
    speaker_audio : list
        a list of strings containing the filenames of the saved audio segments for the two speakers

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    match_list = []

    for audio in speaker_audio:
        try:

            filepath = os.path.join(out_dir, audio)
            _, speech = stranscribe.speech_transcription(filepath, 'en-us', [0,300]) #hardcode for US-EN

            match_score = match_transcript(measures, speech)
            match_list.append(match_score)

        except Exception as e:
            logger.error(f'Error in speaker annotation: {e}')

    rename_speech(match_list, speaker_audio, out_dir)

def slice_audio(df, audio_path, out_dir, measures, c_scale):
    """
    ------------------------------------------------------------------------------------------------------

    The function extracts audio segments for the two speakers in the diarization results, matches the speech
    with the PANSS script sentences using BERT-based sentence embeddings, and renames the audio segments based
    on the match scores.

    Parameters:
    ...........
    df : pandas dataframe
        a dataframe containing the diarization results with columns for start time, end time, and speaker
    audio_path : str
        a string containing the file path of the audio file to extract audio segments from
    out_dir : str
        a string containing the file path of the output directory to save the extracted audio segments in
    measures : dict
        a dictionary of measures containing the PANSS script sentences
    c_scale : str
        a string specifying the cognitive scale used to analyze the speech

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    speaker_list = list(df['speaker'].unique())[:2]
    speaker_audio = diart_speaker(df, speaker_list, audio_path, out_dir)

    if str(c_scale).lower() == 'panns' and len(speaker_audio) == 2:
        annote_speaker(out_dir, measures, speaker_audio)

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
    if scale.lower() not in measures['scale'].strip("[]").replace(" ", "").split(","):

        signal_label['speaker1'] = signal[0]
        signal_label['speaker2'] = signal[1]
        return signal_label

    spk1_txt = ' '.join(df[df['speaker_label'] == 'spk_0'].reset_index(drop=True)['content'])
    spk2_txt = ' '.join(df[df['speaker_label'] == 'spk_1'].reset_index(drop=True)['content'])

    spk1_score = match_transcript(measures, spk1_txt)
    spk2_score = match_transcript(measures, spk2_txt)

    if spk1_score > spk2_score:
        signal_label['rater'] = signal[0]
        signal_label['patient'] = signal[1]

    else:
        signal_label['patient'] = signal[0]
        signal_label['rater'] = signal[1]
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

    if response['results']:
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
    spk0_audio: list
        List of numpy arrays representing the audio segments of speaker 0.
    spk1_audio: list
        List of numpy arrays representing the audio segments of speaker 1.

    ------------------------------------------------------------------------------------------------------
    """
    spk0_audio = []
    spk1_audio = []

    for index, row in df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        speaker_label = row['speaker_label']

        speaker_audio = audio_signal[start_time:end_time]
        speaker_array = np.array(speaker_audio.get_array_of_samples())

        if speaker_label == 'spk_0':
            spk0_audio.append(speaker_array)
        elif speaker_label == 'spk_1':
            spk1_audio.append(speaker_array)

    return spk0_audio, spk1_audio

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
    spk0_audio, spk1_audio = get_segment_signal(audio_signal, df)

    spk0_audio = np.concatenate(spk0_audio)
    spk1_audio = np.concatenate(spk1_audio)

    signal_label = get_patient_rater_label(df, measures, scale, [spk0_audio, spk1_audio])
    return signal_label
