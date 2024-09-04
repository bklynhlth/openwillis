# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        a dataframe containing the start time, end time, interval, and speaker columns

    ------------------------------------------------------------------------------------------------------
    """
    df = pd.DataFrame(start_time, columns=['start_time'])

    df['end_time'] = end_time
    df['interval'] =df['end_time'] - df['start_time']
    
    df['speaker'] = speaker_list
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

    Generates a labeled audio signal based on the given DataFrame, audio signal, scale, and measures (common function).

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

    Identifies speakers from two dataframes based on time intervals (common function).

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
    filtered_df = filtered_df.rename(columns={'conf': 'confidence', 'speaker': 'speaker_label', 'word': 'content', 'start': 'start_time', 'end': 'end_time'}).iloc[:, 2:]

    filtered_df = filtered_df[["start_time", "end_time", "confidence", "speaker_label", "content"]]
    speaker_count = filtered_df['speaker_label'].nunique()
    return filtered_df, speaker_count

def transcribe_response_to_dataframe(response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes(cloud:AWS) a response from a speech-to-text service into a pandas DataFrame.

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

def extract_data(segment_info):
    """
    ------------------------------------------------------------------------------------------------------

    extract data from word_info

    Parameters:
    ----------
    segment_info : object
        The phrase level transcribed data.

    Returns:
    -------
    df : pandas series
        The phrase level transcribed data in a pandas series.

    ------------------------------------------------------------------------------------------------------
    """
    words = segment_info.get("words", None)

    starts = [word.get("start", np.nan) for word in words]
    ends = [word.get("end", np.nan) for word in words]
    phrases = [word.get("word", "") for word in words]
    scores = [word.get("score", 0) for word in words]
    speakers = [segment_info.get("speaker", "no_speaker") for _ in words]

    return pd.DataFrame({"start": starts, "end": ends, "phrase": phrases, "score": scores, "speaker": speakers})

def whisperx_to_dataframe(json_response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes(local:whisperx) a json response into a pandas DataFrame.

    Parameters:
    ----------
    json_response : dict
        The response object containing the transcribed data.

    Returns:
    -------
    df : pandas DataFrame
        The transcribed data in a DataFrame.
    speakers: int
        The number of speakers detected in the transcription.

    ------------------------------------------------------------------------------------------------------
    """
    df = pd.DataFrame(columns=["start_time", "end_time", "content", "confidence", "speaker_label"])
    if 'segments' in json_response:
        
        segment_infos = json_response["segments"]
        df = pd.DataFrame(segment_infos).apply(extract_data, axis=1)
        df = pd.concat(df.tolist(), ignore_index=True)

        df = df[df["score"] > 0].reset_index(drop=True)
        df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
        
        df = df[df["speaker"] != "no_speaker"].reset_index(drop=True)
        df = df.rename(columns={"start": "start_time", "end": "end_time", "score": "confidence", "speaker": "speaker_label", "phrase": "content"})

    speakers = df['speaker_label'].nunique()
    return df, speakers

def vosk_to_dataframe(json_response):
    """
    ------------------------------------------------------------------------------------------------------

    Transcribes(local:vosk) a json response into a pandas DataFrame.

    Parameters:
    ----------
    json_response : dict
        The response object

    Returns:
    -------
    df : pandas DataFrame
        The transcribed data in a DataFrame.

    ------------------------------------------------------------------------------------------------------
    """
    df = pd.DataFrame(columns=["start_time", "end_time", "content", "confidence", "speaker_label"])
        
    df = pd.DataFrame(json_response)

    df = df[df["conf"] > 0].reset_index(drop=True)
    df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
    
    df = df.rename(columns={"start": "start_time", "end": "end_time", "conf": "confidence", "word": "content"})
    df['speaker_label'] = 'speaker0'

    return df

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
        phonation_df_temp = speaker_df_clean[speaker_df_clean['content'].isin(phonation_list)]
        phonation_df_temp['phonation'] = phonation

        phonation_df = pd.concat([phonation_df, phonation_df_temp])

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

    return phonation_dict
