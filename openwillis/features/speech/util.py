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

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openwillis.features.speech import speech_transcribe as stranscribe

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

#NLTK Tag list
tag_dict = {'PRP': 'Pronoun', 'PRP$': 'Pronoun', 'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb' , 'VBN': 'Verb',
            'VBP': 'Verb', 'VBZ': 'Verb', 'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective', 'NN': 'Noun',
            'NNP': 'Noun', 'NNS': 'Noun'}

def filter_rttm_line(line):
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

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def clean_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)

def clean_prexisting(temp_dir, temp_rttm):
    #Clean prexisting dir
    clean_dir(temp_dir)
    clean_dir(temp_rttm)

def make_temp_dir(out_dir, temp_dir, temp_rttm):
    #Make dir
    make_dir(out_dir)
    make_dir(temp_dir)
    make_dir(temp_rttm)

def temp_process(out_dir, file_path, audio_path):
    temp_dir = os.path.join(out_dir, file_path + '_temp')
    temp_rttm = os.path.join(out_dir, file_path + '_rttm')

    clean_prexisting(temp_dir, temp_rttm)#clean dir
    make_temp_dir(out_dir, temp_dir, temp_rttm)#Make dir

    shutil.copy(audio_path, temp_dir)
    return temp_dir, temp_rttm

def overalp_index(df):
    df_combine = df.copy()

    if len(df)>1:
        com_interval = pd.IntervalIndex.from_arrays(df["start_time"], df["end_time"])

        df["isOverlap"] = piso.adjacency_matrix(com_interval).any(axis=1).astype(int).values
        df_n_overlap = df[df['isOverlap']==0]
        df_overlap = df[(df['isOverlap']==1) & (df['interval']>.5)]

        df_combine = pd.concat([df_n_overlap, df_overlap]).reset_index(drop=True)
    return df_combine

def read_rttm(temp_dir, file_path):
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
    aud_list = []

    for index, row in df_driad.iterrows():
        try:

            sound = AudioSegment.from_wav(audio_path)
            st_index = row['start_time']*1000
            end_index = row['end_time']*1000

            split_aud = sound[st_index:end_index+1]
            aud_list.append(split_aud)

        except Exception as e:
            logger.info('Error in audio concationation...')

    concat_audio = sum(aud_list)
    return concat_audio

def diart_speaker(df, speaker_list, audio_path, out_dir):
    """
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
            logger.info('Error in diart seperation')

    return speaker_audio

def get_similarity_prob(sentence_embeddings):
    pscore = cosine_similarity([sentence_embeddings[0]],[sentence_embeddings[1]])
    prob = pscore[0][0]
    return prob

def match_transcript(measures, speech):
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
    if len(match_list)==2:

        rater_index = np.argmax(match_list)
        patient_index = np.argmin(match_list)

        rater_filename = speaker_audio[rater_index].replace('speaker0', 'rater').replace('speaker1', 'rater')
        patient_filename = speaker_audio[patient_index].replace('speaker0', 'patient').replace('speaker1', 'patient')

        #Add threshold in future
        os.rename(os.path.join(out_dir, speaker_audio[rater_index]), os.path.join(out_dir, rater_filename))
        os.rename(os.path.join(out_dir, speaker_audio[patient_index]), os.path.join(out_dir, patient_filename))

def annote_speaker(out_dir, measures, speaker_audio):
    match_list = []

    for audio in speaker_audio:
        try:

            filepath = os.path.join(out_dir, audio)
            _, speech = stranscribe.speech_transcription(filepath, 'en-us', [0,300]) #hardcode for US-EN

            match_score = match_transcript(measures, speech)
            match_list.append(match_score)

        except Exception as e:
            logger.info('Error in speaker annotation')

    rename_speech(match_list, speaker_audio, out_dir)

def slice_audio(df, audio_path, out_dir, measures, c_scale):
    speaker_list = list(df['speaker'].unique())[:2]
    speaker_audio = diart_speaker(df, speaker_list, audio_path, out_dir)

    if str(c_scale).lower() == 'panns' and len(speaker_audio) == 2:
        annote_speaker(out_dir, measures, speaker_audio)

def prepare_diart_interval(start_time, end_time, speaker_list):
    df = pd.DataFrame(start_time, columns=['start_time'])

    df['end_time'] = end_time
    df['interval'] =df['end_time'] - df['start_time']
    df['speaker'] = speaker_list

    df = overalp_index(df)
    return df

def get_diart_interval(diarization):
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
            logger.info('Error in pyannote filtering')

    df = prepare_diart_interval(start_time, end_time, speaker_list)
    return df

def download_nltk_resources():
    """
    -----------------------------------------------------------------------------------------

    This function downloads the required NLTK resources for processing text data.

    -----------------------------------------------------------------------------------------
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

def get_tag(text, tag_dict, measures):
    """
    -----------------------------------------------------------------------------------------

    This function performs part-of-speech tagging on the input text using NLTK, and returns a
    dataframe containing the part-of-speech tags.

    Args:
        text: The input text to be analyzed.
        tag_dict: A dictionary mapping the NLTK tags to more readable tags.
        measures: A dictionary containing the names of the columns in the output dataframes.

    Returns:
        tag_df: A dataframe containing the part-of-speech tags for the input text.

    -----------------------------------------------------------------------------------------
    """
    tag_list = nltk.pos_tag(text.split())

    tag_df = pd.DataFrame(tag_list, columns=[measures['word'], measures['tag']])
    tag_df = tag_df.replace({measures['tag']: tag_dict})
    return tag_df

def get_tag_summ(tag_df, summ_df, word, measures):
    """
    -----------------------------------------------------------------------------------------

    This function calculates the proportions of verbs, pronouns, adjectives, and nouns in the
    transcribed text, and adds them to the output dataframe summ_df.

    Args:
        tag_df: A dataframe containing the part-of-speech tags for the input text.
        summ_df: A dataframe containing the speech characteristics of the input text.
        word: The input text as a list of words.
        measures: A dictionary containing the names of the columns in the output dataframes.

    Returns:
        summ_df: The updated summ_df dataframe.

    -----------------------------------------------------------------------------------------
    """
    word_len = len(word) if len(word)>0 else 1

    verb = len(tag_df[tag_df[measures['tag']] == 'Verb'])/word_len
    pronoun = len(tag_df[tag_df[measures['tag']] == 'Pronoun'])/word_len
    adj = len(tag_df[tag_df[measures['tag']] == 'Adjective'])/word_len
    noun = len(tag_df[tag_df[measures['tag']] == 'Noun'])/word_len

    tag_object = [len(word), verb, adj, pronoun, noun]
    cols = [measures['tot_words'], measures['speech_verb'], measures['speech_adj'], measures['speech_pronoun'],
            measures['speech_noun']]

    summ_df.loc[0, cols] = tag_object
    return summ_df

def get_sentiment(summ_df, word, text, measures):
    """
    -----------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using VADER, and adds them
    to the output dataframe summ_df.

    Args:
        summ_df: A dataframe containing the speech characteristics of the transcribed text.
        word: The input text as a list of words.
        text: The input text to be analyzed.
        measures: A dictionary containing the names of the columns in the output dataframes.

    Returns:
        summ_df: The updated summ_df dataframe.

    -----------------------------------------------------------------------------------------
    """
    sentiment = SentimentIntensityAnalyzer()
    sentiment_dict = sentiment.polarity_scores(text)

    mattr = get_mattr(word)
    cols = [measures['neg'], measures['neu'], measures['pos'], measures['compound'], measures['speech_mattr']]
    sent_list = list(sentiment_dict.values()) + [mattr]

    summ_df.loc[0, cols] = sent_list
    return summ_df

def get_mattr(word):
    """
    -----------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR) of the input text using the
    LexicalRichness library.

    Args:
        word: The input text as a list of words.

    Returns:
        mattr: The calculated MATTR value.

    -----------------------------------------------------------------------------------------
    """
    filter_punc = list(value for value in word if value not in ['.','!','?'])
    filter_punc = " ".join(str(filter_punc))
    mattr = np.nan

    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=lex_richness.words)

    return mattr

def get_stats(summ_df, ros, file_dur, pause_list, measures):
    """
    -----------------------------------------------------------------------------------------

    This function calculates various speech characteristic features of the input text, including pause rate,
    pause mean duration, and silence ratio, and adds them to the output dataframe summ_df.

    Args:
        summ_df: A dataframe containing the speech characteristics of the input text.
        ros: The rate of speech of the input text.
        file_dur: The duration of the input audio file.
        pause_list: A list of pause durations in the input audio file.
        measures: A dictionary containing the names of the columns in the output dataframes.

    Returns:
        summ_df: The updated summ_df dataframe.

    -----------------------------------------------------------------------------------------
    """
    pause_rate = (len(pause_list)/file_dur)*60

    pause_meandur = np.mean(pause_list)
    silence_ratio = np.sum(pause_list)/(file_dur - np.sum(pause_list))

    feature_list = [ros, pause_rate, pause_meandur, silence_ratio]
    col_list = [measures['rate_of_speech'], measures['pause_rate'], measures['pause_meandur'],
                measures['silence_ratio']]

    summ_df.loc[0, col_list] = feature_list
    return summ_df

def get_pause_feature(json_conf, summ_df, word, measures, time_index):
    """
    -----------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic features

    Args:
        json_conf: JSON response objects.
        summ_df: A dataframe containing the speech characteristics of the input text.
        word: Transcribed text as a list of words.
        measures: A dictionary containing the names of the columns in the output dataframes.
        time_index: A list containing the names of the columns in json that contain the start and end times of each word.

    Returns:
        df_feature: The updated pause feature dataframe.

    -----------------------------------------------------------------------------------------
    """
    # Check if json_conf is empty
    if len(json_conf) <=0:
        return summ_df

    # Initialize variables
    pause_list = []
    file_dur = float(json_conf[-1][time_index[1]]) - float(json_conf[0][time_index[0]])
    ros = (len(word)/ file_dur)*60

    # Convert json_conf to a pandas DataFrame
    df_diff = pd.DataFrame(json_conf)

    # Calculate the pause time between each word and add the results to pause_list
    df_diff['pause_diff'] = df_diff.apply(lambda row: float(row[time_index[1]]) - float(row[time_index[0]]), axis=1)
    pause_list = df_diff['pause_diff'].tolist()

    # Calculate speech characteristics related to pause and update summ_df
    df_feature = get_stats(summ_df, ros, file_dur, pause_list, measures)
    return df_feature

def process_language_feature(json_conf, df_list, text, language, measures, time_index):
    """
    -----------------------------------------------------------------------------------------

    This function processes the language features from json response.

    Args:
        json_conf: JSON response object ; df_list: Pandas dataframe
        text: Transcribed text ; language: language type.
        measures: A dictionary containing the names of the columns in the output dataframes.
        time_index: A list containing the start and end time index values for the pause feature analysis.

    Returns:
        tag_df: A dataframe containing the part-of-speech tags for the input text.
        summ_df: A dataframe containing the speech characteristics of the input text.

    -----------------------------------------------------------------------------------------
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    tag_df, summ_df = df_list

    word_list = nltk.tokenize.word_tokenize(text)
    summ_df = get_pause_feature(json_conf, summ_df, word_list, measures, time_index)

    if language == 'en-us':
        tag_df = get_tag(text, tag_dict, measures)

        summ_df = get_tag_summ(tag_df, summ_df, word_list, measures)
        summ_df = get_sentiment(summ_df, word_list, text, measures)
    return tag_df, summ_df
