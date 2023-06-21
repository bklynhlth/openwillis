# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

#NLTK Tag list
TAG_DICT = {'PRP': 'Pronoun', 'PRP$': 'Pronoun', 'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb' , 'VBN': 'Verb',
            'VBP': 'Verb', 'VBZ': 'Verb', 'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective', 'NN': 'Noun',
            'NNP': 'Noun', 'NNS': 'Noun'}

def download_nltk_resources():
    """
    ------------------------------------------------------------------------------------------------------

    This function downloads the required NLTK resources for processing text data.

    Parameters:
    ...........
    None

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

def get_tag(json_conf, tag_dict):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech tagging on the input text using NLTK, and returns an updated
    json_conf list with the part-of-speech tags.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    tag_dict: dict
        A dictionary mapping the NLTK tags to more readable tags.

    Returns:
    ...........
    json_conf: list
        The updated json_conf list.

    ------------------------------------------------------------------------------------------------------
    """
    if len(json_conf) <=0:
        return json_conf
    
    if 'alternatives' not in json_conf[0].keys():
        # local vosk transcriber
        word_list = [word['word'] for word in json_conf if 'word' in word]
    else:
        # aws transcriber
        word_list = [item['alternatives'][0]['content'] for item in json_conf]

    tag_list = nltk.pos_tag(word_list)

    for i, tag in enumerate(tag_list):
        if tag[1] in tag_dict.keys():
            json_conf[i]['tag'] = tag_dict[tag[1]]
        else:
            json_conf[i]['tag'] = tag[1]

    return json_conf

def get_tag_summ(tag_df, summ_df, word, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the proportions of verbs, pronouns, adjectives, and nouns in the
    transcribed text, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    tag_df: pandas dataframe
        A dataframe containing the part-of-speech tags for the input text.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    word: list
        The input text as a list of words.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
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
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using VADER, and adds them
    to the output dataframe summ_df.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the transcribed text.
    word: list
        The input text as a list of words.
    text: str
        The input text to be analyzed.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
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
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR) of the input text using the
    LexicalRichness library.

    Parameters:
    ...........
    word : list
        The input text as a list of words.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
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
    ------------------------------------------------------------------------------------------------------

    This function calculates various speech characteristic features of the input text, including pause rate,
    pause mean duration, and silence ratio, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    ros: float
        The rate of speech of the input text.
    file_dur: float
        The duration of the input audio file.
    pause_list: list
        A list of pause durations in the input audio file.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    pause_rate = (len(pause_list)/file_dur)*60

    pause_meandur = np.mean(pause_list)
    silence_ratio = np.sum(pause_list)/(file_dur - np.sum(pause_list))

    feature_list = [ros, pause_rate, pause_meandur, silence_ratio]
    col_list = [measures['rate_of_speech'], measures['pause_rate'], measures['pause_meandur'],
                measures['silence_ratio']]

    summ_df.loc[0, col_list] = feature_list
    return summ_df

def process_pause_feature(df_diff, df, index_list, time_index, level_name):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic features
     at the phrase or utterance level and adds them to the output dataframe df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information from the JSON response.
    df: pandas dataframe
        A dataframe containing phrase or utterance summary information
    index_list: list
        A list containing the indices of the first and last word in each phrase or utterance.
    time_index: list
        A list containing the names of the columns in json that contain the start and end times of each word.
    level_name: str
        The name of the level being analyzed (phrase or utterance).

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    if level_name not in ['phrase', 'utterance']:
        logger.error('level_name must be either phrase or utterance')
        return df

    for j, index in enumerate(index_list):
        rng = range(index[0], index[1]+1)
        level_json = df_diff[df_diff['old_idx'].isin(rng)]

        pauses = level_json['pause_diff'].values[1:] # remove first pause as it is the pre_pause

        df.loc[j, f'{level_name}_length_minutes'] = (float(level_json.iloc[-1][time_index[1]]) - float(level_json.iloc[0][time_index[0]]))/60
        df.loc[j, f'{level_name}_length_words'] = len(level_json)

        df.loc[j, 'pause_variability'] = np.var(pauses)
        df.loc[j, 'mean_pause_length'] = np.mean(pauses)
        df.loc[j, 'speech_percentage'] = 1 - np.sum(pauses)/(60*df.loc[j, f'{level_name}_length_minutes'])

    df['words_per_min'] = df[f'{level_name}_length_words']/df[f'{level_name}_length_minutes']
    df['pauses_per_min'] = df['words_per_min']

    return df

def update_summ_df(df_diff, summ_df, time_index, word_df, phrase_df):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic features
     at the file level and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information from the JSON response.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    time_index: list
        A list containing the names of the columns in json that contain the start and end times of each word.
    word_df: pandas dataframe
        A dataframe containing word summary information
    phrase_df: pandas dataframe
        A dataframe containing phrase summary information

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    summ_df['speech_length_minutes'] = [(float(df_diff.iloc[-1][time_index[1]]) - float(df_diff.iloc[0][time_index[0]]))/60]
    summ_df['speech_length_words'] = len(df_diff)
    summ_df['words_per_min'] = summ_df['speech_length_words']/summ_df['speech_length_minutes']
    summ_df['pauses_per_min'] = summ_df['words_per_min']
    summ_df['word_pause_length_mean'] = word_df['pre_word_pause'].mean(skipna=True)
    summ_df['word_pause_variability'] = word_df['pre_word_pause'].var(skipna=True)
    summ_df['phrase_pause_length_mean'] = phrase_df['pre_phrase_pause'].mean(skipna=True)
    summ_df['phrase_pause_variability'] = phrase_df['pre_phrase_pause'].var(skipna=True)
    summ_df['speech_percentage'] = 1 - df_diff.loc[1:, 'pause_diff'].sum()/(60*summ_df['speech_length_minutes'])
    return summ_df

def get_pause_feature(json_conf, df_list, text_indices, time_index):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic features

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes.
            word_df, phrase_df, utterance_df, summ_df
    text_indices: list
        List of indices for text_list.
            for phrases and utterances.
    time_index: list
        A list containing the names of the columns in json that contain the start and end times of each word.

    Returns:
    ...........
    df_feature: list
        List of updated pandas dataframes.
            word_df, phrase_df, utterance_df, summ_df

    ------------------------------------------------------------------------------------------------------
    """
    # Check if json_conf is empty
    if len(json_conf) <=0:
        return df_list

    word_df, phrase_df, utterance_df, summ_df = df_list
    phrase_index, utterance_index = text_indices

    # Convert json_conf to a pandas DataFrame
    df_diff = pd.DataFrame(json_conf)

    # Calculate the pause time between each word and add the results to pause_list
    if 'pause_diff' not in df_diff.columns:
        df_diff['pause_diff'] = df_diff[time_index[0]].astype(float) - df_diff[time_index[1]].astype(float).shift(1)

    # word-level analysis
    phrase_starts = [pindex[0] for pindex in phrase_index]
    word_df['pre_word_pause'] = df_diff['pause_diff'].where(~df_diff['old_idx'].isin(phrase_starts), np.nan)

    # phrase-level analysis
    df_diff_phrase = df_diff[df_diff['old_idx'].isin(phrase_starts)] # get the rows corresponding to the start of each phrase

    if len(utterance_index) > 0:
        utterance_starts = [uindex[0] for uindex in utterance_index] # get the start index of each utterance    
        phrase_df['pre_phrase_pause'] = df_diff_phrase['pause_diff'].where(~df_diff_phrase['old_idx'].isin(utterance_starts), np.nan)
    else:
        phrase_df['pre_phrase_pause'] = df_diff_phrase['pause_diff']
    phrase_df = phrase_df.reset_index(drop=True)

    phrase_df = process_pause_feature(df_diff, phrase_df, phrase_index, time_index, 'phrase')

    # utterance-level analysis
    if len(utterance_index) > 0:
        df_diff_utterance = df_diff[df_diff['old_idx'].isin(utterance_starts)] # get the rows corresponding to the start of each utterance

        utterance_df['pre_utterance_pause'] = df_diff_utterance['pause_diff']
        utterance_df = utterance_df.reset_index(drop=True)

        utterance_df = process_pause_feature(df_diff, utterance_df, utterance_index, time_index, 'utterance')

    # file-level analysis
    summ_df = update_summ_df(df_diff, summ_df, time_index, word_df, phrase_df)

    df_feature = [word_df, phrase_df, utterance_df, summ_df]

    return df_feature

def process_language_feature(json_conf, df_list, text_list, text_indices, language, measures, time_index):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the language features from json response.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes.
         word_df, phrase_df, utterance_df, summ_df
    text_list: list
        List of transcribed text.
         split into phrases, utterances, and full text.
    text_indices: list
        List of indices for text_list.
         for phrases and utterances.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    time_index: list
        A list containing the names of the columns in json that contain the start and end times of each word.

    Returns:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information
    phrase_df: pandas dataframe
        A dataframe containing phrase summary information
    utterance_df: pandas dataframe
        A dataframe containing utterance summary information
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """

    df_list = get_pause_feature(
        json_conf, df_list, text_indices, time_index
    )

    if language == 'en-us':
        json_conf = get_tag(json_conf, TAG_DICT)

        summ_df = get_tag_summ(tag_df, summ_df, word_list, measures)
        summ_df = get_sentiment(summ_df, word_list, text, measures)
    return word_df, phrase_df, utterance_df, summ_df
