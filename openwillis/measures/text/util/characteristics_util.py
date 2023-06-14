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
tag_dict = {'PRP': 'Pronoun', 'PRP$': 'Pronoun', 'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb' , 'VBN': 'Verb',
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

def get_tag(text, tag_dict, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech tagging on the input text using NLTK, and returns a
    dataframe containing the part-of-speech tags.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.
    tag_dict: dict
        A dictionary mapping the NLTK tags to more readable tags.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    tag_df: pandas dataframe
        A dataframe containing the part-of-speech tags for the input text.

    ------------------------------------------------------------------------------------------------------
    """
    tag_list = nltk.pos_tag(text.split())

    tag_df = pd.DataFrame(tag_list, columns=[measures['word'], measures['tag']])
    tag_df = tag_df.replace({measures['tag']: tag_dict})
    return tag_df

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

def get_pause_feature(json_conf, summ_df, word, measures, time_index):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic features

    Parameters:
    ...........
    json_conf: list
        JSON response objects.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    word: list
        Transcribed text as a list of words.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    time_index: list
        A list containing the names of the columns in json that contain the start and end times of each word.

    Returns:
    ...........
    df_feature: pandas dataframe
        The updated pause feature dataframe.

    ------------------------------------------------------------------------------------------------------
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
    df_diff['pause_diff'] = df_diff[time_index[0]].astype(float) - df_diff[time_index[1]].astype(float).shift(1)
    pause_list = df_diff['pause_diff'].tolist()[1:] # Remove the first NaN value

    # Calculate speech characteristics related to pause and update summ_df
    df_feature = get_stats(summ_df, ros, file_dur, pause_list, measures)
    return df_feature

def process_language_feature(json_conf, df_list, text, language, measures, time_index):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the language features from json response.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes.
    text: str
        Transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    time_index: list
        A list containing the names of the columns in json that contain the start and end times of each word.

    Returns:
    ...........
    tag_df: pandas dataframe
        A dataframe containing the part-of-speech tags for the input text.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.

    ------------------------------------------------------------------------------------------------------
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
