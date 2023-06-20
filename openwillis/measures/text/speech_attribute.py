# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

import nltk
import pandas as pd

from openwillis.measures.text.util import characteristics_util as cutil

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def get_config():
    """
    ------------------------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes,
    and returns the contents of the file as a dictionary.

    Parameters:
    ...........
    None

    Returns:
    ...........
    measures: A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    measure_path = os.path.abspath(os.path.join(dir_name, 'config/text.json'))

    file = open(measure_path)
    measures = json.load(file)
    return measures

def create_empty_dataframes():
    """
    ------------------------------------------------------------------------------------------------------

    Creating empty measures dataframes

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
    
    word_df = pd.DataFrame(columns=["pre_word_pause", "part_of_speech", "sentiment_pos",
                                    "sentiment_neg", "sentiment_neu", "sentiment_overall"])

    phrase_df = pd.DataFrame(columns=["pre_phrase_pause", "phrase_length_minutes", "phrase_length_words",
                                        "words_per_min", "pauses_per_min", "pause_variability",
                                        "mean_pause_length", "speech_percentage", "noun_percentage",
                                        "verb_percentage", "adjective_percentage", "pronoun_percentage",
                                        "sentiment_pos", "sentiment_neg", "sentiment_neu",
                                        "sentiment_overall", "mattr"])

    utterance_df = pd.DataFrame(columns=["pre_utterance_pause", "utterance_length_minutes",
                                            "utterance_length_words", "words_per_min", "pauses_per_min",
                                            "pause_variability", "mean_pause_length", "speech_percentage",
                                            "noun_percentage", "verb_percentage", "adjective_percentage",
                                            "pronoun_percentage", "sentiment_pos", "sentiment_neg",
                                            "sentiment_neu", "sentiment_overall", "mattr"])

    summ_df = pd.DataFrame(columns=["speech_length_minutes", "speech_length_words", "words_per_min",
                                        "pauses_per_min", "word_pause_length_mean", "word_pause_variability",
                                        "phrase_pause_length_mean", "phrase_pause_variability",
                                        "speech_percentage", "noun_percentage", "verb_percentage",
                                        "adjective_percentage", "pronoun_percentage", "sentiment_pos",
                                        "sentiment_neg", "sentiment_neu", "sentiment_overall", "mattr"])

    return word_df, phrase_df, utterance_df, summ_df

def is_amazon_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks if the json response object is from Amazon Transcribe.

    Parameters:
    ...........
    json_conf: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object is from Amazon Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    return 'jobName' in json_conf and 'results' in json_conf

def filter_transcribe(json_conf, speaker_label=None):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text and filters the JSON data for Amazon Transcribe json response objects.
     Also, it filters the JSON data based on the speaker label if provided.

    Parameters:
    ...........
    json_conf: dict
        aws transcribe json response.
    speaker_label: str
        Speaker label

    Returns:
    ...........
    phrases: list
        A list of phrases extracted from the JSON object.
    phrases_idxs: list
        A list of tuples containing the start and end indices of the phrases in the JSON object.
    utterances: list
        A list of utterances extracted from the JSON object.
    utterances_idxs: list
        A list of tuples containing the start and end indices of the utterances in the JSON object.
    text: str
        The text extracted from the JSON object.
         if speaker_label is not None, then only the text from the speaker label is extracted.
    filter_json: list
        The filtered JSON object containing only the relevant data for processing.

    Raises:
    ...........
    ValueError: If the speaker label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf['results']['items']

    # make a dictionary to map old indices to new indices
    for i, item in enumerate(item_data):
        item['old_idx'] = i
    text = " ".join([item['alternatives'][0]['content'] for item in item_data if 'alternatives' in item])


    # phrase-split
    phrases = nltk.tokenize.sent_tokenize(text)
    phrases_idxs = []

    start_idx = 0
    for phrase in phrases:
        end_idx = start_idx + len(phrase.split()) - 1
        phrases_idxs.append((start_idx, end_idx))
        start_idx = end_idx + 1

    # utterance-split
    utterances = []
    utterances_idxs = []

    if speaker_label is not None:
        speaker_labels = [item['speaker_label'] for item in item_data if 'speaker_label' in item]

        if speaker_label not in speaker_labels:
            raise ValueError(f'Speaker label {speaker_label} not found in the json response object.')

        # phrase-split for the speaker label
        phrases_idxs2 = []
        phrases2 = []
        for i, phrase in enumerate(phrases_idxs):
            start_idx = phrase[0]
            if item_data[start_idx].get('speaker_label', '') == speaker_label:
                phrases_idxs2.append(phrase)
                phrases2.append(phrases[i])

        phrases_idxs = phrases_idxs2
        phrases = phrases2

        # utterance-split for the speaker label
        start_idx = 0
        for i, item in enumerate(item_data):
            if i > 0 and item.get('speaker_label', '') == speaker_label and item_data[i - 1].get('speaker_label', '') != speaker_label:
                start_idx = i
            elif i > 0 and item.get('speaker_label', '') != speaker_label and item_data[i - 1].get('speaker_label', '') == speaker_label:
                utterances_idxs.append((start_idx, i - 1))
                # create utterances texts
                utterances.append(" ".join([item['alternatives'][0]['content'] for item in item_data[start_idx:i]]))

        if start_idx not in [item[0] for item in utterances_idxs]:
            utterances_idxs.append((start_idx, len(item_data) - 1))
            utterances.append(" ".join([item['alternatives'][0]['content'] for item in item_data[start_idx:]]))

    # entire transcript - by joining all the phrases
    text = " ".join(phrases)

    # filter json to only include items with start_time and end_time
    filter_json = [item for item in item_data if 'start_time' in item and 'end_time' in item]

    # calculate time difference between each word
    for i, item in enumerate(filter_json):
        if i > 0:
            item['time_diff'] = item['start_time'] - filter_json[i - 1]['end_time']
        else:
            item['time_diff'] = 0

    if speaker_label is not None:
        filter_json = [item for item in filter_json if item.get('speaker_label', '') == speaker_label]

    return phrases, phrases_idxs, utterances, utterances_idxs, text, filter_json

def filter_vosk(json_conf):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text for json_conf objects from sources other than Amazon Transcribe.

    Parameters:
    ...........
    json_conf: dict
        The input text in the form of a JSON object.

    Returns:
    ...........
    phrases: list
        A list of phrases extracted from the JSON object.
    phrases_idxs: list
        A list of tuples containing the start and end indices of the phrases in the JSON object.
    text: str
        The text extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    text_list = [word['word'] for word in json_conf if 'word' in word]
    text = " ".join(text_list)

    # phrase-split
    phrases = nltk.tokenize.sent_tokenize(text)
    phrases_idxs = []

    start_idx = 0
    for phrase in phrases:
        end_idx = start_idx + len(phrase.split()) - 1
        phrases_idxs.append((start_idx, end_idx))
        start_idx = end_idx + 1

    return phrases, phrases_idxs, text

def speech_characteristics(json_conf, language='en-us', speaker_label=None):
    """
    ------------------------------------------------------------------------------------------------------

    Speech Characteristics

    Parameters:
    ...........
    json_conf: dict
        Transcribed json file
    language: str
        Language type
    speaker_label: str
        Speaker label

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
    measures = get_config()
    word_df, phrase_df, utterance_df, summ_df = create_empty_dataframes()

    try:
        if bool(json_conf):
            cutil.download_nltk_resources()

            if is_amazon_transcribe(json_conf):
                phrases, phrases_idxs, utterances, utterances_idxs, text, filter_json = filter_transcribe(json_conf, speaker_label=speaker_label)

                if len(filter_json) > 0 and len(text) > 0:
                    word_df, phrase_df, utterance_df, summ_df = cutil.process_language_feature(filter_json, [word_df, phrase_df, utterance_df, summ_df],
                                                               [phrases, utterances, text], [phrases_idxs, utterances_idxs], language,
                                                               measures, ['start_time', 'end_time'])
            else:
                phrases, phrases_idxs, text = filter_vosk(json_conf)
                if len(text) > 0:
                    word_df, phrase_df, utterance_df, summ_df = cutil.process_language_feature(json_conf, [word_df, phrase_df, utterance_df, summ_df],
                                                               [phrases, [], text], [phrases_idxs, []], language,
                                                               measures, ['start', 'end'])

    except Exception as e:
        logger.error(f'Error in speech Characteristics {e}')

    finally:
        return word_df, phrase_df, utterance_df, summ_df
