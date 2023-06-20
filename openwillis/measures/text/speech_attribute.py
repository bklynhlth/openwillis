# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import pandas as pd

import logging
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
    text = json_conf['results']['transcripts'][0].get('transcript', '')
    item_data = json_conf['results']['items']
    if speaker_label is not None:
        speaker_labels = [item['speaker_label'] for item in item_data if 'speaker_label' in item]

        if speaker_label not in speaker_labels:
            raise ValueError(f'Speaker label {speaker_label} not found in the json response object.')

        # filter the json data based on the speaker label
        item_data2 = [item for item in item_data if item.get('speaker_label', '') == speaker_label]

        # extract the text from the filtered json data
        text_list = [item['alternatives'][0]['content'] for item in item_data2 if 'alternatives' in item]
        text = " ".join(text_list)

    filter_json = [item for item in item_data if 'start_time' in item and 'end_time' in item]
    return text, filter_json

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
    text: str
        The input text extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    text_list = [word['word'] for word in json_conf if 'word' in word]
    text = " ".join(text_list)
    return text

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
                text, filter_json = filter_transcribe(json_conf, speaker_label=speaker_label)

                if len(filter_json) > 0 and len(text) > 0:
                    word_df, phrase_df, utterance_df, summ_df = cutil.process_language_feature(filter_json, [word_df, phrase_df, utterance_df, summ_df], text, language,
                                                               measures, ['start_time', 'end_time'], speaker_label=speaker_label)
            else:
                text = filter_vosk(json_conf)
                if len(text) > 0:
                    word_df, phrase_df, utterance_df, summ_df = cutil.process_language_feature(json_conf, [word_df, phrase_df, utterance_df, summ_df], text, language, measures,
                                                               ['start', 'end'])

    except Exception as e:
        logger.error(f'Error in speech Characteristics {e}')

    finally:
        return word_df, phrase_df, utterance_df, summ_df
