# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

import nltk
import numpy as np
import pandas as pd
from openwillis.measures.text.util import characteristics_util as cutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def get_config(filepath, json_file):
    """
    ------------------------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes,
    and returns the contents of the file as a dictionary.

    Parameters:
    ...........
    filepath : str
        The path to the configuration file.
    json_file : str
        The name of the configuration file.

    Returns:
    ...........
    measures: A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(filepath)
    measure_path = os.path.abspath(os.path.join(dir_name, f"config/{json_file}"))

    file = open(measure_path)
    measures = json.load(file)
    return measures


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
    bool: True if the json response object
     is from Amazon Transcribe, False otherwise.
    ------------------------------------------------------------------------------------------------------
    """
    return "jobName" in json_conf and "results" in json_conf


def is_whisper_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks if the json response object is from Whisper Transcribe.

    Parameters:
    ...........
    json_conf: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object
     is from Whisper Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    if "segments" in json_conf:
        if "words" in json_conf["segments"][0]:
            return True
    return False


def filter_transcribe(json_conf, measures, speaker_label=None):
    """
    ------------------------------------------------------------------------------------------------------
    This function extracts the text and filters the JSON data
     for Amazon Transcribe json response objects.
     Also, it filters the JSON data based on the speaker label if provided.
    Parameters:
    ...........
    json_conf: dict
        aws transcribe json response.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    speaker_label: str
        Speaker label
    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    text_list: list
        List of transcribed text.
         split into words, phrases, turns, and full text.
    text_indices: list
        List of indices for text_list.
         for phrases and turns.
    Raises:
    ...........
    ValueError: If the speaker label is not found in the json response object.
    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["results"]["items"]

    # make a dictionary to map old indices to new indices
    item_data = cutil.create_index_column(item_data, measures)

    # extract text
    text = " ".join(
        [
            item["alternatives"][0]["content"]
            for item in item_data
            if "alternatives" in item
        ]
    )

    # phrase-split
    phrases, phrases_idxs = cutil.phrase_split(text)

    # turn-split
    turns = []
    turns_idxs = []

    if speaker_label is not None:

        turns_idxs, turns, phrases_idxs, phrases = cutil.filter_speaker(
            item_data, speaker_label, turns_idxs, turns, phrases_idxs, phrases
        )

    # entire transcript - by joining all the phrases
    text = " ".join(phrases)

    # filter json to only include items with start_time and end_time
    filter_json = cutil.filter_json_transcribe(item_data, speaker_label, measures)

    # extract words
    words = [word["alternatives"][0]["content"] for word in filter_json]

    text_list = [words, phrases, turns, text]
    text_indices = [phrases_idxs, turns_idxs]

    return filter_json, text_list, text_indices


def filter_whisper(json_conf, measures, speaker_label=None):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text and filters the JSON data
        for Whisper Transcribe json response objects.
        Also, it filters the JSON data based on the speaker label if provided.

    Parameters:
    ...........
    json_conf: dict
        whisper transcribe json response.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    speaker_label: str
        Speaker label

    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    text_list: list
        List of transcribed text.
            split into words, phrases, turns, and full text.
    text_indices: list
        List of indices for phrases and turns.

    Raises:
    ...........
    ValueError: If the speaker label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["segments"]

    # make a dictionary to map old indices to new indices
    item_data = cutil.create_index_column(item_data, measures)

    # phrase-split
    phrases_idxs, phrases = cutil.filter_phrases(
        item_data, speaker_label, measures
    )

    # turn-split
    if speaker_label is not None:
        turns_idxs, turns = cutil.filter_turns(
            item_data, speaker_label, measures
        )
    else:
        turns_idxs, turns = [], []


    # filter json to only include items with start_time and end_time
    filter_json = cutil.filter_json_transcribe(item_data, speaker_label, measures)

    # extract words
    words = [w["word"] for w in filter_json]

    # entire transcript - by joining all the phrases
    text = " ".join(phrases)

    text_list = [words, phrases, turns, text]
    text_indices = [phrases_idxs, turns_idxs]

    return filter_json, text_list, text_indices


def filter_vosk(json_conf, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text for json_conf objects
     from sources other than Amazon Transcribe.

    Parameters:
    ...........
    json_conf: dict
        The input text in the form of a JSON object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    words: list
        A list of words extracted from the JSON object.
    text: str
        The text extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    words = [word["word"] for word in json_conf if "word" in word]
    text = " ".join(words)

    # make a dictionary to map old indices to new indices
    for i, item in enumerate(json_conf):
        item[measures["old_index"]] = i


    return words, text


def speech_characteristics(json_conf, language="en", speaker_label=None):
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
    df_list: list, contains:
        word_df: pandas dataframe
            A dataframe containing word summary information
        phrase_df: pandas dataframe
            A dataframe containing phrase summary information
        turn_df: pandas dataframe
            A dataframe containing turn summary information
        summ_df: pandas dataframe
            A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """

    measures = get_config(os.path.abspath(__file__), "text.json")
    df_list = cutil.create_empty_dataframes(measures)

    try:
        if not isinstance(language, str):
            raise ValueError("Language should be a string")
        if len(language) < 2:
            # if language is not specified, then set it to "xx"
            # run speech characteristics as not english
            language = "xx"
        else:
            language = language[:2].lower()

        if bool(json_conf):
            cutil.download_nltk_resources()

            if is_whisper_transcribe(json_conf):
                filter_json, text_list, text_indices = filter_whisper(
                    json_conf, measures, speaker_label
                )

                if len(filter_json) > 0 and len(text_list[-1]) > 0:
                    df_list = cutil.process_language_feature(
                        filter_json, df_list, text_list,
                        text_indices, language, measures,
                    )
            else:
                words, text = filter_vosk(json_conf, measures)
                if len(text) > 0:
                    df_list = cutil.process_language_feature(
                        json_conf, df_list, [words, [], [], text],
                        [[], []], language, measures,
                    )
            
    except Exception as e:
        logger.error(f"Error in Speech Characteristics {e}")

    finally:
        # if word_df is empty, then add a row of NaNs
        if df_list[0].empty:
            df_list[0].loc[0] = np.nan
        # if phrase_df is empty, then add a row of NaNs
        if df_list[1].empty:
            df_list[1].loc[0] = np.nan
        # if turn_df is empty, then add a row of NaNs
        if df_list[2].empty:
            df_list[2].loc[0] = np.nan
        # if summ_df is empty, then add a row of NaNs
        if df_list[3].empty:
            df_list[3].loc[0] = np.nan

        return df_list
