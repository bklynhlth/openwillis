# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

import nltk
import numpy as np
import pandas as pd

from openwillis.measures.commons import get_config
from openwillis.measures.text.util import characteristics_util as cutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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


def speech_characteristics(json_conf, language="en-us", speaker_label=None):
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
    turn_df: pandas dataframe
        A dataframe containing turn summary information
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """
    measures = get_config(os.path.abspath(__file__), "text.json")
    word_df, phrase_df, turn_df, summ_df = cutil.create_empty_dataframes(measures)

    try:
        if bool(json_conf):
            cutil.download_nltk_resources()

            if is_amazon_transcribe(json_conf):
                filter_json, text_list, text_indices = filter_transcribe(
                    json_conf, measures, speaker_label
                )

                if len(filter_json) > 0 and len(text) > 0:
                    (
                        word_df,
                        phrase_df,
                        turn_df,
                        summ_df,
                    ) = cutil.process_language_feature(
                        filter_json,
                        [word_df, phrase_df, turn_df, summ_df],
                        text_list,
                        text_indices,
                        language,
                        ["start_time", "end_time"],
                        measures,
                    )
            else:
                words, text = filter_vosk(json_conf, measures)
                if len(text) > 0:
                    (
                        word_df,
                        phrase_df,
                        turn_df,
                        summ_df,
                    ) = cutil.process_language_feature(
                        json_conf,
                        [word_df, phrase_df, turn_df, summ_df],
                        [words, [], [], text],
                        [[], []],
                        language,
                        ["start", "end"],
                        measures,
                    )
            
            # if phrase_df is empty, then add a row of NaNs
            if phrase_df.empty:
                phrase_df.loc[0] = np.nan
            # if turn_df is empty, then add a row of NaNs
            if turn_df.empty:
                turn_df.loc[0] = np.nan
    except Exception as e:
        logger.error(f"Error in Speech Characteristics {e}")

    finally:
        return word_df, phrase_df, turn_df, summ_df
