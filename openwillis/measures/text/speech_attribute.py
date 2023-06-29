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


def filter_speaker_label(item_data, speaker_label, turns_idxs, turns, phrases_idxs, phrases):
    """
    ------------------------------------------------------------------------------------------------------

    This function updates the turns and phrases lists
        to only include the speaker label provided.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    speaker_label: str
        Speaker label
    turns_idxs: list
        A list of tuples containing
            the start and end indices of the turns in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.
    phrases_idxs: list
        A list of tuples containing
            the start and end indices of the phrases in the JSON object.
    phrases: list
        A list of phrases extracted from the JSON object.

    Returns:
    ...........
    turns_idxs: list
        A list of tuples containing
            the start and end indices of the turns in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.
    phrases_idxs: list
        A list of tuples containing
            the start and end indices of the phrases in the JSON object.
    phrases: list
        A list of phrases extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """

    # phrase-split for the speaker label
    phrases_idxs2 = []
    phrases2 = []
    for i, phrase in enumerate(phrases_idxs):
        start_idx = phrase[0]
        if item_data[start_idx].get("speaker_label", "") == speaker_label:
            phrases_idxs2.append(phrase)
            phrases2.append(phrases[i])

    phrases_idxs = phrases_idxs2
    phrases = phrases2

    # turn-split for the speaker label
    start_idx = 0
    for i, item in enumerate(item_data):
        if (
            i > 0
            and item.get("speaker_label", "") == speaker_label
            and item_data[i - 1].get("speaker_label", "") != speaker_label
        ):
            start_idx = i
        elif (
            i > 0
            and item.get("speaker_label", "") != speaker_label
            and item_data[i - 1].get("speaker_label", "") == speaker_label
        ):
            turns_idxs.append((start_idx, i - 1))
            # create turns texts
            turns.append(
                " ".join(
                    [
                        item["alternatives"][0]["content"]
                        for item in item_data[start_idx:i]
                    ]
                )
            )

    if start_idx not in [item[0] for item in turns_idxs]:
        turns_idxs.append((start_idx, len(item_data) - 1))
        turns.append(
            " ".join(
                [
                    item["alternatives"][0]["content"]
                    for item in item_data[start_idx:]
                ]
            )
        )

    return turns_idxs, turns, phrases_idxs, phrases


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
    words: list
        A list of words extracted from the JSON object.
    phrases: list
        A list of phrases extracted from the JSON object.
    phrases_idxs: list
        A list of tuples containing
         the start and end indices of the phrases in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.
    turns_idxs: list
        A list of tuples containing
         the start and end indices of the turns in the JSON object.
    text: str
        The text extracted from the JSON object.
         if speaker_label is not None,
         then only the text from the speaker label is extracted.
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.

    Raises:
    ...........
    ValueError: If the speaker label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["results"]["items"]

    # make a dictionary to map old indices to new indices
    for i, item in enumerate(item_data):
        item[measures["old_index"]] = i
    text = " ".join(
        [
            item["alternatives"][0]["content"]
            for item in item_data
            if "alternatives" in item
        ]
    )

    # phrase-split
    phrases = nltk.tokenize.sent_tokenize(text)
    phrases_idxs = []

    start_idx = 0
    for phrase in phrases:
        end_idx = start_idx + len(phrase.split()) - 1
        phrases_idxs.append((start_idx, end_idx))
        start_idx = end_idx + 1

    # turn-split
    turns = []
    turns_idxs = []

    if speaker_label is not None:
        speaker_labels = [
            item["speaker_label"] for item
            in item_data if "speaker_label" in item
        ]

        if speaker_label not in speaker_labels:
            raise ValueError(
                f"Speaker label {speaker_label} "
                "not found in the json response object."
            )

        turns_idxs, turns, phrases_idxs, phrases = filter_speaker_label(
            item_data, speaker_label, turns_idxs, turns, phrases_idxs, phrases
        )

    # entire transcript - by joining all the phrases
    text = " ".join(phrases)

    # filter json to only include items with start_time and end_time
    filter_json = [
        item for item in item_data
        if "start_time" in item and "end_time" in item
    ]

    # calculate time difference between each word
    for i, item in enumerate(filter_json):
        if i > 0:
            item[measures["pause"]] = float(item["start_time"]) - float(
                filter_json[i - 1]["end_time"]
            )
        else:
            item[measures["pause"]] = np.nan

    if speaker_label is not None:
        filter_json = [
            item
            for item in filter_json
            if item.get("speaker_label", "") == speaker_label
        ]

    # extract words
    words = [word["alternatives"][0]["content"] for word in filter_json]

    return (
        words, phrases, phrases_idxs, turns,
        turns_idxs, text, filter_json
    )


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
                (
                    words,
                    phrases,
                    phrases_idxs,
                    turns,
                    turns_idxs,
                    text,
                    filter_json,
                ) = filter_transcribe(json_conf, measures, speaker_label=speaker_label)

                if len(filter_json) > 0 and len(text) > 0:
                    (
                        word_df,
                        phrase_df,
                        turn_df,
                        summ_df,
                    ) = cutil.process_language_feature(
                        filter_json,
                        [word_df, phrase_df, turn_df, summ_df],
                        [words, phrases, turns, text],
                        [phrases_idxs, turns_idxs],
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
        logger.error(f"Error in speech Characteristics {e}")

    finally:
        return word_df, phrase_df, turn_df, summ_df
