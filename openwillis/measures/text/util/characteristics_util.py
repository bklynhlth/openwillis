# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def create_empty_dataframes(measures):
    """
    ------------------------------------------------------------------------------------------------------

    Creating empty measures dataframes

    Parameters:
    ...........
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

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

    word_df = pd.DataFrame(
        columns=[
            measures["word_pause"],
            measures["num_syllables"],
            measures["part_of_speech"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
        ]
    )

    phrase_df = pd.DataFrame(
        columns=[
            measures["phrase_pause"],
            measures["phrase_minutes"],
            measures["phrase_words"],
            measures["word_rate"],
            measures["syllable_rate"],
            measures["pause_rate"],
            measures["pause_var"],
            measures["pause_meandur"],
            measures["speech_percentage"],
            measures["speech_noun"],
            measures["speech_verb"],
            measures["speech_adj"],
            measures["speech_pronoun"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
            measures["speech_mattr"],
        ]
    )

    turn_df = pd.DataFrame(
        columns=[
            measures["turn_pause"],
            measures["turn_minutes"],
            measures["turn_words"],
            measures["word_rate"],
            measures["syllable_rate"],
            measures["pause_rate"],
            measures["pause_var"],
            measures["pause_meandur"],
            measures["speech_percentage"],
            measures["speech_noun"],
            measures["speech_verb"],
            measures["speech_adj"],
            measures["speech_pronoun"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
            measures["speech_mattr"],
            measures["interrupt_flag"],
        ]
    )

    summ_df = pd.DataFrame(
        columns=[
            measures["speech_minutes"],
            measures["speech_words"],
            measures["word_rate"],
            measures["syllable_rate"],
            measures["pause_rate"],
            measures["word_pause_mean"],
            measures["word_pause_var"],
            measures["phrase_pause_mean"],
            measures["phrase_pause_var"],
            measures["speech_percentage"],
            measures["speech_noun"],
            measures["speech_verb"],
            measures["speech_adj"],
            measures["speech_pronoun"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
            measures["speech_mattr"],
            measures["num_turns"],
            measures["turn_minutes_mean"],
            measures["turn_words_mean"],
            measures["turn_pause_mean"],
            measures["num_one_word_turns"],
            measures["num_interrupts"],
        ]
    )

    return word_df, phrase_df, turn_df, summ_df


def create_empty_dataframes_clinician(measures):
    """
    ------------------------------------------------------------------------------------------------------

    Creating empty measures dataframes for clinician attributes

    Parameters:
    ...........
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    prompt_df: pandas dataframe
        A dataframe containing clinician prompts information on the speech
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """

    prompt_df = pd.DataFrame(
        columns=[
            measures["prompt_id"],
            measures["prompt_pause"],
            measures["prompt_length_minutes"],
            measures["prompt_percentage"],
            measures["prompt_adherence"],
            measures["word_rate"],
            measures["syllable_rate"],
            measures["word_pause_mean"],
            measures["word_pause_var"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
        ]
    )

    summ_df = pd.DataFrame(
        columns=[
            measures["speech_minutes"],
            measures["no_prompts"],
            measures["mean_prompt_adherence"],
            measures["promps_turns_percentage"],
            measures["speech_percentage"],
            measures["word_rate"],
            measures["syllable_rate"],
            measures["word_pause_mean"],
            measures["word_pause_var"],
            measures["turn_pause_mean"],
            measures["turn_pause_var"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
        ]
    )

    return prompt_df, summ_df


def filter_speaker_phrase(item_data, speaker_label, phrases_idxs, phrases):
    """
    ------------------------------------------------------------------------------------------------------

    This function updates the phrases list
        to only include the speaker label provided.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    speaker_label: str
        Speaker label
    phrases_idxs: list
        A list of tuples containing
            the start and end indices of the phrases in the JSON object.
    phrases: list
        A list of phrases extracted from the JSON object.

    Returns:
    ...........
    phrases_idxs: list
        A list of tuples containing
            the start and end indices of the phrases in the JSON object.
    phrases: list
        A list of phrases extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    phrases_idxs2 = []
    phrases2 = []
    for i, phrase in enumerate(phrases_idxs):
        try:
            start_idx = phrase[0]
            if item_data[start_idx].get("speaker_label", "") == speaker_label:
                phrases_idxs2.append(phrase)
                phrases2.append(phrases[i])
        except Exception as e:
            logger.error(f"Error in phrase-split for speaker {speaker_label}: {e}")
            continue

    return phrases_idxs2, phrases2


def filter_speaker_turn(item_data, speaker_label, turns_idxs, turns):
    """
    ------------------------------------------------------------------------------------------------------
    
    This function updates the turns list
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

    Returns:
    ...........
    turns_idxs: list
        A list of tuples containing
            the start and end indices of the turns in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    start_idx = 0
    for i, item in enumerate(item_data):
        try:
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
        except Exception as e:
            logger.error(f"Error in turn-split for speaker {speaker_label}: {e}")
            continue

    # if the last item is the speaker label
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

    return turns_idxs, turns


def filter_speaker(item_data, speaker_label, turns_idxs, turns, phrases_idxs, phrases):
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

    Raises:
    ...........
        ValueError: If the speaker label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """

    speaker_labels = [
        item["speaker_label"] for item
        in item_data if "speaker_label" in item
    ]

    if speaker_label not in speaker_labels:
        raise ValueError(
            f"Speaker label {speaker_label} "
            "not found in the json response object."
        )

    # phrase-split for the speaker label
    phrases_idxs, phrases = filter_speaker_phrase(
        item_data, speaker_label, phrases_idxs, phrases
    )

    # turn-split for the speaker label
    turns_idxs, turns = filter_speaker_turn(
        item_data, speaker_label, turns_idxs, turns
    )

    return turns_idxs, turns, phrases_idxs, phrases


def filter_rater(item_data, rater_label):
    """
    ------------------------------------------------------------------------------------------------------

    This function updates the turns and phrases lists
        to only include the rater label provided.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    rater_label: str
        Speaker label for clinician

    Returns:
    ...........
    turns_idxs: list
        A list of tuples containing
            the start and end indices of the turns in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.

    Raises:
    ...........
        ValueError: If the rater label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """

    turns_idxs, turns = [], []

    speaker_labels = [
        item["speaker_label"] for item
        in item_data if "speaker_label" in item
    ]

    if rater_label not in speaker_labels:
        raise ValueError(
            f"Rater label {rater_label} "
            "not found in the json response object."
        )

    # turn-split for the speaker label
    turns_idxs, turns = filter_speaker_turn(
        item_data, rater_label, turns_idxs, turns
    )

    return turns_idxs, turns


def create_index_column(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates an index column in the JSON response object.

    Parameters:
    ...........
    item_data: dict
        JSON response object.

    Returns:
    ...........
    item_data: dict
        The updated JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    for i, item in enumerate(item_data):
        item[measures["old_index"]] = i
    
    return item_data


def phrase_split(text):
    """
    ------------------------------------------------------------------------------------------------------

    This function splits the input text into phrases.

    Parameters:
    ...........
    text: str
        The input text.

    Returns:
    ...........
    phrases: list
        A list of phrases extracted from the input text.
    phrases_idxs: list
        A list of tuples containing
            the start and end indices of the phrases in the input text.

    ------------------------------------------------------------------------------------------------------
    """
    phrases = nltk.tokenize.sent_tokenize(text)
    phrases_idxs = []

    start_idx = 0
    for phrase in phrases:
        end_idx = start_idx + len(phrase.split()) - 1
        phrases_idxs.append((start_idx, end_idx))
        start_idx = end_idx + 1

    return phrases, phrases_idxs


def pause_calculation(filter_json, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the pause duration between each word.

    Parameters:
    ...........
    filter_json: list
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    for i, item in enumerate(filter_json):
        if i > 0:
            item[measures["pause"]] = float(item["start_time"]) - float(
                filter_json[i - 1]["end_time"]
            )
        else:
            item[measures["pause"]] = np.nan
    
    return filter_json


def filter_json_transcribe(item_data, speaker_label, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the JSON response object to only include items with start_time and end_time.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    speaker_label: str
        Speaker label
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    filter_json = [
        item for item in item_data
        if "start_time" in item and "end_time" in item
    ]

    # calculate time difference between each word
    filter_json = pause_calculation(filter_json, measures)

    if speaker_label is not None:
        filter_json = [
            item
            for item in filter_json
            if item.get("speaker_label", "") == speaker_label
        ]

    return filter_json


def download_nltk_resources():
    """
    ------------------------------------------------------------------------------------------------------

    This function downloads the
     required NLTK resources for processing text data.

    Parameters:
    ...........
    None

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
