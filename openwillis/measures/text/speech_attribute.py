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
        if len(json_conf["segments"])>0:

            if "words" in json_conf["segments"][0]:
                return True
    return False

def filter_transcribe(json_conf, measures, min_turn_length, speaker_label=None):
    """
    ------------------------------------------------------------------------------------------------------
    This function extracts the text and filters the JSON data for Amazon Transcribe json response objects.
    Also, it filters the JSON data based on the speaker label if provided.
    Parameters:
    ...........
    json_conf: dict
        aws transcribe json response.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    min_turn_length: int
        minimum words required in each turn
    speaker_label: str
        Speaker label
    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    text_list: list
        List of transcribed text.
         split into words, turns, and full text.
    text_indices: list
        List of indices for text_list
         split into turns and unfiltered turns.
    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["results"]["items"]
    
    for i, item in enumerate(item_data): # create_index_column
        item[measures["old_index"]] = i

    # extract text
    text = " ".join([item["alternatives"][0]["content"] for item in item_data if "alternatives" in item])

    if speaker_label is not None:
        turns_idxs, turns = cutil.filter_speaker_aws(item_data, min_turn_length, speaker_label)
        turns_idxs2, _ = cutil.filter_speaker_aws(item_data, 1, speaker_label)

        text = " ".join(turns)
        
    else:
        turns_idxs, turns_idxs2, turns = [], [], []

    filter_json = cutil.filter_json_transcribe_aws(item_data, speaker_label, measures)
    words = [word["alternatives"][0]["content"] for word in filter_json]

    text_list = [words, turns, text]
    text_indices = [turns_idxs, turns_idxs2]

    return filter_json, text_list, text_indices


def filter_whisper(json_conf, measures, min_turn_length, speaker_label=None):
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
    min_turn_length: int
        minimum words required in each turn

    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    text_list: list
        List of transcribed text.
            split into words, phrases, turns, and full text.
    text_indices: list
        List of indices for turns
            split into turns and unfiltered turns.

    Raises:
    ...........
    ValueError: If the speaker label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["segments"]
    text = " ".join(item.get("text", "") for item in item_data)

    if speaker_label is not None:
        item_data = [segment for segment in item_data if "speaker" in segment]
        
    item_data = cutil.create_index_column(item_data, measures)
    if speaker_label is not None:    
        turns_idxs, turns = cutil.filter_turns(item_data, speaker_label, measures, min_turn_length)
        turns_idxs2, _ = cutil.filter_turns(item_data, speaker_label, measures, 1)
        
        text = " ".join(turns)
    else:
        turns_idxs, turns_idxs2, turns = [], [], []
    
    # filter json to only include items with start_time and end_time
    filter_json = cutil.filter_json_transcribe(item_data, speaker_label, measures)
    words = [value["word"] for value in filter_json]
    
    text_list = [words, turns, text]
    text_indices = [turns_idxs, turns_idxs2]

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

def common_summary_feature(df_summ, json_data, model, speaker_label):
    """
    ------------------------------------------------------------------------------------------------------

    Calculate file features based on JSON data.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech
    model: str
        model name
    speaker_label: str
        Speaker label

    Returns:
    ...........
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """
    try:
        if model == 'vosk':
            if len(json_data) > 0 and 'end' in json_data[-1]:

                last_dict = json_data[-1]
                df_summ['file_length'] = [last_dict['end']]

        else:
            if model == 'aws':
                json_data = json_data["results"]
                fl_length, spk_pct = cutil.calculate_file_feature(json_data, model, speaker_label)

            else:
                fl_length, spk_pct = cutil.calculate_file_feature(json_data, model, speaker_label)
            
            df_summ['file_length'] = [fl_length]
            df_summ['speaker_percentage'] = [spk_pct]# if speaker_label is not None else df_summ['speaker_percentage']
            
    except Exception as e:
        logger.error("Error in file length calculation")
    return df_summ

def process_transcript(df_list, json_conf, measures, min_turn_length, speaker_label, source, language):
    """
    ------------------------------------------------------------------------------------------------------
    
    Process transcript
    
    Parameters:
    ...........
    df_list: list, :
        contains pandas dataframe
    json_conf: dict
        Transcribed json file
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    min_turn_length: int
        minimum words required in each turn
    speaker_label: str
        Speaker label
    source: str
        model name
    language: str
        Language type
    
    Returns:
    ...........
    df_list: list
        contains pandas dataframe
    
    ------------------------------------------------------------------------------------------------------
    """
    common_summary_feature(df_list[2], json_conf, source, speaker_label)

    if source == 'whisper':
        info = filter_whisper(json_conf, measures, min_turn_length, speaker_label)
        
    elif source == 'aws':
        info = filter_transcribe(json_conf, measures, min_turn_length, speaker_label)
        
    else:
        words, text = filter_vosk(json_conf, measures)
        info = (json_conf, [words, [], text], [[], []])

    if len(info[0]) > 0 and len(info[1][-1]) > 0:
        df_list = cutil.process_language_feature(df_list, info, language, get_time_columns(source), measures)
    return df_list

def get_time_columns(source):
    """
    ------------------------------------------------------------------------------------------------------
    
    get time columns
    
    Parameters:
    ...........
    source: str
        model name
    
    Returns:
    ...........
    object: list
        time index name
        
    ------------------------------------------------------------------------------------------------------
    """
    if source == 'aws':
        return ["start_time", "end_time"]
    else:
        return ["start", "end"]

def speech_characteristics(json_conf, language="en", speaker_label=None, min_turn_length=1):
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
    min_turn_length: int
        minimum words required in each turn

    Returns:
    ...........
    df_list: list, contains:
        word_df: pandas dataframe
            A dataframe containing word summary information
        turn_df: pandas dataframe
            A dataframe containing turn summary information
        summ_df: pandas dataframe
            A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """
    try:
        # Load configuration measures
        measures = get_config(os.path.abspath(__file__), "text.json")
        df_list = cutil.create_empty_dataframes(measures)

        if bool(json_conf):
            language = language[:2].lower() if (language and len(language) >= 2) else "na"

            if language == 'en':
                cutil.download_nltk_resources()

            if is_whisper_transcribe(json_conf):
                df_list = process_transcript(df_list, json_conf, measures, min_turn_length, speaker_label, 'whisper', language)

            elif is_amazon_transcribe(json_conf):
                df_list = process_transcript(df_list, json_conf, measures, min_turn_length, speaker_label, 'aws', language)

            else:
                df_list = process_transcript(df_list, json_conf, measures, min_turn_length, speaker_label, 'vosk', language)

    except Exception as e:
        logger.error(f"Error in Speech Characteristics {e}")

    finally:
        for df in df_list:
            df.loc[0] = np.nan if df.empty else df.loc[0]

    return df_list
