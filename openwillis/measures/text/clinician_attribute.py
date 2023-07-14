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

    file = open(measure_path, encoding="utf8")
    measures = json.load(file)
    return measures


def get_prompts(interview_type, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function returns the prompts for the specified interview type.

    Parameters:
    ...........
    interview_type: str
        Interview type (can be 'panss' or 'madrs')
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    prompts: list
        A list of prompts for the specified interview type.
    prompt_indices: list
        A list of indices for the prompts.

    Raises:
    ...........
    ValueError: If the interview type is not 'panss' or 'madrs'.

    ------------------------------------------------------------------------------------------------------
    """

    # force interview_type to be lowercase
    interview_type = interview_type.lower()

    if interview_type not in ['madrs', 'panss']:
        raise ValueError("Invalid interview type. Must be 'madrs' or 'panss'.")
    else:
        prompts_dict = get_config(os.path.abspath(__file__), f"{interview_type}.json")

    # extract prompt indices    
    prompt_indices = [str(i) for i in measures[f"{interview_type}_prompt_ids"]]

    # select prompts from prompts_dict
    prompts = [prompts_dict[prompt]["text"] for prompt in prompt_indices]

    return prompts, prompt_indices


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


def filter_transcribe(json_conf, measures, rater_label):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text and filters the JSON data
     for Amazon Transcribe json response objects.
     Also, it filters the JSON data based on the rater label if provided.

    Parameters:
    ...........
    json_conf: dict
        aws transcribe json response.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    rater_label: str
        Speaker label for clinician

    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    text_list: list
        List of transcribed text.
         split into words, phrases, turns, and full text.
    turn_indices: list
        indices for turns/prompts.
    interview_time: float
        Interview time in minutes.

    Raises:
    ...........
    ValueError: If the rater label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["results"]["items"]

    # make a dictionary to map old indices to new indices
    item_data = cutil.create_index_column(item_data, measures)

    # get interview time
    item_data_df = pd.DataFrame(item_data)
    interview_time = (
        item_data_df["end_time"].astype(float).max(skipna=True)
        - item_data_df["start_time"].astype(float).min(skipna=True)
    ) / 60
    
    turn_indices, turns = cutil.filter_rater(item_data, rater_label)

    # entire transcript - by joining all the phrases
    text = " ".join(turns)

    # filter json to only include items with start_time and end_time
    filter_json = cutil.filter_json_transcribe(item_data, rater_label, measures)

    text_list = [turns, text]

    return filter_json, text_list, turn_indices, interview_time


def clinician_characteristics(json_conf, language="en-us", rater_label = 'clinician', interview_type = 'panss'):
    """
    ------------------------------------------------------------------------------------------------------

    Clinician Characteristics

    Parameters:
    ...........
    json_conf: dict
        Transcribed json file
    language: str
        Language type
    rater_label: str
        Speaker label for clinician
    interview_type: str
        Interview type (can be 'panss' or 'madrs')

    Returns:
    ...........
    df_list: list, contains:
        prompts_df: pandas dataframe
            A dataframe containing clinician prompts information on the speech
        summ_df: pandas dataframe
            A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """
    measures = get_config(os.path.abspath(__file__), "text.json")
    df_list = cutil.create_empty_dataframes_clinician(measures)

    try:
        prompts, prompt_ids = get_prompts(interview_type, measures)
        # add prompt ids to prompts_df
        df_list[0][measures["prompt_id"]] = prompt_ids
        
        if bool(json_conf):
            cutil.download_nltk_resources()

            if is_amazon_transcribe(json_conf):
                filter_json, text_list, turn_indices, interview_time = filter_transcribe(
                    json_conf, measures, rater_label
                )

                # add prompts to text_list
                text_list.append(prompts)
                # create text indices for turns/prompts
                text_indices = [turn_indices.copy()]

                if len(filter_json) > 0 and len(text_list[-1]) > 0:
                    df_list = cutil.process_rater_feature(
                        filter_json, df_list, text_list,
                        text_indices, language, interview_time,
                        measures,
                    )
            else:
                raise ValueError("Invalid JSON response object; not from Amazon Transcribe")
            
            # if prompt_df is empty, then add a row of NaNs
            if df_list[0].empty:
                df_list[0].loc[0] = np.nan
            # if summ_df is empty, then add a row of NaNs
            if df_list[1].empty:
                df_list[1].loc[0] = np.nan
    except Exception as e:
        logger.error(f"Error in Clinician Characteristics {e}")

    finally:
        return df_list
