# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

from openwillis.measures.text.util.characteristics_util import create_empty_dataframes
from openwillis.measures.text.util.text_features import get_num_of_syllables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def process_pause_feature(df_diff, df, text_level, index_list, time_index, level_name, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech
     characteristic features at the phrase or turn
     level and adds them to the output dataframe df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information
         from the JSON response.
    df: pandas dataframe
        A dataframe containing phrase or turn summary information
    text_level: list
        List of transcribed text at the phrase or turn level.
    index_list: list
        A list containing the indices of the first and last word
         in each phrase or turn.
    time_index: list
        A list containing the names of the columns in json that contain
         the start and end times of each word.
    level_name: str
        The name of the level being analyzed (phrase or turn).
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    if level_name not in [measures["phrase"], measures["turn"]]:
        logger.error(
            f"level_name must be either {measures['phrase']} or {measures['turn']}"
        )
        return df

    for j, index in enumerate(index_list):
        try:
            rng = range(index[0], index[1] + 1)
            level_json = df_diff[df_diff[measures["old_index"]].isin(rng)]

            # remove first pause as it is the pre_pause
            pauses = level_json[measures["pause"]].values[1:]

            df.loc[j, measures[f"{level_name}_minutes"]] = (
                float(level_json.iloc[-1][time_index[1]])
                - float(level_json.iloc[0][time_index[0]])
            ) / 60
            df.loc[j, measures[f"{level_name}_words"]] = len(level_json)

            # if there is 1 pause
            if len(pauses) == 1:
                df.loc[j, measures["pause_var"]] = 0
                df.loc[j, measures["pause_meandur"]] = np.mean(pauses)
            # if there are more than 1 pauses
            elif len(pauses) > 1:
                df.loc[j, measures["pause_var"]] = np.var(pauses)
                df.loc[j, measures["pause_meandur"]] = np.mean(pauses)

            if df.loc[j, measures[f"{level_name}_minutes"]] > 0:
                df.loc[j, measures["speech_percentage"]] = 100 * (
                    1 - np.sum(pauses) / (
                        60 * df.loc[j, measures[f"{level_name}_minutes"]]
                    )
                )

                # articulation rate
                df.loc[j, measures["syllable_rate"]] = (
                    get_num_of_syllables(text_level[j]) / df.loc[j, measures[f"{level_name}_minutes"]]
                )

                df.loc[j, measures["word_rate"]] = (
                    df.loc[j, measures[f"{level_name}_words"]] / df.loc[j, measures[f"{level_name}_minutes"]]
                )
        except Exception as e:
            logger.error(f"Error in pause feature calculation for {level_name} {j}: {e}")
            continue

    df[measures["pause_rate"]] = df[measures["word_rate"]]

    return df


def process_pause_feature_prompt(df_diff, prompt_df, turn_list, turn_index, prompt_turn_index, interview_time, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech
     characteristic features at the prompt level
     and adds them to the output dataframe prompt_df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information
         from the JSON response.
    prompt_df: pandas dataframe
        A dataframe containing prompt summary information
    turn_list: list
        List of transcribed text at the turn level.
    turn_index: list
        A list containing the indices of the first and last word
         in each turn.
    prompt_turn_index: list
        A list containing the indices of the first and last word
         in each prompt.
    interview_time: float
        The duration of the interview in minutes.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    prompt_df: pandas dataframe
        The updated prompt_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    for i, index in enumerate(prompt_turn_index):
        try:

            prompt_range = range(index[0], index[1] + 1)
            # get index in turn_index
            j = turn_index.index(index)

            # get response range for each prompt
            if j != len(turn_index) - 1:
                response_end = turn_index[j+1][0]
                response_end_time = df_diff[df_diff[measures["old_index"]] == response_end].iloc[0]["start_time"]
            else:
                response_end_time = interview_time * 60

            prompt_json = df_diff[df_diff[measures["old_index"]].isin(prompt_range)]

            # remove first pause as it is the pre_pause
            pauses = prompt_json[measures["pause"]].values[1:]

            prompt_time = float(prompt_json.iloc[-1]["end_time"]) - float(prompt_json.iloc[0]["start_time"])
            prompt_response_time = float(response_end_time) - float(prompt_json.iloc[0]["start_time"])

            # total length of prompt in minutes
            prompt_df.loc[i, measures["prompt_length_minutes"]] = prompt_time / 60
            # percentage of total duration of rater prompt vs. duration of prompt and answer
            if prompt_response_time > 0:
                prompt_df.loc[i, measures["prompt_percentage"]] = 100 * prompt_time / prompt_response_time

            # if there is 1 pause
            if len(pauses) == 1:
                prompt_df.loc[i, measures["word_pause_var"]] = 0
                prompt_df.loc[i, measures["word_pause_mean"]] = np.mean(pauses)
            elif len(pauses) > 1:
                prompt_df.loc[i, measures["word_pause_var"]] = np.var(pauses)
                prompt_df.loc[i, measures["word_pause_mean"]] = np.mean(pauses)

            if prompt_time > 0:
                prompt_df.loc[i, measures["syllable_rate"]] = (
                    get_num_of_syllables(turn_list[j]) / prompt_df.loc[i, measures["prompt_length_minutes"]]
                )

                prompt_df.loc[i, measures["word_rate"]] = (
                    len(prompt_json) / prompt_df.loc[i, measures["prompt_length_minutes"]]
                )
        except Exception as e:
            logger.error(f"Error in pause feature calculation for row {i} in prompt_df: {e}")
            continue

    return prompt_df


def update_summ_df(
    df_diff, summ_df, full_text, time_index, word_df, phrase_df, turn_df, measures
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
     features at the file level and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information
         from the JSON response.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    time_index: list
        A list containing the names of the columns in json
         that contain the start and end times of each word.
    word_df: pandas dataframe
        A dataframe containing word summary information
    phrase_df: pandas dataframe
        A dataframe containing phrase summary information
    turn_df: pandas dataframe
        A dataframe containing turn summary information
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    if len(turn_df) > 0:
        speech_minutes = turn_df[measures["turn_minutes"]].sum()
    else:
        speech_minutes = (float(df_diff.iloc[-1][time_index[1]]) - float(df_diff.iloc[0][time_index[0]])) / 60
    summ_df[measures["speech_minutes"]] = [speech_minutes]

    summ_df[measures["speech_words"]] = len(df_diff)
    if speech_minutes > 0:
        summ_df[measures["word_rate"]] = (
            summ_df[measures["speech_words"]] / summ_df[measures["speech_minutes"]]
        )
        summ_df[measures["syllable_rate"]] = (
            get_num_of_syllables(full_text) / summ_df[measures["speech_minutes"]]
        )
        summ_df[measures["speech_percentage"]] = 100 * (
        1
        - df_diff.loc[1:, measures["pause"]].sum()
        / (60 * summ_df[measures["speech_minutes"]])
    )

    summ_df[measures["pause_rate"]] = summ_df[measures["word_rate"]]
    
    if len(word_df[measures["word_pause"]]) > 1:
        summ_df[measures["word_pause_mean"]] = word_df[measures["word_pause"]].mean(
            skipna=True
        )
        summ_df[measures["word_pause_var"]] = word_df[measures["word_pause"]].var(
            skipna=True
        )
    
    if len(phrase_df[measures["phrase_pause"]]) > 1:
        summ_df[measures["phrase_pause_mean"]] = phrase_df[measures["phrase_pause"]].mean(
            skipna=True
        )
        summ_df[measures["phrase_pause_var"]] = phrase_df[measures["phrase_pause"]].var(
            skipna=True
        )
    
    if len(turn_df) > 0:
        summ_df[measures["num_turns"]] = len(turn_df)
        summ_df[measures["turn_minutes_mean"]] = turn_df[
            measures["turn_minutes"]
        ].mean(skipna=True)
        summ_df[measures["turn_words_mean"]] = turn_df[
            measures["turn_words"]
        ].mean(skipna=True)
        summ_df[measures["turn_pause_mean"]] = turn_df[
            measures["turn_pause"]
        ].mean(skipna=True)
        summ_df["num_one_word_turns"] = len(
            turn_df[turn_df[measures["turn_words"]] == 1]
        )
        summ_df[measures["num_interrupts"]] = sum(turn_df[measures["interrupt_flag"]])

    return summ_df


def update_summ_df_rater(
    df_diff, summ_df, turn_df, full_text, interview_lenth, measures,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
     features at the file level and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    turn_df: pandas dataframe
        A dataframe containing turn summary information
    full_text: str
        The full transcribed text.
    interview_lenth: float
        The length of the interview in minutes.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    speech_minutes = turn_df[measures["turn_minutes"]].sum()
    summ_df[measures["speech_minutes"]] = [speech_minutes]

    no_words = len(df_diff)
    if speech_minutes > 0:
        summ_df[measures["word_rate"]] = (
            no_words / summ_df[measures["speech_minutes"]]
        )
        summ_df[measures["syllable_rate"]] = (
            get_num_of_syllables(full_text) / summ_df[measures["speech_minutes"]]
        )
    
    summ_df[measures["speech_percentage"]] = 100 * summ_df[measures["speech_minutes"]] / (
        interview_lenth
    )

    if len(df_diff[measures["pause"]]) > 1:
        summ_df[measures["word_pause_mean"]] = df_diff[measures["pause"]].mean(
            skipna=True
        )
        summ_df[measures["word_pause_var"]] = df_diff[measures["pause"]].var(
            skipna=True
        )
    
    if len(turn_df) > 1:
        summ_df[measures["turn_pause_mean"]] = turn_df[
            measures["turn_pause"]
        ].mean(skipna=True)
        summ_df[measures["turn_pause_var"]] = turn_df[
            measures["turn_pause"]
        ].var(skipna=True)

    return summ_df


def get_pause_feature_word(word_df, df_diff, word_list, phrase_index, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
        features at the word level and adds them to the output dataframe word_df.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    word_list: list
        List of transcribed text at the word level.
    phrase_index: list
        A list containing the indices of the first and last word
            in each phrase or turn.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    phrase_starts = [pindex[0] for pindex in phrase_index]

    word_df[measures["word_pause"]] = df_diff[measures["pause"]].where(
        ~df_diff[measures["old_index"]].isin(phrase_starts), np.nan
    )

    # calculate the number of syllables in each word from the word list
    word_df[measures["num_syllables"]] = [
        get_num_of_syllables(word) for word in word_list
    ]
    return word_df


def get_pause_feature_phrase(phrase_df, df_diff, phrase_list, phrase_index, turn_index, time_index, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
        features at the phrase level and adds them to the output dataframe phrase_df.

    Parameters:
    ...........
    phrase_df: pandas dataframe
        A dataframe containing phrase summary information
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    phrase_list: list
        List of transcribed text at the phrase level.
    phrase_index: list
        A list containing the indices of the first and last word
            in each phrase
    turn_index: list
        A list containing the indices of the first and last word
            in each turn.
    time_index: list
        A list containing the names of the columns in json that contain
            the start and end times of each word.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    phrase_df: pandas dataframe
        The updated phrase_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    phrase_starts = [pindex[0] for pindex in phrase_index]

    df_diff_phrase = df_diff[
        df_diff[measures["old_index"]].isin(phrase_starts)
    ]  # get the rows corresponding to the start of each phrase

    if len(turn_index) > 0:
        turn_starts = [
            uindex[0] for uindex in turn_index
        ]  # get the start index of each turn
        phrase_df[measures["phrase_pause"]] = df_diff_phrase[measures["pause"]].where(
            ~df_diff_phrase[measures["old_index"]].isin(turn_starts), np.nan
        )
    else:
        phrase_df[measures["phrase_pause"]] = df_diff_phrase[measures["pause"]]

    phrase_df = phrase_df.reset_index(drop=True)

    phrase_df = process_pause_feature(
        df_diff, phrase_df, phrase_list, phrase_index, time_index, measures["phrase"], measures
    )

    return phrase_df


def get_pause_feature_turn(turn_df, df_diff, turn_list, turn_index, time_index, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
        features at the turn level and adds them to the output dataframe turn_df.

    Parameters:
    ...........
    turn_df: pandas dataframe
        A dataframe containing turn summary information
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    turn_list: list
        List of transcribed text at the turn level.
    turn_index: list
        A list containing the indices of the first and last word
            in each turn.
    time_index: list
        A list containing the names of the columns in json that contain
            the start and end times of each word.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    turn_starts = [uindex[0] for uindex in turn_index]

    # get the rows corresponding to the start of each turn
    df_diff_turn = df_diff[
        df_diff[measures["old_index"]].isin(turn_starts)
    ]

    turn_df[measures["turn_pause"]] = df_diff_turn[measures["pause"]]
    turn_df[measures["interrupt_flag"]] = False
    # set pre_turn_pause to 0 if negative (due to overlapping turns)
    # and set interrupt_flag to True
    negative_pause = turn_df[measures["turn_pause"]] < 0
    turn_df.loc[negative_pause, measures["turn_pause"]] = 0
    turn_df.loc[negative_pause, measures["interrupt_flag"]] = True

    turn_df = turn_df.reset_index(drop=True)

    turn_df = process_pause_feature(
        df_diff, turn_df, turn_list, turn_index, time_index, measures["turn"], measures
    )

    return turn_df


def get_pause_feature(json_conf, df_list, text_list, text_indices, time_index, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related
     speech characteristic features

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes.
            word_df, phrase_df, turn_df, summ_df
    text_list: list
        List of transcribed text.
            split into words, phrases, turns, and full text.
    text_indices: list
        List of indices for text_list.
            for phrases and turns.
    time_index: list
        A list containing the names of the columns
         in json that contain the start and end times of each word.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_feature: list
        List of updated pandas dataframes.
            word_df, phrase_df, turn_df, summ_df

    ------------------------------------------------------------------------------------------------------
    """
    # Check if json_conf is empty
    if len(json_conf) <= 0:
        return df_list

    word_df, phrase_df, turn_df, summ_df = df_list
    word_list, phrase_list, turn_list, full_text = text_list
    phrase_index, turn_index = text_indices

    # Convert json_conf to a pandas DataFrame
    df_diff = pd.DataFrame(json_conf)

    # Calculate the pause time between
    # each word and add the results to pause_list
    if measures["pause"] not in df_diff.columns:
        df_diff[measures["pause"]] = df_diff[time_index[0]].astype(float) - df_diff[
            time_index[1]
        ].astype(float).shift(1)

    # word-level analysis
    word_df = get_pause_feature_word(word_df, df_diff, word_list, phrase_index, measures)

    # phrase-level analysis
    phrase_df = get_pause_feature_phrase(
        phrase_df, df_diff, phrase_list, phrase_index, turn_index, time_index, measures
    )

    # turn-level analysis
    if len(turn_index) > 0:
        turn_df = get_pause_feature_turn(
            turn_df, df_diff, turn_list, turn_index, time_index, measures
        )

    # file-level analysis
    summ_df = update_summ_df(
        df_diff, summ_df, full_text, time_index, word_df, phrase_df, turn_df, measures
    )

    df_feature = [word_df, phrase_df, turn_df, summ_df]

    return df_feature


def get_pause_feature_prompt(prompt_df, df_diff, turn_list, turn_index, prompt_turn_index, interview_time, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
        features at the prompt level and adds them to the output dataframe prompt_df.

    Parameters:
    ...........
    prompt_df: pandas dataframe
        A dataframe containing prompt summary information
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    turn_list: list
        List of transcribed text at the turn level.
    turn_index: list
        A list containing the indices of the first and last word
            in each turn.
    prompt_turn_index: list
        A list containing the indices of the first and last word
            in each prompt.
    interview_time: float
        The length of the interview in minutes.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    
    Returns:
    ...........
    prompt_df: pandas dataframe
        The updated prompt_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    turn_starts = [uindex[0] for uindex in prompt_turn_index]

    # get the rows corresponding to the start of each turn
    turn_starts_df = pd.DataFrame({measures["old_index"]: turn_starts})
    df_diff_prompt = df_diff.merge(turn_starts_df, on=measures["old_index"], how='right')

    prompt_df.loc[:, measures["prompt_pause"]] = df_diff_prompt[measures["pause"]].reset_index(drop=True)

    prompt_df = process_pause_feature_prompt(
        df_diff, prompt_df, turn_list, turn_index, prompt_turn_index, interview_time, measures
    )

    return prompt_df


def get_pause_rater(json_conf, df_list, text_list, text_indices, interview_time, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic features

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes.
            prompt_df, summ_df
    text_list: list
        List of transcribed text.
            split into turns, full text and prompt-adherent turns.
    text_indices: list
        List of indices for text_list.
            for turns and prompt-adherent turns.
    interview_time: float
        The length of the interview in minutes.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_feature: list
        List of updated pandas dataframes.
            prompt_df, summ_df

    ------------------------------------------------------------------------------------------------------
    """
    # Check if json_conf is empty
    if len(json_conf) <= 0:
        return df_list

    prompt_df, summ_df = df_list
    turn_list, full_text, _ = text_list
    turn_index, prompt_turn_index = text_indices

    # Convert json_conf to a pandas DataFrame
    df_diff = pd.DataFrame(json_conf)

    # turn_df pause calculations - to use for summ_df    
    if len(turn_index) > 0:
        turn_df = create_empty_dataframes(measures)[2]
        turn_df = get_pause_feature_turn(
            turn_df, df_diff, turn_list, turn_index, ["start_time", "end_time"], measures
        )

    # prompt-level analysis
    if len(prompt_turn_index) > 0:
        prompt_df = get_pause_feature_prompt(
            prompt_df, df_diff, turn_list, turn_index, prompt_turn_index, interview_time, measures
        )

    # file-level analysis
    summ_df = update_summ_df_rater(
        df_diff, summ_df, turn_df, full_text, interview_time, measures
    )

    df_feature = [prompt_df, summ_df]

    return df_feature
