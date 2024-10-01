# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import logging

import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def get_num_of_syllables(text):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the number of syllables in the input text.

    Parameters:
    ...........
    text: str
        The input text.

    Returns:
    ...........
    syllable_count: int
        The number of syllables in the input text.

    ---------------------------------------------------------------------------------------
    """

    syllable_tokenizer = nltk.tokenize.SyllableTokenizer()
    punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~" # remove punctuation
    
    syllables = [syllable_tokenizer.tokenize(token) for token in nltk.word_tokenize(text) if token not in punctuation]
    syllable_count = sum([len(token) for token in syllables])

    return syllable_count

def get_pause_feature_word(word_df, df_diff, word_list, turn_index, measures):
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
    turn_index: list
        A list containing the indices of the first and last word
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    turn_starts = [pindex[0] for pindex in turn_index]
    word_df[measures["word_pause"]] = df_diff[measures["pause"]].where(~df_diff[measures["old_index"]].isin(turn_starts), np.nan)
    
    word_df[measures["num_syllables"]] = [get_num_of_syllables(word) for word in word_list]
    return word_df

def process_pause_feature(df_diff, df, text_level, index_list, time_index, level_name, measures, language):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech
     characteristic features at the turn
     level and adds them to the output dataframe df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information from the JSON response.
    df: pandas dataframe
        A dataframe containing turn summary information
    text_level: list
        List of transcribed text at the turn level.
    index_list: list
        A list containing the indices of the first and last word in each turn.
    time_index: list
        A list containing the names of the columns in json that contain
         the start and end times of each word.
    level_name: str
        The name of the level being analyzed turn.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    if level_name not in [measures["turn"]]:
        logger.error(f"level_name must be turn")
        return df

    for j, index in enumerate(index_list):
        try:
            
            rng = range(index[0], index[1] + 1)
            level_json = df_diff[df_diff[measures["old_index"]].isin(rng)]
            
            pauses = level_json[measures["pause"]].values[1:]
            level_min_val = (float(level_json.iloc[-1][time_index[1]]) - float(level_json.iloc[0][time_index[0]])) / 60
            
            df.loc[j, measures[f"{level_name}_minutes"]] = level_min_val
            df.loc[j, measures[f"{level_name}_words"]] = len(level_json)

            if len(pauses) == 1:
                df.loc[j, measures["pause_var"]] = 0
                df.loc[j, measures["pause_meandur"]] = np.mean(pauses)

            elif len(pauses) > 1:
                df.loc[j, measures["pause_var"]] = np.var(pauses)
                df.loc[j, measures["pause_meandur"]] = np.mean(pauses)

            if df.loc[j, measures[f"{level_name}_minutes"]] > 0:
                speech_pct_val = 100 * (1 - np.sum(pauses) / (60 * df.loc[j, measures[f"{level_name}_minutes"]]))
                df.loc[j, measures["speech_percentage"]] = speech_pct_val

                if language in measures["english_langs"]:
                    syllable_rate = (get_num_of_syllables(text_level[j]) / df.loc[j, measures[f"{level_name}_minutes"]])
                    df.loc[j, measures["syllable_rate"]] = syllable_rate
                
                word_rate_val = (df.loc[j, measures[f"{level_name}_words"]] / df.loc[j, measures[f"{level_name}_minutes"]])
                df.loc[j, measures["word_rate"]] = word_rate_val
                
        except Exception as e:
            logger.error(f"Error in pause feature calculation for {level_name} {j}: {e}")
            continue

    return df

def get_pause_feature_turn(turn_df, df_diff, turn_list, turn_index, time_index, measures, language):
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
    df_diff_turn = df_diff[df_diff[measures["old_index"]].isin(turn_starts)]

    turn_df[measures["turn_pause"]] = df_diff_turn[measures["pause"]]
    turn_df[measures["interrupt_flag"]] = False
    
    negative_pause = turn_df[measures["turn_pause"]] <= 0
    turn_df.loc[negative_pause, measures["turn_pause"]] = 0
    
    turn_df.loc[negative_pause, measures["interrupt_flag"]] = True
    turn_df = turn_df.reset_index(drop=True)

    turn_df = process_pause_feature(df_diff, turn_df, turn_list, turn_index, time_index, measures["turn"], measures, language)
    return turn_df

def update_summ_df(df_diff, summ_df, full_text, time_index, word_df, turn_df, measures):
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
        speech_words = turn_df[measures["turn_words"]].sum()
    else:
        speech_minutes = (float(df_diff.iloc[-1][time_index[1]]) - float(df_diff.iloc[0][time_index[0]])) / 60
        speech_words = len(df_diff)

    summ_df[measures["speech_minutes"]] = [speech_minutes]    
    summ_df[measures["speech_words"]] = [speech_words]

    if speech_minutes > 0:
        
        summ_df[measures["word_rate"]] = (summ_df[measures["speech_words"]] / summ_df[measures["speech_minutes"]])
        summ_df[measures["syllable_rate"]] = (get_num_of_syllables(full_text) / summ_df[measures["speech_minutes"]])
        summ_df[measures["speech_percentage"]] = 100 * (summ_df[measures["speech_minutes"]] / summ_df[measures["file_length"]])

    if len(word_df[measures["word_pause"]]) > 1:
        summ_df[measures["word_pause_mean"]] = word_df[measures["word_pause"]].mean(skipna=True)
        summ_df[measures["word_pause_var"]] = word_df[measures["word_pause"]].var(skipna=True)
    
    if len(turn_df) > 0:
        summ_df[measures["num_turns"]] = len(turn_df)
        summ_df[measures["turn_minutes_mean"]] = turn_df[measures["turn_minutes"]].mean(skipna=True)
        
        summ_df[measures["turn_words_mean"]] = turn_df[measures["turn_words"]].mean(skipna=True)
        summ_df[measures["turn_pause_mean"]] = turn_df[measures["turn_pause"]].mean(skipna=True)
        
        summ_df[measures["num_one_word_turns"]] = len(turn_df[turn_df[measures["turn_words"]] == 1])
        summ_df[measures["num_interrupts"]] = len(turn_df[turn_df[measures["interrupt_flag"]]==True])

    return summ_df

def get_pause_feature(json_conf, df_list, text_list, turn_index, measures, time_index, language):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related
     speech characteristic features

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes: word_df, turn_df, summ_df
    text_list: list
        List of transcribed text: split into words, turns, and full text.
    turn_index: list
        List of indices for text_list.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    time_index: list
        timepoint index (start/end)
    language: str
        Language of the transcribed text.

    Returns:
    ...........
    df_feature: list
        List of updated pandas dataframes (word_df, turn_df and summ_df)

    ------------------------------------------------------------------------------------------------------
    """
    try:
        if len(json_conf) <= 0:
            return df_list

        word_df, turn_df, summ_df = df_list
        word_list, turn_list, full_text = text_list
        df_diff = pd.DataFrame(json_conf)

        # Calculate the pause time between; each word and add the results to pause_list
        if measures["pause"] not in df_diff.columns:
            df_diff[measures["pause"]] = df_diff[time_index[0]].astype(float) - df_diff[time_index[1]].astype(float).shift(1)

        # word-level analysis
        word_df = get_pause_feature_word(word_df, df_diff, word_list, turn_index, measures)

        # turn-level analysis
        if len(turn_index) > 0:
            turn_df = get_pause_feature_turn(turn_df, df_diff, turn_list, turn_index, time_index, measures, language)

        # file-level analysis
        summ_df = update_summ_df(df_diff, summ_df, full_text, time_index, word_df, turn_df, measures)
        df_feature = [word_df, turn_df, summ_df]
        return df_feature
    except Exception as e:
        logger.error(f"Error in pause feature calculation: {e}")
        return df_list