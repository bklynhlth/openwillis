# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import re
import string
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# NLTK Tag list
TAG_DICT = {"PRP": "Pronoun", "PRP$": "Pronoun", "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", 
            "VBZ": "Verb", "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective", "NN": "Noun", "NNP": "Noun", "NNS": "Noun",
            "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb", "DT": "Determiner"}
FIRST_PERSON_PRONOUNS = ["I", "me", "my", "mine", "myself"]

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
    tuple: pandas dataframe
        An empty dataframe for word, turn and summary measures

    ------------------------------------------------------------------------------------------------------
    """

    word_df = pd.DataFrame(columns=[measures["word_pause"], measures["num_syllables"], measures["part_of_speech"],
                                    measures["first_person"], measures["verb_tense"],
                                    measures["word_coherence"], measures["word_coherence_5"], measures["word_coherence_10"],
                                    measures["word_coherence_variability_2"], measures["word_coherence_variability_3"],
                                    measures["word_coherence_variability_4"], measures["word_coherence_variability_5"],
                                    measures["word_coherence_variability_6"], measures["word_coherence_variability_7"],
                                    measures["word_coherence_variability_8"], measures["word_coherence_variability_9"],
                                    measures["word_coherence_variability_10"]
                                    ])
    turn_df = pd.DataFrame(columns=[measures["turn_pause"], measures["turn_minutes"], measures["turn_words"], 
                                    measures["word_rate"], measures["syllable_rate"], measures["speech_percentage"], 
                                    measures["pause_meandur"], measures["pause_var"], measures["pos"], measures["neg"], 
                                    measures["neu"], measures["compound"], measures["speech_mattr"],
                                    measures["first_person_percentage"], measures["first_person_sentiment_positive"], measures["first_person_sentiment_negative"],
                                    measures["word_repeat_percentage"], measures["phrase_repeat_percentage"],
                                    measures["sentence_tangeniality1"], measures["sentence_tangeniality2"],
                                    measures["turn_to_turn_tangeniality"], measures["perplexity"],
                                    measures["interrupt_flag"]])

    summ_df = pd.DataFrame(
        columns=[measures["file_length"], measures["speech_minutes"], measures["speech_words"], measures["word_rate"],
                measures["syllable_rate"], measures["word_pause_mean"], measures["word_pause_var"], 
                measures["speech_percentage"], measures["pos"], measures["neg"], measures["neu"], measures["compound"], 
                measures["speech_mattr"], measures["first_person_percentage"], measures["first_person_sentiment_positive"],
                measures["first_person_sentiment_negative"], measures["first_person_sentiment_overall"],
                measures["word_repeat_percentage"], measures["phrase_repeat_percentage"],
                measures["word_coherence_mean"], measures["word_coherence_var"],
                measures["word_coherence_5_mean"], measures["word_coherence_5_var"],
                measures["word_coherence_10_mean"], measures["word_coherence_10_var"],
                measures["word_coherence_variability_2_mean"], measures["word_coherence_variability_2_var"],
                measures["word_coherence_variability_3_mean"], measures["word_coherence_variability_3_var"],
                measures["word_coherence_variability_4_mean"], measures["word_coherence_variability_4_var"],
                measures["word_coherence_variability_5_mean"], measures["word_coherence_variability_5_var"],
                measures["word_coherence_variability_6_mean"], measures["word_coherence_variability_6_var"],
                measures["word_coherence_variability_7_mean"], measures["word_coherence_variability_7_var"],
                measures["word_coherence_variability_8_mean"], measures["word_coherence_variability_8_var"],
                measures["word_coherence_variability_9_mean"], measures["word_coherence_variability_9_var"],
                measures["word_coherence_variability_10_mean"], measures["word_coherence_variability_10_var"],
                measures["num_turns"], measures["num_one_word_turns"], measures["turn_minutes_mean"],
                measures["turn_words_mean"], measures["turn_pause_mean"], measures["speaker_percentage"], 
                measures["sentence_tangeniality1_mean"], measures["sentence_tangeniality1_var"],
                measures["sentence_tangeniality2_mean"], measures["sentence_tangeniality2_var"],
                measures["turn_to_turn_tangeniality_mean"], measures["turn_to_turn_tangeniality_var"],
                measures["turn_to_turn_tangeniality_slope"], measures["perplexity_mean"], measures["perplexity_var"],
                measures["num_interrupts"]])

    return word_df, turn_df, summ_df

def create_index_column(item_data, measures):
    """
    This function creates an index column in the JSON response object.

    Parameters:
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    item_data: dict
        The updated JSON response object.
    """
    index = 0
    for item in item_data:
        
        for word in item.get("words", []):
            word[measures["old_index"]] = index
            index += 1

    return item_data

def download_nltk_resources():
    """
    ------------------------------------------------------------------------------------------------------

    This function downloads the
     required NLTK resources for processing text data.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
        
def create_turns_aws(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates a dataframe of turns from the JSON response object for AWS.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    utterances: pandas dataframe
        A dataframe containing the turns extracted from the JSON object,
            along with word, phrase and utterance indices and texts.

    ------------------------------------------------------------------------------------------------------
    """

    utterances, current_utterance, utterance_texts = [], [], []
    current_words, words_texts = [], []
    current_speaker = None
    utterance_id = 0

    for item in item_data:
        # Check if the item is a continuation of the current speaker's turn
        if item['speaker_label'] == current_speaker:
            current_utterance.append(utterance_id)
            utterance_texts.append(item['alternatives'][0]['content'])
            if 'start_time' in item and 'end_time' in item:
                current_words.append(utterance_id)
                words_texts.append(item['alternatives'][0]['content'])
        else:
            # If not, save the current utterance (if any) and start a new one
            if current_utterance:
                # split utterance into phrases
                phrases = nltk.tokenize.sent_tokenize(' '.join(utterance_texts))
                phrases_idxs = []

                start_idx = current_utterance[0]
                for phrase in phrases:
                    end_idx = start_idx + len(phrase.split()) - 1
                    phrases_idxs.append((start_idx, end_idx))
                    start_idx = end_idx + 1

                utterances.append({
                    measures['utterance_ids']: (current_utterance[0], current_utterance[-1]),
                    measures['utterance_text']: ' '.join(utterance_texts),
                    measures['phrases_ids']: phrases_idxs,
                    measures['phrases_texts']: phrases.copy(),
                    measures['words_ids']: current_words.copy(),
                    measures['words_texts']: words_texts.copy(),
                    measures['speaker_label']: current_speaker,
                })
                current_utterance.clear()
                utterance_texts.clear()
                current_words.clear()
                words_texts.clear()
            
            current_speaker = item['speaker_label']
            current_utterance.append(utterance_id)
            utterance_texts.append(item['alternatives'][0]['content'])
            if 'start_time' in item and 'end_time' in item:
                current_words.append(utterance_id)
                words_texts.append(item['alternatives'][0]['content'])
        
        utterance_id += 1

    # Don't forget to add the last utterance if the loop ends
    if current_utterance:
        phrases = nltk.tokenize.sent_tokenize(' '.join(utterance_texts))
        phrases_idxs = []

        start_idx = current_utterance[0]
        for phrase in phrases:
            end_idx = start_idx + len(phrase.split()) - 1
            phrases_idxs.append((start_idx, end_idx))
            start_idx = end_idx + 1

        utterances.append({
            measures['utterance_ids']: (current_utterance[0], current_utterance[-1]),
            measures['utterance_text']: ' '.join(utterance_texts),
            measures['phrases_ids']: phrases_idxs,
            measures['phrases_texts']: phrases.copy(),
            measures['words_ids']: current_words.copy(),
            measures['words_texts']: words_texts.copy(),
            measures['speaker_label']: current_speaker,
        })

    return pd.DataFrame(utterances)

def filter_json_transcribe_aws(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the JSON response object to only include items with start_time and end_time.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    filter_json = [item for item in item_data if "start_time" in item and "end_time" in item]
    filter_json = pause_calculation(filter_json, measures, ['start_time', 'end_time'])

    return filter_json

def create_turns_whisper(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates a dataframe of turns from the JSON response object for Whisper.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    utterances: pandas dataframe
        A dataframe containing the turns extracted from the JSON object,
            along with word, phrase and utterance indices and texts.

    ------------------------------------------------------------------------------------------------------
    """

    data = []
    current_speaker = None
    aggregated_text = ""
    aggregated_ids = []
    word_ids, word_texts = [], []
    phrase_ids, phrase_texts = [], []

    for item in item_data:
        if item['speaker'] == current_speaker:
            idxs = [word[measures["old_index"]] for word in item['words']]
            # Continue aggregating text and ids for the current speaker
            aggregated_text += " " + item['text']
            aggregated_ids.extend(idxs)

            word_ids.extend(idxs)
            word_texts.extend([word['word'] for word in item['words']])

            phrase_ids.append((idxs[0], idxs[-1]))
            phrase_texts.append(item['text'])

        else:
            # If the speaker changes, save the current aggregation (if it exists) and start new aggregation
            if aggregated_ids:  # Check to ensure it's not the first item
                data.append({
                    measures['utterance_ids']: (aggregated_ids[0], aggregated_ids[-1]),
                    measures['utterance_text']: aggregated_text.strip(),
                    measures['phrases_ids']: phrase_ids,
                    measures['phrases_texts']: phrase_texts,
                    measures['words_ids']: word_ids,
                    measures['words_texts']: word_texts,
                    measures['speaker_label']: current_speaker
                })
            
            # Reset aggregation for the new speaker
            current_speaker = item['speaker']
            aggregated_text = item['text']
            aggregated_ids = [word[measures["old_index"]] for word in item['words']]

            word_ids = [word[measures["old_index"]] for word in item['words']]
            word_texts = [word['word'] for word in item['words']]

            phrase_ids = [(word_ids[0], word_ids[-1])]
            phrase_texts = [item['text']]
    
    # Don't forget to add the last aggregated utterance
    if aggregated_ids:
        data.append({
            measures['utterance_ids']: (aggregated_ids[0], aggregated_ids[-1]),
            measures['utterance_text']: aggregated_text.strip(),
            measures['phrases_ids']: phrase_ids,
            measures['phrases_texts']: phrase_texts,
            measures['words_ids']: word_ids,
            measures['words_texts']: word_texts,
            measures['speaker_label']: current_speaker
        })

    return pd.DataFrame(data)

def pause_calculation(filter_json, measures, time_index):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the pause duration between each item.

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
            item[measures["pause"]] = float(item[time_index[0]]) - float(filter_json[i - 1][time_index[1]])
        
        else:
            item[measures["pause"]] = np.nan
    return filter_json

def filter_json_transcribe(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the JSON response object to only include items with start and end time.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data2 = []
    for item in item_data:
        try:
            if "speaker" in item:
                speaker = item["speaker"]
            else:
                speaker = ""
            words = item["words"]
            
            for j, w in enumerate(words):# update speaker labels
                words[j]["speaker"] = speaker
            
            item_data2 += words
        except Exception as e:
            logger.error(f"Failed to filter word: {e}")
    
    filter_json = [item for item in item_data2 if "start" in item and "end" in item]
    filter_json = pause_calculation(filter_json, measures, ['start', 'end'])

    return filter_json

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

def get_mattr(text):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR)
     of the input text using the
     LexicalRichness library.

    Parameters:
    ...........
    text : str
        The input text to be analyzed.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
    """
    word = nltk.word_tokenize(text)
    filter_punc = list(value for value in word if value not in [".", "!", "?"])
    filter_punc = " ".join(filter_punc)
    mattr = np.nan

    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=lex_richness.words)

    return mattr

def get_tag(word_df, word_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech
     tagging on the input text using NLTK, and returns
     word-level part-of-speech tags.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    word_list: list
        List of transcribed text at the word level.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    tag_list = nltk.pos_tag(word_list)
    
    tag_list_pos = [TAG_DICT[tag[1]] if tag[1] in TAG_DICT.keys() else "Other" for tag in tag_list]
    word_df[measures["part_of_speech"]] = tag_list_pos

    word_df[measures["first_person"]] = [word in FIRST_PERSON_PRONOUNS for word in word_list]
    # make non pronouns None
    word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]] == "Pronoun", None)

    present_tense = ["VBP", "VBZ"]
    past_tense = ["VBD", "VBN"]
    tag_list_verb = ["Present" if tag[1] in present_tense else "Past" if tag[1] in past_tense else "Other" for tag in tag_list]
    word_df[measures["verb_tense"]] = tag_list_verb
    # make non verbs None
    word_df[measures["verb_tense"]] = word_df[measures["verb_tense"]].where(word_df[measures["part_of_speech"]] == "Verb", None)

    return word_df

def calculate_first_person_sentiment(df, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates a measure of the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    res1: list
        A list containing the calculated measure of the influence of positive sentiment on the use of first person pronouns.
    res2: list
        A list containing the calculated measure of the influence of negative sentiment on the use of first person pronouns.

    ------------------------------------------------------------------------------------------------------
    """
    
    res1 = []
    res2 = []
    for i in range(len(df)):
        perc = df.loc[i, measures["first_person_percentage"]]
        pos = df.loc[i, measures["pos"]]
        neg = df.loc[i, measures["neg"]]

        if perc is np.nan:
            res1.append(np.nan)
            res2.append(np.nan)
            continue

        res1.append((100-perc)*pos)
        res2.append(perc*neg)

    return res1, res2

def calculate_first_person_percentage(text):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the percentage of first person pronouns in the input text.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.

    Returns:
    ...........
    float
        The calculated percentage of first person pronouns in the input text.

    ------------------------------------------------------------------------------------------------------
    """

    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    # filter out non pronouns
    pronouns = [tag[0] for tag in tags if tag[1] == "PRP" or tag[1] == "PRP$"]
    if len(pronouns) == 0:
        return np.nan

    first_person_pronouns = len([word for word in pronouns if word in FIRST_PERSON_PRONOUNS])
    return (first_person_pronouns / len(pronouns)) * 100

def get_first_person_turn(turn_df, turn_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in each turn.
     Specifically, it calculates the percentage of first person pronouns in each turn,
     and the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    turn_list: list
        List of transcribed text at the turn level.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    first_person_percentages = [calculate_first_person_percentage(turn) for turn in turn_list]

    turn_df[measures["first_person_percentage"]] = first_person_percentages

    first_pos, first_neg = calculate_first_person_sentiment(turn_df, measures)

    turn_df[measures["first_person_sentiment_positive"]] = first_pos
    turn_df[measures["first_person_sentiment_negative"]] = first_neg

    return turn_df

def get_first_person_summ(summ_df, turn_df, full_text, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in the transcript.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing summary information.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    full_text: str
        The full transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    summ_df[measures["first_person_percentage"]] = calculate_first_person_percentage(full_text)

    if len(turn_df) > 0:
        summ_df[measures["first_person_sentiment_positive"]] = turn_df[measures["first_person_sentiment_positive"]].mean(skipna=True)
        summ_df[measures["first_person_sentiment_negative"]] = turn_df[measures["first_person_sentiment_negative"]].mean(skipna=True)

        first_person_sentiment = []
        for i in range(len(turn_df)):
            if turn_df.loc[i, measures["pos"]] > turn_df.loc[i, measures["neg"]]:
                first_person_sentiment.append(turn_df.loc[i, measures["first_person_sentiment_positive"]])
            else:    
                first_person_sentiment.append(turn_df.loc[i, measures["first_person_sentiment_negative"]])

        summ_df[measures["first_person_sentiment_overall"]] = np.nanmean(first_person_sentiment)
    else:
        first_pos, first_neg = calculate_first_person_sentiment(summ_df, measures)
        summ_df[measures["first_person_sentiment_positive"]] = first_pos
        summ_df[measures["first_person_sentiment_negative"]] = first_neg

        if summ_df[measures["pos"]].values[0] > summ_df[measures["neg"]].values[0]:
            summ_df[measures["first_person_sentiment_overall"]] = summ_df[measures["first_person_sentiment_positive"]].values[0]
        else:
            summ_df[measures["first_person_sentiment_overall"]] = summ_df[measures["first_person_sentiment_negative"]].values[0]

    return summ_df

def get_pos_tag(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the part-of-speech measures
        and adds them to the output dataframes

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, turn_df, summ_df = df_list
    word_list, turn_list, full_text = text_list

    word_df = get_tag(word_df, word_list, measures)

    if len(turn_list) > 0:
        turn_df = get_first_person_turn(turn_df, turn_list, measures)

    summ_df = get_first_person_summ(summ_df, turn_df, full_text, measures)

    df_list = [word_df, turn_df, summ_df]
    return df_list

def get_sentiment(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, turn_df, summ_df = df_list
    word_list, turn_list, full_text = text_list

    sentiment = SentimentIntensityAnalyzer()
    cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"], measures["speech_mattr"]]

    for idx, u in enumerate(turn_list):
        try:
            
            sentiment_dict = sentiment.polarity_scores(u)
            mattr = get_mattr(u)
            turn_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            continue
            
    sentiment_dict = sentiment.polarity_scores(full_text)
    mattr = get_mattr(full_text)

    summ_df.loc[0, cols] = list(sentiment_dict.values()) + [mattr]
    df_list = [word_df, turn_df, summ_df]
    return df_list

def calculate_repetitions(words_texts, phrases_texts):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the percentage of repeated words and phrases in the input lists.

    Parameters:
    ...........
    words_texts: list
        List of transcribed text at the word level.
    phrases_texts: list
        List of transcribed text at the phrase level.

    Returns:
    ...........
    word_reps_perc: float
        The percentage of repeated words in the input lists.
    phrase_reps_perc: float
        The percentage of repeated phrases in the input lists.

    ------------------------------------------------------------------------------------------------------
    """
    # remove punctuation - lowercase
    words_texts = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in words_texts]
    phrases_texts = [phrase.translate(str.maketrans('', '', string.punctuation)).lower() for phrase in phrases_texts]

    # remove empty strings
    words_texts = [word for word in words_texts if word != '']
    phrases_texts = [phrase for phrase in phrases_texts if phrase != '']

    # calculate repetitions
    word_reps = len(words_texts) - len(set(words_texts))
    phrase_reps = len(phrases_texts) - len(set(phrases_texts))

    if len(phrases_texts) == 0:
        return 100*word_reps/len(words_texts), np.nan

    return 100*word_reps/len(words_texts), 100*phrase_reps/len(phrases_texts)


def get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures):
    """

    This function calculates the percentage of repeated words and phrases in the input text
    and adds them to the output dataframes.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    utterances_speaker_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker
        after filtering out turns with less than min_turn_length words.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    """
    
    word_df, turn_df, summ_df = df_list

    # turn-level
    if len(turn_df) > 0:
        for i in range(len(utterances_speaker_filtered)):
            row = utterances_speaker_filtered.iloc[i]
            words_texts = row[measures['words_texts']]
            phrases_texts = row[measures['phrases_texts']]

            word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

            turn_df.loc[i, measures['word_repeat_percentage']] = word_reps_perc
            turn_df.loc[i, measures['phrase_repeat_percentage']] = phrase_reps_perc

    # summary-level
    if len(turn_df) > 0:
        summ_df[measures['word_repeat_percentage']] = turn_df[measures['word_repeat_percentage']].mean(skipna=True)
        summ_df[measures['phrase_repeat_percentage']] = turn_df[measures['phrase_repeat_percentage']].mean(skipna=True)
    else:
        words_texts = [word for words in utterances_speaker[measures['words_texts']] for word in words]
        phrases_texts = [phrase for phrases in utterances_speaker[measures['phrases_texts']] for phrase in phrases]

        word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

        summ_df[measures['word_repeat_percentage']] = word_reps_perc
        summ_df[measures['phrase_repeat_percentage']] = phrase_reps_perc

    df_list = [word_df, turn_df, summ_df]
    return df_list

def get_word_embeddings(word_list, tokenizer, model):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the word embeddings for the input text using BERT.

    Parameters:
    ...........
    word_list: list
        List of transcribed text at the word level.
    language: str
        Language of the transcribed text.

    Returns:
    ...........
    word_embeddings: numpy array
        The calculated word embeddings.

    ------------------------------------------------------------------------------------------------------
    """
    if len(word_list) >= 512:
        # split the text into chunks of 512 tokens
        word_list = [word_list[i:i+512] for i in range(0, len(word_list), 512)]
        word_embeddings = []
        for chunk in word_list:
            inputs = tokenizer(chunk, return_tensors='pt', padding=True)
            outputs = model(**inputs)
            word_embeddings.append(outputs.last_hidden_state.mean(1).detach().numpy())
        word_embeddings = np.concatenate(word_embeddings, axis=0)
    else:

        inputs = tokenizer(word_list, return_tensors='pt', padding=True)
        outputs = model(**inputs)
        # Average pooling of the hidden states
        word_embeddings = outputs.last_hidden_state.mean(1).detach().numpy()
    return word_embeddings

def get_word_coherence_utterance(row, tokenizer, model, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures for a single utterance.

    Parameters:
    ...........
    row: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    tokenizer: BertTokenizer
        A tokenizer object for BERT.
    model: BertModel
        A BERT model object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_coherence: list
        A list containing the calculated semantic similarity of each word to the immediately preceding word.
    word_coherence_5: list
        A list containing the calculated semantic similarity of each word in 5-words window.
    word_coherence_10: list
        A list containing the calculated semantic similarity of each word in 10-words window.
    word_word_variability: dict
        A dictionary containing the calculated word-to-word variability at k inter-word distances.

    ------------------------------------------------------------------------------------------------------
    """
    words_texts = row[measures['words_texts']]
    word_embeddings = get_word_embeddings(words_texts, tokenizer, model)
    similarity_matrix = cosine_similarity(word_embeddings)

    # calculate semantic similarity of each word to the immediately preceding word
    if len(words_texts) > 1:
        word_coherence = [np.nan] + [similarity_matrix[j, j-1] for j in range(1, len(words_texts))]
    else:
        word_coherence = [np.nan]*len(words_texts)

    # calculate semantic similarity of each word in 5-words window
    if len(words_texts) > 5:
        word_coherence_5 = [np.nan]*2 + [np.mean(similarity_matrix[j-2:j+3, j]) for j in range(2, len(words_texts)-2)] + [np.nan]*2
    else:
        word_coherence_5 = [np.nan]*len(words_texts)

    # calculate semantic similarity of each word in 10-words window
    if len(words_texts) > 10:
        word_coherence_10 = [np.nan]*5 + [np.mean(similarity_matrix[j-5:j+6, j]) for j in range(5, len(words_texts)-5)] + [np.nan]*5
    else:
        word_coherence_10 = [np.nan]*len(words_texts)

    # calculate word-to-word variability at k inter-word distances (for k from 2 to 10)
    # indicating semantic similarity between each word and the next following word at k inter-word distance
    word_word_variability = {}
    for k in range(2, 11):
        if len(words_texts) > k:
            word_word_variability[k] = [similarity_matrix[j, j+k] for j in range(len(words_texts)-k)] + [np.nan]*k
        else:
            word_word_variability[k] = [np.nan]*len(words_texts)

    return word_coherence, word_coherence_5, word_coherence_10, word_word_variability    

def get_word_coherence_summary(word_df, summ_df, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function summarizes the word coherence measures at the summary level.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    summ_df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    summ_df[measures['word_coherence_mean']] = word_df[measures['word_coherence']].mean(skipna=True)
    summ_df[measures['word_coherence_var']] = word_df[measures['word_coherence']].var(skipna=True)
    summ_df[measures['word_coherence_5_mean']] = word_df[measures['word_coherence_5']].mean(skipna=True)
    summ_df[measures['word_coherence_5_var']] = word_df[measures['word_coherence_5']].var(skipna=True)
    summ_df[measures['word_coherence_10_mean']] = word_df[measures['word_coherence_10']].mean(skipna=True)
    summ_df[measures['word_coherence_10_var']] = word_df[measures['word_coherence_10']].var(skipna=True)
    for k in range(2, 11):
        summ_df[measures[f'word_coherence_variability_{k}_mean']] = word_df[measures[f'word_coherence_variability_{k}']].mean(skipna=True)
        summ_df[measures[f'word_coherence_variability_{k}_var']] = word_df[measures[f'word_coherence_variability_{k}']].var(skipna=True)

    return summ_df

def get_word_coherence(df_list, utterances_speaker, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, turn_df, summ_df = df_list

    # model init
    if language in measures["english_langs"]:
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('bert-base-cased')
    elif language in measures["supported_langs_bert"]:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        logger.error(f"Language {language} not supported for word coherence analysis")
        return df_list


    # word-level
    word_coherence_list = []
    word_coherence_5_list = []
    word_coherence_10_list = []
    word_coherence_variability = {
        x: [] for x in range(2, 11)
    }
    for i in range(len(utterances_speaker)):
        row = utterances_speaker.iloc[i]

        word_coherence, word_coherence_5, word_coherence_10, word_word_variability = get_word_coherence_utterance(row, tokenizer, model, measures)

        word_coherence_list += word_coherence
        word_coherence_5_list += word_coherence_5
        word_coherence_10_list += word_coherence_10
        for k in range(2, 11):
            word_coherence_variability[k] += word_word_variability[k]
    
    word_df[measures['word_coherence']] = word_coherence_list
    word_df[measures['word_coherence_5']] = word_coherence_5_list
    word_df[measures['word_coherence_10']] = word_coherence_10_list
    for k in range(2, 11):
        word_df[measures[f'word_coherence_variability_{k}']] = word_coherence_variability[k]

    # summary-level
    summ_df = get_word_coherence_summary(word_df, summ_df, measures)

    df_list = [word_df, turn_df, summ_df]
    return df_list

def calculate_perplexity(text, model, tokenizer):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the pseudo-perplexity of the input text using BERT.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.
    model: BertForMaskedLM
        A BERT model.
    tokenizer: BertTokenizer
        A BERT tokenizer.

    Returns:
    ...........
    float
        The calculated pseudo-perplexity of the input text.

    ------------------------------------------------------------------------------------------------------
    """
    # Tokenize input text
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r'\s+', ' ', clean_text)

    tokens = tokenizer(clean_text, return_tensors='pt')
    input_ids = tokens.input_ids
    masked_input_ids = input_ids.clone()

    log_probs = []
    # Iterate over each token in the input
    for i in range(input_ids.size(1)):
        # filter so that all input_ids include the masked location +- 256 tokens
        if i < 256:
            start = 0
        else:
            start = i - 256
        if i > input_ids.size(1) - 256:
            end = input_ids.size(1)
        else:
            end = i + 256

        masked_input_ids[0, i] = tokenizer.mask_token_id
        
        input_ids2 = input_ids[:, start:end]
        masked_input_ids2 = masked_input_ids[:, start:end]

        with torch.no_grad():
            outputs = model(input_ids=masked_input_ids2, labels=input_ids2)
        
        # Calculate log probability of the original token
        logit_prob = outputs.logits[0, i].softmax(dim=0)
        true_log_prob = logit_prob[input_ids[0, i]].log().item()
        log_probs.append(true_log_prob)
        
        # Unmask the token for the next iteration
        masked_input_ids[0, i] = input_ids[0, i]
    
    # Calculate perplexity
    perplexity = np.exp(-np.mean(log_probs))
    return perplexity

def calculate_phrase_tangeniality(phrases_texts, utterance_text, sentence_encoder, bert, tokenizer):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the semantic similarity of each phrase to the immediately preceding phrase,
    the semantic similarity of each phrase to the phrase 2 turns before, and the pseudo-perplexity of the turn.

    Parameters:
    ...........
    phrases_texts: list
        List of transcribed text at the phrase level.
    utterance_text: str
        The full transcribed text.
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    bert: BertForMaskedLM
        A BERT model.
    tokenizer: BertTokenizer
        A BERT tokenizer.

    Returns:
    ...........
    sentence_tangeniality1: float
        The semantic similarity of each phrase to the immediately preceding phrase.
    sentence_tangeniality2: float
        The semantic similarity of each phrase to the phrase 2 turns before.
    perplexity: float
        The pseudo-perplexity of the turn.

    ------------------------------------------------------------------------------------------------------
    """
    if sentence_encoder is not None and len(phrases_texts) > 0:
        phrase_embeddings = sentence_encoder.encode(phrases_texts)
        similarity_matrix = cosine_similarity(phrase_embeddings)

        # calculate semantic similarity of each phrase to the immediately preceding phrase
        if len(phrases_texts) > 1:
            sentence_tangeniality1 = np.mean([similarity_matrix[j, j-1] for j in range(1, len(phrases_texts))])
        else:
            sentence_tangeniality1 = np.nan

        # calculate semantic similarity of each phrase to the phrase 2 turns before
        if len(phrases_texts) > 2:
            sentence_tangeniality2 = np.mean([similarity_matrix[j-2, j] for j in range(2, len(phrases_texts))])
        else:
            sentence_tangeniality2 = np.nan
    else:
        sentence_tangeniality1 = np.nan
        sentence_tangeniality2 = np.nan

    if tokenizer is not None and bert is not None:
        # calculate pseudo-perplexity of the turn and indicating how predictable the turn is
        perplexity = calculate_perplexity(utterance_text, bert, tokenizer)
    else:
        perplexity = np.nan

    return sentence_tangeniality1, sentence_tangeniality2, perplexity

def calculate_slope(y):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the slope
     of the input list using linear regression

    Parameters:
    ...........
    y: list
        A list of values.

    Returns:
    ...........
    float
        The calculated slope of the input list.

    ------------------------------------------------------------------------------------------------------
    """

    x = range(len(y))
    slope, _ = np.polyfit(x, y, 1)

    return slope


def get_phrase_coherence(df_list, utterances_filtered, speaker_label, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    speaker_label: str
        Speaker label
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, turn_df, summ_df = df_list

    if language in measures["english_langs"]:
        sentence_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        bert = BertForMaskedLM.from_pretrained('bert-base-cased')
    else:
        if language not in measures["supported_langs_sentence_embeddings"] + measures["supported_langs_bert"]:
            logger.error(f"Language {language} not supported for phrase coherence nor perplexity analysis")
            return df_list

        if language in measures["supported_langs_sentence_embeddings"]:
            sentence_encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        else:
            sentence_encoder = None
            logger.error(f"Language {language} not supported for phrase coherence analysis")

        if language in measures["supported_langs_bert"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            bert = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        else:
            tokenizer = None
            bert = None
            logger.error(f"Language {language} not supported for perplexity analysis")


    # turn-level
    if len(turn_df) > 0:
        sentence_tangeniality1_list = []
        sentence_tangeniality2_list = []
        perplexity_list = []
        for i in range(len(utterances_filtered)):
            row = utterances_filtered.iloc[i]
            current_speaker = row[measures['speaker_label']]

            if current_speaker == speaker_label:
                phrases_texts = row[measures['phrases_texts']]
                utterance_text = row[measures['utterance_text']]
                
                sentence_tangeniality1, sentence_tangeniality2, perplexity = calculate_phrase_tangeniality(
                    phrases_texts, utterance_text, sentence_encoder, bert, tokenizer
                )
                
                sentence_tangeniality1_list.append(sentence_tangeniality1)
                sentence_tangeniality2_list.append(sentence_tangeniality2)
                perplexity_list.append(perplexity)

        ## semantic similarity of current turn to previous turn of the other speaker
        utterances_texts = utterances_filtered[measures['utterance_text']].values.tolist()
        utterances_embeddings = sentence_encoder.encode(utterances_texts)
        similarity_matrix = cosine_similarity(utterances_embeddings)

        ## get the indices of the turns of our speaker
        speaker_indices = utterances_filtered[utterances_filtered[measures['speaker_label']] == speaker_label].index.tolist()
        if speaker_indices[0] == 0:
            turn_to_turn_tangeniality_list = [np.nan] + [similarity_matrix[i, i-1] for i in speaker_indices[1:]]
        else:
            turn_to_turn_tangeniality_list = [similarity_matrix[i, i-1] for i in speaker_indices]

        turn_df[measures['sentence_tangeniality1']] = sentence_tangeniality1_list
        turn_df[measures['sentence_tangeniality2']] = sentence_tangeniality2_list
        turn_df[measures['perplexity']] = perplexity_list
        turn_df[measures['turn_to_turn_tangeniality']] = turn_to_turn_tangeniality_list


    # summary-level
    if len(turn_df) > 0:
        summ_df[measures['sentence_tangeniality1_mean']] = turn_df[measures['sentence_tangeniality1']].mean(skipna=True)
        summ_df[measures['sentence_tangeniality1_var']] = turn_df[measures['sentence_tangeniality1']].var(skipna=True)
        summ_df[measures['sentence_tangeniality2_mean']] = turn_df[measures['sentence_tangeniality2']].mean(skipna=True)
        summ_df[measures['sentence_tangeniality2_var']] = turn_df[measures['sentence_tangeniality2']].var(skipna=True)
        summ_df[measures['perplexity_mean']] = turn_df[measures['perplexity']].mean(skipna=True)
        summ_df[measures['perplexity_var']] = turn_df[measures['perplexity']].var(skipna=True)
        summ_df[measures['turn_to_turn_tangeniality_mean']] = turn_df[measures['turn_to_turn_tangeniality']].mean(skipna=True)
        summ_df[measures['turn_to_turn_tangeniality_var']] = turn_df[measures['turn_to_turn_tangeniality']].var(skipna=True)
        summ_df[measures['turn_to_turn_tangeniality_slope']] = calculate_slope(turn_df[measures['turn_to_turn_tangeniality']])

    df_list = [word_df, turn_df, summ_df]
    return df_list

def calculate_file_feature(json_data, model, speakers):
    """
    ------------------------------------------------------------------------------------------------------

    Calculate file features based on JSON data.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    model: str
        model name (vosk/aws/whisper)
    speakers: str
        speakers label

    Returns:
    ...........
    tuple: A tuple containing two values - the total file length and the percentage of time spent speaking.

    ------------------------------------------------------------------------------------------------------
    """
    
    if model == 'aws':
        segments = json_data.get('items', [])
        file_length = max(float(segment.get("end_time", "0")) for segment in segments)
        
        if speakers is None:
            return file_length/60, np.NaN

        speaking_time = sum(float(segment.get("end_time", "0") or "0") - float(segment.get("start_time", "0") or "0")
                           for segment in segments if segment.get("speaker_label", "") in speakers)
    else:
        segments = json_data.get('segments', [])
        file_length = max(segment.get('end', 0) for segment in segments)
        
        if speakers is None:
            return file_length/60, np.NaN
        speaking_time = sum(segment['end'] - segment['start'] for segment in segments if segment.get('speaker', '') in speakers)

    speaking_pct = (speaking_time / file_length) * 100
    return file_length/60, speaking_pct

def create_text_list(utterances_speaker, speaker_label, min_turn_length, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates a list of transcribed text at the word, turn, and full text levels,
        and a list of the indices of the first and last word in each turn.

    Parameters:
    ...........
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
         for the specified speaker.
    speaker_label: str
        Speaker label
    min_turn_length: int
        minimum words required in each turn
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    text_list: list
        List of transcribed text at the word, turn, and full text levels.
    turn_indices: list
        A list containing the indices of the first and last word in each turn.

    ------------------------------------------------------------------------------------------------------

    """

    word_list = []
    turn_list = []
    text = ""
    turn_indices = []
    for i in range(len(utterances_speaker)):
        row = utterances_speaker.iloc[i]

        utterance_text = row[measures['utterance_text']]
        words_texts = row[measures['words_texts']]
        utterance_ids = row[measures['utterance_ids']]

        text += " " + utterance_text
        word_list += words_texts

        if speaker_label is not None and len(words_texts) >= min_turn_length:
            turn_list.append(utterance_text)
            turn_indices.append(utterance_ids)

    text_list = [word_list, turn_list, text]

    return text_list, turn_indices

def process_language_feature(df_list, transcribe_info, speaker_label, min_turn_length, language, time_index, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the language features from json response.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    transcribe_info: list
        transcribed info
    min_turn_length: int
        minimum words required in each turn
    speaker_label: str
        Speaker label
    time_index: list
        timepoint index (start/end)
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of pandas dataframes (word_df, turn_df and summ_df)

    ------------------------------------------------------------------------------------------------------
    """
    json_conf, utterances = transcribe_info
    if speaker_label is not None:
        utterances_speaker = utterances[utterances[measures['speaker_label']] == speaker_label]
        json_conf_speaker = [item for item in json_conf if item.get("speaker_label", "") == speaker_label or item.get("speaker", "") == speaker_label]

        if len(utterances_speaker) <= 0:
            logger.error(f"No utterances found for speaker {speaker_label}")
            return df_list
    else:
        utterances_speaker = utterances.copy()
        json_conf_speaker = json_conf.copy()

    text_list, turn_indices = create_text_list(utterances_speaker, speaker_label, min_turn_length, measures)
    if speaker_label is not None and len(turn_indices) <= 0:
        logger.error(f"No utterances found for speaker {speaker_label} with minimum length {min_turn_length}")
        return df_list
    
    # filter utterances with minimum length
    utterances_speaker_filtered = utterances_speaker[utterances_speaker[measures['words_texts']].apply(lambda x: len(x) >= min_turn_length)].reset_index(drop=True)
    # filter utterances with minimum length for speaker
    if speaker_label is not None:
        utterances_filtered = utterances.copy().iloc[0:0]
        for i in range(len(utterances)):
            if utterances.iloc[i][measures['speaker_label']] != speaker_label:
                utterances_filtered = pd.concat([utterances_filtered, utterances.iloc[i:i+1]])
            else:
                if len(utterances.iloc[i][measures['words_texts']]) >= min_turn_length:
                    utterances_filtered = pd.concat([utterances_filtered, utterances.iloc[i:i+1]])

        utterances_filtered = utterances_filtered.reset_index(drop=True)
    else:
        utterances_filtered = utterances.copy()

    df_list = get_pause_feature(json_conf_speaker, df_list, text_list, turn_indices, measures, time_index, language)
    df_list = get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures)
    df_list = get_word_coherence(df_list, utterances_speaker, language, measures)
    df_list = get_phrase_coherence(df_list, utterances_filtered, speaker_label, language, measures)

    if language in measures["english_langs"]:
        df_list = get_sentiment(df_list, text_list, measures)
        df_list = get_pos_tag(df_list, text_list, measures)

    return df_list
