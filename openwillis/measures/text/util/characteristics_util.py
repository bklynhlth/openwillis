# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import pandas as pd
import numpy as np
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# NLTK Tag list
TAG_DICT = {"PRP": "Pronoun", "PRP$": "Pronoun", "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", 
            "VBZ": "Verb", "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective", "NN": "Noun", "NNP": "Noun", "NNS": "Noun"}

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

    word_df = pd.DataFrame(columns=[measures["word_pause"], measures["num_syllables"], measures["part_of_speech"]])
    turn_df = pd.DataFrame(columns=[measures["turn_pause"], measures["turn_minutes"], measures["turn_words"], 
                                    measures["word_rate"], measures["syllable_rate"], measures["speech_percentage"], 
                                    measures["pause_meandur"], measures["pause_var"], measures["pos"], measures["neg"], 
                                    measures["neu"], measures["compound"], measures["speech_mattr"], 
                                    measures["interrupt_flag"]])

    summ_df = pd.DataFrame(
        columns=[measures["file_length"], measures["speech_minutes"], measures["speech_words"], measures["word_rate"],
                 measures["syllable_rate"], measures["word_pause_mean"], measures["word_pause_var"], 
                 measures["speech_percentage"], measures["pos"], measures["neg"], measures["neu"], measures["compound"], 
                 measures["speech_mattr"], measures["num_turns"], measures["num_one_word_turns"], measures["turn_minutes_mean"],
                 measures["turn_words_mean"], measures["turn_pause_mean"], measures["speaker_percentage"], 
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

def filter_turn_aws(item_data, min_turn_length, speaker_label):
    """
    ------------------------------------------------------------------------------------------------------
    
    This function updates the turns list
        to only include the speaker label provided.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    min_turn_length: int
        minimum words required in each turn
    speaker_label: str
        Speaker label
        
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
    turns_idxs, turns = [], []
    for i, item in enumerate(item_data):
        
        try:
            if (i > 0 and item.get("speaker_label", "") == speaker_label and item_data[i - 1].get("speaker_label", "") != speaker_label):
                start_idx = i
            
            elif (i > 0 and item.get("speaker_label", "") != speaker_label and item_data[i - 1].get("speaker_label", "") == speaker_label):
                turn_text = " ".join([item["alternatives"][0]["content"] for item in item_data[start_idx:i]])

                if len(turn_text.split(" ")) >= min_turn_length:
                    turns_idxs.append((start_idx, i - 1))
                    turns.append(turn_text)
                
        except Exception as e:
            logger.error(f"Error in turn-split for speaker {speaker_label}: {e}")
            continue

    if start_idx not in [item[0] for item in turns_idxs]:
        turn_text = " ".join([item["alternatives"][0]["content"] for item in item_data[start_idx:]])

        if len(turn_text.split(" ")) >= min_turn_length:
            turns_idxs.append((start_idx, len(item_data) - 1))

            turns.append(turn_text)
    return turns_idxs, turns

def filter_speaker_aws(item_data, min_turn_length, speaker_label):
    """
    ------------------------------------------------------------------------------------------------------

    This function updates the turns lists to only include the speaker label provided.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    min_turn_length: int
        minimum words required in each turn
    speaker_label: str
        Speaker label

    Returns:
    ...........
    turns_idxs: list
        A list of tuples containing the start and end indices of the turns in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """

    speaker_labels = [item["speaker_label"] for item in item_data if "speaker_label" in item]

    if speaker_label not in speaker_labels:
        logger.error(f"Speaker label {speaker_label} not found in the json response object.")

    turns_idxs, turns = filter_turn_aws(item_data, min_turn_length, speaker_label)
    return turns_idxs, turns

def filter_json_transcribe_aws(item_data, speaker_label, measures):
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
    filter_json = [item for item in item_data if "start_time" in item and "end_time" in item]
    filter_json = pause_calculation(filter_json, measures, ['start_time', 'end_time'])

    if speaker_label is not None:
        filter_json = [item for item in filter_json if item.get("speaker_label", "") == speaker_label]

    return filter_json

def filter_turns(item_data, speaker_label, measures, min_turn_length):
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
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    min_turn_length: int
        minimum words required in each turn

    Returns:
    ...........
    turns_idxs: list
        A list of tuples containing
            the start and end indices of the turns in the JSON object.
    turns: list
        A list of turns extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    turns_idxs, turns = [], []
    current_turn = None

    for item in item_data:
        try:
            
            if "speaker" in item:
                if item["speaker"] == speaker_label:
                    current_turn = [item] if current_turn is None else current_turn + [item]
                    
                else:
                    if current_turn is not None:
                        
                        if len(current_turn)>0 and len(current_turn[0]["words"])>0: 
                            start_idx2 = current_turn[0]["words"][0][measures["old_index"]]
                            
                            end_idx2 = current_turn[-1]["words"][-1][measures["old_index"]]
                            turn_text = " ".join(item["text"] for item in current_turn)
                            
                            if len(turn_text.split(" ")) >= min_turn_length:
                                turns_idxs.append((start_idx2, end_idx2))

                                turns.append(turn_text)
                        current_turn = None
                        
        except Exception as e:
            logger.error(f"Error in turn calculation {e}")
    
    if current_turn is not None:
        start_idx2 = current_turn[0]["words"][0][measures["old_index"]]
        
        end_idx2 = current_turn[-1]["words"][-1][measures["old_index"]]
        turn_text = " ".join(item["text"] for item in current_turn)
        
        if len(turn_text.split(" ")) >= min_turn_length: 
            turns_idxs.append((start_idx2, end_idx2))
            
            turns.append(turn_text)
    return turns_idxs, turns

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

def filter_json_transcribe(item_data, speaker_label, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the JSON response object to only include items with start and end time.

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
    item_data2 = []
    for item in item_data:
        try:
            
            speaker = item["speaker"]
            words = item["words"]
            
            for j, w in enumerate(words):# update speaker labels
                words[j]["speaker"] = speaker
            
            item_data2 += words
        except Exception as e:
            logger.error(f"Failed to filter word: {e}")
    
    filter_json = [item for item in item_data2 if "start" in item and "end" in item]
    filter_json = pause_calculation(filter_json, measures, ['start', 'end'])

    if speaker_label is not None:
        filter_json = [item for item in filter_json if item.get("speaker", "") == speaker_label]
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

                if language == 'en':
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
        
        summ_df["num_one_word_turns"] = len(turn_df[turn_df[measures["turn_words"]] == 1])
        summ_df[measures["num_interrupts"]] = len(turn_df[turn_df[measures["interrupt_flag"]]==True])

    return summ_df

def get_pause_feature(json_conf, df_list, text_list, text_indices, measures, time_index, language):
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
    text_indices: list
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

    turn_index, turn_index2 = text_indices

    word_df, turn_df, summ_df = df_list
    word_list, turn_list, full_text = text_list
    df_diff = pd.DataFrame(json_conf)

    # Calculate the pause time between; each word and add the results to pause_list
    if measures["pause"] not in df_diff.columns:
        df_diff[measures["pause"]] = df_diff[time_index[0]].astype(float) - df_diff[time_index[1]].astype(float).shift(1)

    # word-level analysis
    word_df = get_pause_feature_word(word_df, df_diff, word_list, turn_index2, measures)

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

def get_tag(json_conf, tag_dict, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech
     tagging on the input text using NLTK, and returns an updated
     json_conf list with the part-of-speech tags.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    tag_dict: dict
        A dictionary mapping the NLTK tags to more readable tags.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    json_conf: list
        The updated json_conf list.

    ------------------------------------------------------------------------------------------------------
    """
    if len(json_conf) <= 0:
        return json_conf

    if "alternatives" not in json_conf[0].keys():
        word_list = [word["word"] for word in json_conf if "word" in word]# local vosk transcriber
    
    else:
        word_list = [item["alternatives"][0]["content"] for item in json_conf]# aws transcriber

    tag_list = nltk.pos_tag(word_list)
    for i, tag in enumerate(tag_list):
        
        if tag[1] in tag_dict.keys():
            json_conf[i][measures["tag"]] = tag_dict[tag[1]]
        
        else:
            json_conf[i][measures["tag"]] = "Other"
    return json_conf

def get_tag_summ(json_conf, df_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the proportions of verbs,
     pronouns, adjectives, and nouns in the
     transcribed text, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes: word_df, turn_df, summ_df
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, turn_df, summ_df = df_list
    df_conf = pd.DataFrame(json_conf)
    word_df[measures["part_of_speech"]] = df_conf[measures["tag"]]
    
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

def process_language_feature(df_list, transcribe_info, language, time_index, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the language features from json response.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    transcribe_info: list
        transcribed info
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
    json_conf, text_list, text_indices = transcribe_info
    df_list = get_pause_feature(json_conf, df_list, text_list, text_indices, measures, time_index, language)

    if language == "en":
        json_conf = get_tag(json_conf, TAG_DICT, measures)
        df_list = get_tag_summ(json_conf, df_list, measures)

        df_list = get_sentiment(df_list, text_list, measures)
    return df_list
