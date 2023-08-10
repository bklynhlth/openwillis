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
TAG_DICT = {
    "PRP": "Pronoun",
    "PRP$": "Pronoun",
    "VB": "Verb",
    "VBD": "Verb",
    "VBG": "Verb",
    "VBN": "Verb",
    "VBP": "Verb",
    "VBZ": "Verb",
    "JJ": "Adjective",
    "JJR": "Adjective",
    "JJS": "Adjective",
    "NN": "Noun",
    "NNP": "Noun",
    "NNS": "Noun",
}


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
        # local vosk transcriber
        word_list = [word["word"] for word in json_conf if "word" in word]
    else:
        # aws transcriber
        word_list = [item["alternatives"][0]["content"] for item in json_conf]

    tag_list = nltk.pos_tag(word_list)

    for i, tag in enumerate(tag_list):
        if tag[1] in tag_dict.keys():
            json_conf[i][measures["tag"]] = tag_dict[tag[1]]
        else:
            json_conf[i][measures["tag"]] = "Other"

    return json_conf


def get_part_of_speech(df, tags, measures, index=0):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the proportions of verbs,
     pronouns, adjectives, and nouns in the
     transcribed text, and adds them to the output dataframe df.

    Parameters:
    ...........
    df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    tags: list
        A list of part-of-speech tags for the input text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    index: int
        The index of the row in the output dataframe df.

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    if len(tags) == 0:
        return df

    df.loc[index, measures["speech_noun"]] = (
        100 * len(tags[tags == "Noun"]) / len(tags)
    )
    df.loc[index, measures["speech_verb"]] = (
        100 * len(tags[tags == "Verb"]) / len(tags)
    )
    df.loc[index, measures["speech_adj"]] = (
        100 * len(tags[tags == "Adjective"]) / len(tags)
    )
    df.loc[index, measures["speech_pronoun"]] = (
        100 * len(tags[tags == "Pronoun"]) / len(tags)
    )

    return df


def get_tag_summ(json_conf, df_list, text_indices, measures):
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
        List of pandas dataframes.
            word_df, phrase_df, turn_df, summ_df
    text_indices: list
        List of indices for text_list.
            for phrases and turns.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """

    word_df, phrase_df, turn_df, summ_df = df_list
    phrase_index, turn_index = text_indices

    df_conf = pd.DataFrame(json_conf)

    # word-level analysis
    word_df[measures["part_of_speech"]] = df_conf[measures["tag"]]

    # phrase-level analysis
    for j, pindex in enumerate(phrase_index):
        prange = range(pindex[0], pindex[1] + 1)
        phrase_tags = df_conf.loc[df_conf[measures["old_index"]].isin(prange), measures["tag"]]

        phrase_df = get_part_of_speech(phrase_df, phrase_tags, measures, j)

    # turn-level analysis
    for j, uindex in enumerate(turn_index):
        urange = range(uindex[0], uindex[1] + 1)
        turn_tags = df_conf.loc[df_conf[measures["old_index"]].isin(urange), measures["tag"]]

        turn_df = get_part_of_speech(turn_df, turn_tags, measures, j)

    # file-level analysis
    summ_df = get_part_of_speech(summ_df, df_conf[measures["tag"]], measures)

    df_list = [word_df, phrase_df, turn_df, summ_df]

    return df_list


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


def get_sentiment(df_list, text_list, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
            word_df, phrase_df, turn_df, summ_df
    text_list: list
        List of transcribed text.
            split into words, phrases, turns, and full text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    word_df, phrase_df, turn_df, summ_df = df_list
    word_list, phrase_list, turn_list, full_text = text_list

    sentiment = SentimentIntensityAnalyzer()

    # column names
    cols = [
        measures["neg"],
        measures["neu"],
        measures["pos"],
        measures["compound"],
        measures["speech_mattr"],
    ]

    # word-level analysis
    for idx, w in enumerate(word_list):
        try:
            sentiment_dict = sentiment.polarity_scores(w)

            word_df.loc[idx, cols[:-1]] = list(sentiment_dict.values())
        except Exception as e:
            logger.error(f"Error in sentiment analysis for word {w}: {e}")
            continue

    # phrase-level analysis
    for idx, p in enumerate(phrase_list):
        try:
            sentiment_dict = sentiment.polarity_scores(p)
            mattr = get_mattr(p)

            phrase_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]
        except Exception as e:
            logger.error(f"Error in sentiment analysis for phrase {p}: {e}")
            continue

    # turn-level analysis
    for idx, u in enumerate(turn_list):
        try:
            sentiment_dict = sentiment.polarity_scores(u)
            mattr = get_mattr(u)

            turn_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]
        except Exception as e:
            logger.error(f"Error in sentiment analysis for turn {u}: {e}")
            continue

    # file-level analysis
    sentiment_dict = sentiment.polarity_scores(full_text)
    mattr = get_mattr(full_text)

    summ_df.loc[0, cols] = list(sentiment_dict.values()) + [mattr]

    df_list = [word_df, phrase_df, turn_df, summ_df]

    return df_list


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

    # remove punctuation
    punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
    syllables = [syllable_tokenizer.tokenize(token) for token in nltk.word_tokenize(text) if token not in punctuation]
    # count the number of syllables in each word
    syllable_count = sum([len(token) for token in syllables])

    return syllable_count


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
    if len(phrase_df) > 0:
        speech_minutes = phrase_df[measures["phrase_minutes"]].sum()
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


def process_language_feature(
    json_conf, df_list, text_list,
    text_indices, language, time_index, measures,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the language features from json response.

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
    language: str
        Language of the transcribed text.
    time_index: list
        A list containing the names of the columns in json that contain the
         start and end times of each word.
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

    df_list = get_pause_feature(json_conf, df_list, text_list, text_indices, time_index, measures)

    if language == "en-us":
        json_conf = get_tag(json_conf, TAG_DICT, measures)
        df_list = get_tag_summ(json_conf, df_list, text_indices, measures)

        df_list = get_sentiment(df_list, text_list, measures)

    word_df, phrase_df, turn_df, summ_df = df_list
    return word_df, phrase_df, turn_df, summ_df
