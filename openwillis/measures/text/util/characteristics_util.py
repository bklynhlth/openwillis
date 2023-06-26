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


def get_tag(json_conf, tag_dict):
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
            json_conf[i]["tag"] = tag_dict[tag[1]]
        else:
            json_conf[i]["tag"] = "Other"

    return json_conf


def get_part_of_speech(df, tags, index=0):
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
    index: int
        The index of the row in the output dataframe df.

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    df.loc[index, "noun_percentage"] = (
        100 * len(tags[tags == "Noun"]) / len(tags)
    )
    df.loc[index, "verb_percentage"] = (
        100 * len(tags[tags == "Verb"]) / len(tags)
    )
    df.loc[index, "adjective_percentage"] = (
        100 * len(tags[tags == "Adjective"]) / len(tags)
    )
    df.loc[index, "pronoun_percentage"] = (
        100 * len(tags[tags == "Pronoun"]) / len(tags)
    )

    return df


def get_tag_summ(json_conf, df_list, text_indices):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the proportions of verbs,
     pronouns, adjectives, and nouns in the
     transcribed text, and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    tag_df: pandas dataframe
        A dataframe containing the part-of-speech tags for the input text.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    word: list
        The input text as a list of words.
    measures: dict
        A dictionary containing the names
         of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    word_df, phrase_df, turn_df, summ_df = df_list
    phrase_index, turn_index = text_indices

    df_conf = pd.DataFrame(json_conf)

    # word-level analysis
    word_df["part_of_speech"] = df_conf["tag"]

    # phrase-level analysis
    for j, pindex in enumerate(phrase_index):
        prange = range(pindex[0], pindex[1] + 1)
        phrase_tags = df_conf.loc[df_conf["old_idx"].isin(prange), "tag"]

        phrase_df = get_part_of_speech(phrase_df, phrase_tags, j)

    # turn-level analysis
    for j, uindex in enumerate(turn_index):
        urange = range(uindex[0], uindex[1] + 1)
        turn_tags = df_conf.loc[df_conf["old_idx"].isin(urange), "tag"]

        turn_df = get_part_of_speech(turn_df, turn_tags, j)

    # file-level analysis
    summ_df = get_part_of_speech(summ_df, df_conf["tag"])

    df_list = [word_df, phrase_df, turn_df, summ_df]

    return df_list


def get_mattr(word):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR)
     of the input text using the
     LexicalRichness library.

    Parameters:
    ...........
    word : list
        The input text as a list of words.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
    """
    filter_punc = list(value for value in word if value not in [".", "!", "?"])
    filter_punc = " ".join(str(filter_punc))
    mattr = np.nan

    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=lex_richness.words)

    return mattr


def get_sentiment(df_list, text_list):
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
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_pos",
        "sentiment_overall",
        "mattr",
    ]

    # word-level analysis
    for idx, w in enumerate(word_list):
        sentiment_dict = sentiment.polarity_scores(w)
        mattr = get_mattr(w)

        word_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]

    # phrase-level analysis
    for idx, p in enumerate(phrase_list):
        sentiment_dict = sentiment.polarity_scores(p)
        mattr = get_mattr(p)

        phrase_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]

    # turn-level analysis
    for idx, u in enumerate(turn_list):
        sentiment_dict = sentiment.polarity_scores(u)
        mattr = get_mattr(u)

        turn_df.loc[idx, cols] = list(sentiment_dict.values()) + [mattr]

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


def process_pause_feature(df_diff, df, text_level, index_list, time_index, level_name):
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

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    if level_name not in ["phrase", "turn"]:
        logger.error("level_name must be either phrase or turn")
        return df

    for j, index in enumerate(index_list):
        rng = range(index[0], index[1] + 1)
        level_json = df_diff[df_diff["old_idx"].isin(rng)]

        # remove first pause as it is the pre_pause
        pauses = level_json["pause_diff"].values[1:]

        df.loc[j, f"{level_name}_length_minutes"] = (
            float(level_json.iloc[-1][time_index[1]])
            - float(level_json.iloc[0][time_index[0]])
        ) / 60
        df.loc[j, f"{level_name}_length_words"] = len(level_json)

        df.loc[j, "pause_variability"] = np.var(pauses)
        df.loc[j, "mean_pause_length"] = np.mean(pauses)
        df.loc[j, "speech_percentage"] = 100 * (
            1 - np.sum(pauses) / (
                60 * df.loc[j, f"{level_name}_length_minutes"]
            )
        )

        # articulation rate
        df.loc[j, "syllables_per_min"] = (
            get_num_of_syllables(text_level[j]) / df.loc[j, f"{level_name}_length_minutes"]
        )

    df["words_per_min"] = (
        df[f"{level_name}_length_words"] / df[f"{level_name}_length_minutes"]
    )
    df["pauses_per_min"] = df["words_per_min"]

    return df


def update_summ_df(
    df_diff, summ_df, full_text, time_index, word_df, phrase_df, turn_df
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

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    summ_df["speech_length_minutes"] = [
        (
            float(df_diff.iloc[-1][time_index[1]])
            - float(df_diff.iloc[0][time_index[0]])
        ) / 60
    ]
    summ_df["speech_length_words"] = len(df_diff)
    summ_df["words_per_min"] = (
        summ_df["speech_length_words"] / summ_df["speech_length_minutes"]
    )
    summ_df["syllables_per_min"] = (
        get_num_of_syllables(full_text) / summ_df["speech_length_minutes"]
    )
    summ_df["pauses_per_min"] = summ_df["words_per_min"]
    summ_df["word_pause_length_mean"] = word_df["pre_word_pause"].mean(
        skipna=True
    )
    summ_df["word_pause_variability"] = word_df["pre_word_pause"].var(
        skipna=True
    )
    summ_df["phrase_pause_length_mean"] = phrase_df["pre_phrase_pause"].mean(
        skipna=True
    )
    summ_df["phrase_pause_variability"] = phrase_df["pre_phrase_pause"].var(
        skipna=True
    )
    summ_df["speech_percentage"] = 100 * (
        1
        - df_diff.loc[1:, "pause_diff"].sum()
        / (60 * summ_df["speech_length_minutes"])
    )
    if len(turn_df) > 0:
        summ_df["num_turns"] = len(turn_df)
        summ_df["mean_turn_length_minutes"] = turn_df[
            "turn_length_minutes"
        ].mean()
        summ_df["mean_turn_length_words"] = turn_df[
            "turn_length_words"
        ].mean()
        summ_df["mean_pre_turn_pause"] = turn_df[
            "pre_turn_pause"
        ].mean(skipna=True)
        summ_df["num_one_word_turns"] = len(
            turn_df[turn_df["turn_length_words"] == 1]
        )

    return summ_df


def get_pause_feature(json_conf, df_list, text_list, text_indices, time_index):
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
    if "pause_diff" not in df_diff.columns:
        df_diff["pause_diff"] = df_diff[time_index[0]].astype(float) - df_diff[
            time_index[1]
        ].astype(float).shift(1)

    # word-level analysis
    phrase_starts = [pindex[0] for pindex in phrase_index]
    word_df["pre_word_pause"] = df_diff["pause_diff"].where(
        ~df_diff["old_idx"].isin(phrase_starts), np.nan
    )
    # calculate the number of syllables in each word from the word list
    word_df["num_syllables"] = [
        get_num_of_syllables(word) for word in word_list
    ]

    # phrase-level analysis
    df_diff_phrase = df_diff[
        df_diff["old_idx"].isin(phrase_starts)
    ]  # get the rows corresponding to the start of each phrase

    if len(turn_index) > 0:
        turn_starts = [
            uindex[0] for uindex in turn_index
        ]  # get the start index of each turn
        phrase_df["pre_phrase_pause"] = df_diff_phrase["pause_diff"].where(
            ~df_diff_phrase["old_idx"].isin(turn_starts), np.nan
        )
    else:
        phrase_df["pre_phrase_pause"] = df_diff_phrase["pause_diff"]
    phrase_df = phrase_df.reset_index(drop=True)

    phrase_df = process_pause_feature(
        df_diff, phrase_df, phrase_list, phrase_index, time_index, "phrase"
    )

    # turn-level analysis
    if len(turn_index) > 0:
        df_diff_turn = df_diff[
            df_diff["old_idx"].isin(turn_starts)
        ]  # get the rows corresponding to the start of each turn

        turn_df["pre_turn_pause"] = df_diff_turn["pause_diff"]
        turn_df = turn_df.reset_index(drop=True)

        turn_df = process_pause_feature(
            df_diff, turn_df, turn_list, turn_index, time_index, "turn"
        )

    # file-level analysis
    summ_df = update_summ_df(
        df_diff, summ_df, full_text, time_index, word_df, phrase_df, turn_df
    )

    df_feature = [word_df, phrase_df, turn_df, summ_df]

    return df_feature


def process_language_feature(
    json_conf, df_list, text_list, text_indices, language, time_index
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
    time_index: list
        A list containing the names of the columns in json that contain the
         start and end times of each word.

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

    df_list = get_pause_feature(json_conf, df_list, text_list, text_indices, time_index)

    if language == "en-us":
        json_conf = get_tag(json_conf, TAG_DICT)
        df_list = get_tag_summ(json_conf, df_list, text_indices)

        df_list = get_sentiment(df_list, text_list)

    word_df, phrase_df, turn_df, summ_df = df_list
    return word_df, phrase_df, turn_df, summ_df
